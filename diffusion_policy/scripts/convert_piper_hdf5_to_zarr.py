import os
import pathlib
import shutil

import click
import cv2
import h5py
import numcodecs
import numpy as np
import zarr
from tqdm import tqdm


DEFAULT_INPUT = "/home/gx4070/Desktop/arm-datasets-collect/data/piper_xy"
DEFAULT_OUTPUT = "/home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr"


def _make_valid_mask(f: h5py.File) -> np.ndarray:
    valid = np.ones(f["action"].shape[0], dtype=bool)
    for key in [
        "valid/wrist_camera",
        "valid/global_camera",
        "valid/robot_feedback",
        "valid/action",
    ]:
        valid &= f[key][:].astype(bool)
    valid &= np.all(np.isfinite(f["action"][:]), axis=-1)
    valid &= np.all(np.isfinite(f["observations/qpos"][:]), axis=-1)
    valid &= np.all(np.isfinite(f["observations/eef_pose"][:]), axis=-1)
    return valid


def _resize_images(src, out_h: int, out_w: int) -> np.ndarray:
    result = np.empty((src.shape[0], out_h, out_w, 3), dtype=np.uint8)
    for i in range(src.shape[0]):
        result[i] = cv2.resize(src[i], (out_w, out_h), interpolation=cv2.INTER_AREA)
    return result


def _write_array(root, name, data, chunks, compressor):
    root.create_dataset(
        name,
        data=data,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        compressor=compressor,
        overwrite=True,
    )


def _verify_episode(zarr_path: pathlib.Path, expected_t: int, out_h: int, out_w: int):
    root = zarr.open(str(zarr_path), mode="r")
    expected = {
        "camera_wrist": ((expected_t, out_h, out_w, 3), np.dtype("uint8")),
        "camera_global": ((expected_t, out_h, out_w, 3), np.dtype("uint8")),
        "robot_qpos": ((expected_t, 6), np.dtype("float32")),
        "robot_eef_pose": ((expected_t, 6), np.dtype("float32")),
        "action": ((expected_t, 6), np.dtype("float32")),
        "valid": ((expected_t,), np.dtype("bool")),
        "timestamp": ((expected_t,), np.dtype("float64")),
    }
    for key, (shape, dtype) in expected.items():
        arr = root[key]
        assert arr.shape == shape, (key, arr.shape, shape)
        assert arr.dtype == dtype, (key, arr.dtype, dtype)
    assert int(root["valid"][:].sum()) > 0
    _ = root["camera_wrist"][0]
    _ = root["camera_global"][-1]
    _ = root["action"][0]
    return True


def convert_episode(
        hdf5_path: pathlib.Path,
        output_dir: pathlib.Path,
        out_h: int,
        out_w: int,
        delete_source: bool,
        overwrite: bool,
    ) -> pathlib.Path:
    final_path = output_dir.joinpath(hdf5_path.stem + ".zarr")
    tmp_path = output_dir.joinpath(hdf5_path.stem + ".zarr.tmp")
    if final_path.exists():
        if not overwrite:
            print(f"Skip existing {final_path}")
            return final_path
        shutil.rmtree(final_path)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
    lowdim_compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    with h5py.File(hdf5_path, "r") as f:
        T = int(f["action"].shape[0])
        root = zarr.open(str(tmp_path), mode="w")
        root.attrs["source_hdf5"] = str(hdf5_path)
        root.attrs["schema"] = "piper_zarr_real_image_v1"
        root.attrs["image_shape_hwc"] = [out_h, out_w, 3]
        root.attrs["source_schema_version"] = f.attrs.get("schema_version", "")

        wrist = _resize_images(f["observations/images/wrist"], out_h, out_w)
        global_img = _resize_images(f["observations/images/global"], out_h, out_w)
        robot_qpos = f["observations/qpos"][:].astype(np.float32)
        robot_eef_pose = f["observations/eef_pose"][:].astype(np.float32)
        action = f["action"][:].astype(np.float32)
        valid = _make_valid_mask(f)
        timestamp = (f["time/timestamp_ns"][:].astype(np.float64) / 1e9)

    _write_array(root, "camera_wrist", wrist, (1, out_h, out_w, 3), image_compressor)
    _write_array(root, "camera_global", global_img, (1, out_h, out_w, 3), image_compressor)
    _write_array(root, "robot_qpos", robot_qpos, (min(T, 4096), 6), lowdim_compressor)
    _write_array(root, "robot_eef_pose", robot_eef_pose, (min(T, 4096), 6), lowdim_compressor)
    _write_array(root, "action", action, (min(T, 4096), 6), lowdim_compressor)
    _write_array(root, "valid", valid.astype(bool), (min(T, 4096),), lowdim_compressor)
    _write_array(root, "timestamp", timestamp, (min(T, 4096),), lowdim_compressor)

    _verify_episode(tmp_path, expected_t=T, out_h=out_h, out_w=out_w)
    os.replace(str(tmp_path), str(final_path))
    _verify_episode(final_path, expected_t=T, out_h=out_h, out_w=out_w)

    if delete_source:
        os.remove(hdf5_path)
    return final_path


@click.command()
@click.option("--input", "-i", "input_dir", default=DEFAULT_INPUT, help="Directory with Piper .hdf5 episodes.")
@click.option("--output", "-o", "output_dir", default=DEFAULT_OUTPUT, help="Directory for .zarr episodes.")
@click.option("--limit", default=None, type=int, help="Convert only the first N sorted episodes.")
@click.option("--height", default=240, type=int, help="Output image height.")
@click.option("--width", default=320, type=int, help="Output image width.")
@click.option("--delete-source", is_flag=True, default=False, help="Delete each source .hdf5 after successful verification.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing .zarr episodes.")
def main(input_dir, output_dir, limit, height, width, delete_source, overwrite):
    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    output_dir = pathlib.Path(os.path.expanduser(output_dir))
    hdf5_paths = sorted(input_dir.glob("*.hdf5"))
    if limit is not None:
        hdf5_paths = hdf5_paths[:limit]
    assert len(hdf5_paths) > 0, f"No .hdf5 episodes found in {input_dir}"

    print(f"input: {input_dir}")
    print(f"output: {output_dir}")
    print(f"episodes: {len(hdf5_paths)}")
    print(f"delete_source: {delete_source}")
    for hdf5_path in tqdm(hdf5_paths, desc="Converting Piper HDF5"):
        zarr_path = convert_episode(
            hdf5_path=hdf5_path,
            output_dir=output_dir,
            out_h=height,
            out_w=width,
            delete_source=delete_source,
            overwrite=overwrite,
        )
        print(f"Converted {hdf5_path.name} -> {zarr_path.name}")


if __name__ == "__main__":
    main()
