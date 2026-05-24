import pathlib

import click
import hydra
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option(
    "--dataset_path",
    "-d",
    default="/home/gx4070/Desktop/arm-datasets-collect/data/piper_xy_zarr",
    help="Directory containing Piper .zarr episodes.",
)
@click.option("--config-name", default="train_diffusion_unet_piper_zarr_real_image_workspace")
def main(dataset_path, config_name):
    config_path = pathlib.Path(__file__).parents[1].joinpath("config")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path)):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[f"task.dataset_path={dataset_path}"],
        )
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    print(f"dataset_path: {dataset_path}")
    print(f"episodes: {len(dataset.episode_paths)}")
    print(f"episode length min/max/sum: "
          f"{dataset.episode_lengths.min()} / "
          f"{dataset.episode_lengths.max()} / "
          f"{dataset.episode_lengths.sum()}")
    print(f"train windows: {len(dataset)}")
    print(f"rgb keys: {dataset.rgb_keys}")
    print(f"lowdim keys: {dataset.lowdim_keys}")

    sample = dataset[0]
    print("sample:")
    for key, value in sample["obs"].items():
        print(f"  obs/{key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"  action: shape={tuple(sample['action'].shape)} dtype={sample['action'].dtype}")

    normalizer = dataset.get_normalizer()
    print(f"normalizer keys: {list(normalizer.params_dict.keys())}")


if __name__ == "__main__":
    main()
