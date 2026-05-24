from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class PiperImageRunner(BaseImageRunner):
    """No-op runner for Piper real-robot training.

    Real robot evaluation is handled by eval_piper_real_robot.py instead of the
    training loop, so this runner only keeps the workspace interface satisfied.
    """

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy):
        return {}
