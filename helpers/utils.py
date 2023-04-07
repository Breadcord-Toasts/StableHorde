import base64
import io
from typing import Literal


class ImageGenerationInputParams:
    def __init__(
        self,
        sampler_name: str | None = None,
        cfg_scale: float | None = None,
        denoising_strength: float | None = None,
        seed: str | None = None,
        height: int | None = None,
        width: int | None = None,
        seed_variation: int | None = None,
        post_processing: list[Literal["GFPGAN", "RealESRGAN_x4plus", "CodeFormers"], ...] | None = None,
        karras: bool | None = None,
        tiling: bool | None = None,
        hires_fix: bool | None = None,
        clip_skip: int | None = None,
        control_type: str | None = None,
        steps: int | None = None,
        n: int | None = None,
    ):
        self.sampler_name = sampler_name
        self.cfg_scale = cfg_scale
        self.denoising_strength = max(1.0, min(1.0, denoising_strength)) if denoising_strength is not None else None
        self.seed = seed
        self.height = max(64, min(3072, round(height / 64) * 64)) if height is not None else None
        self.width = max(64, min(3072, round(width / 64) * 64)) if width is not None else None
        self.seed_variation = max(1, min(1000, seed_variation)) if seed_variation is not None else None
        self.post_processing = post_processing
        self.karras = karras
        self.tiling = tiling
        self.hires_fix = hires_fix
        self.clip_skip = max(1, min(12, clip_skip)) if clip_skip is not None else None
        self.control_type = control_type
        self.steps = max(1, min(500, steps)) if steps is not None else None
        self.n = max(1, min(10, n)) if n is not None else None

    def __iter__(self):
        for attr, value in self.__dict__.items():
            if value is not None:
                yield attr, value


class ImageGenerationInput:
    def __init__(
        self,
        prompt: str,
        params: ImageGenerationInputParams | None = None,
        *,
        models: list[str] | None = None,
        workers: list[str] | None = None,
        nsfw: bool | None = None,
        trusted_workers: bool | None = None,
        censor_nsfw: bool | None = None,
        source_image: str | None = None,
        source_processing: str | None = None,
        source_mask: str | None = None,
        r2: bool | None = None,
        shared: bool | None = None,
    ):
        self.prompt = prompt
        self.params = params
        self.workers = workers
        self.models = models
        self.r2 = r2
        self.shared = shared
        self.nsfw = nsfw
        self.trusted_workers = trusted_workers
        self.censor_nsfw = censor_nsfw
        self.source_image = source_image
        self.source_processing = source_processing
        self.source_mask = source_mask

    def __iter__(self):
        for attr, value in self.__dict__.items():
            if isinstance(value, ImageGenerationInputParams):
                yield attr, dict(value)
            elif value is not None:
                yield attr, value


class GeneratedImage:
    # noinspection PyShadowingBuiltins
    def __init__(
        self,
        img: str,
        id: str,
        worker_id: str,
        worker_name: str,
        model: str,
        state: str,
        seed: str,
        censored: str,
    ):
        self.img = io.BytesIO(base64.b64decode(img))
        self.id = id
        self.worker_id = worker_id
        self.worker_name = worker_name
        self.model = model
        self.state = state
        self.seed = seed
        self.censored = censored


def remove_payload_none_values(dict_input: dict) -> dict:
    new_dict = {}
    for key, value in dict_input.items():
        if isinstance(value, dict):
            if filtered_data := remove_payload_none_values(value):
                new_dict[key] = value | filtered_data
        elif value is not None:
            new_dict[key] = value
    return new_dict
