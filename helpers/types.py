import json
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic import BaseModel, constr, Field, conlist, confloat, conint, conset, computed_field, Extra

__all__ = [
    "GenerationRequest",
    "GenerationParams",
    "QueuedGeneration",
    "GenerationCheck",
    "GenerationStatus",
    "ActiveModel",
    "ModelType",
    "SourceProcessors",
    "LoRA",
    "TextualInversion",
    "TextualInversionInject",
    "PostProcessors",
    "ControlType",
    "InterrogationRequest",
    "InterrogationForm",
    "InterrogationType",
    "InterrogationStatuses",
    "CivitAIData",
]

from .constants import HORDE_API_BASE
from .errors import RequestError


# noinspection SpellCheckingInspection
class SourceProcessors(Enum):
    IMG2IMG = "img2img"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting "


# noinspection SpellCheckingInspection
class SamplerNames(Enum):
    K_LMS = "k_lms"
    K_HEUN = "k_heun"
    K_EULER = "k_euler"
    K_EULER_A = "k_euler_a"
    K_DPM_2 = "k_dpm_2"
    K_DPM_2_A = "k_dpm_2_a"
    K_DPM_FAST = "k_dpm_fast"
    K_DPM_ADAPTIVE = "k_dpm_adaptive"
    K_DPMPP_2S_A = "k_dpmpp_2s_a"
    K_DPMPP_2M = "k_dpmpp_2m"
    DPMSOLVER = "dpmsolver"
    K_DPMPP_SDE = "k_dpmpp_sde"
    DDIM = "DDIM"


# noinspection SpellCheckingInspection
class PostProcessors(Enum):
    GFPGAN = "GFPGAN"
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    NMKD_SIAX = "NMKD_Siax"
    FOURX_ANIMESHARP = "4x_AnimeSharp"
    CODEFORMERS = "CodeFormers"
    STRIP_BACKGROUND = "strip_background"


# noinspection SpellCheckingInspection
class ControlType(Enum):
    CANNY = "canny"
    HED = "hed"
    DEPTH = "depth"
    NORMAL = "normal"
    OPENPOSE = "openpose"
    SEG = "seg"
    SCRIBBLE = "scribble"
    FAKESCRIBBLES = "fakescribbles"
    HOUGH = "hough"


class TextualInversionInject(Enum):
    POSITIVE = "prompt"
    NEGATIVE = "negprompt"


# noinspection SpellCheckingInspection
class LoRA(BaseModel):
    name: constr(min_length=1, max_length=255) = Field(
        description="The exact name or CivitAI ID of the LoRa."
    )
    model: confloat(ge=-5.0, le=5.0) = Field(
        1.0,
        description="The strength of the LoRa to apply to the SD model."
    )
    clip: confloat(ge=-5.0, le=5.0) = Field(
        1.0,
        description="The strength of the LoRa to apply to the clip model."
    )
    inject_trigger: constr(min_length=1, max_length=30) | None = Field(
        None,
        description=(
            "If set, will try to discover a trigger for this LoRa "
            "which matches or is similar to this string and inject it into the prompt. "
            "If 'any' is specified it will be pick the first trigger."
        )
    )

    # None of the following fields are part of the stable horde API and will be excluded in dumps
    nsfw: bool = Field(False, exclude=True)
    tags: list[str] = Field([], exclude=True)
    actual_name: str = Field("", exclude=True)


class TextualInversion(BaseModel):
    name: constr(min_length=1, max_length=255) = Field(
        description="The exact name or CivitAI ID of the Textual Inversion."
    )
    inject_ti: TextualInversionInject | None = Field(
        None,
        description=(
            "If set, will automatically add this textual inversion filename "
            "to the prompt or negative prompt accordingly using the provided strength. "
            "If this is set to None, then the user will have to manually add the embed to the prompt themselves."
        )
    )

    strength: confloat(ge=-5.0, le=5.0) = Field(
        1.0,
        description=(
            "The strength with which to apply the textual inversion to the prompt. "
            "Only used when inject_ti is not None"
        )
    )

    # None of the following fields are part of the stable horde API and will be excluded in dumps
    nsfw: bool = Field(False, exclude=True)
    tags: list[str] = Field([], exclude=True)
    actual_name: str = Field("", exclude=True)


# noinspection SpellCheckingInspection
class GenerationParams(BaseModel):
    class Config:
        use_enum_values = True

    sampler_name: SamplerNames = SamplerNames.K_EULER_A
    cfg_scale: confloat(ge=0, le=100, multiple_of=0.5) = 7.5
    denoising_strength: confloat(ge=0.01, le=1) | None = None
    seed: str | None = Field(
        None,
        description="The seed to use to generate this request. You can pass text as well as numbers."
    )
    height: conint(ge=64, le=3072, multiple_of=64) = Field(
        512,
        description="The height of the image to generate."
    )
    width: conint(ge=64, le=3072, multiple_of=64) = Field(
        512,
        description="The width of the image to generate."
    )
    seed_variation: conint(ge=1, le=1000) | None = Field(
        None,
        description="If passed with multiple n, the provided seed will be incremented every time by this value."
    )
    post_processing: conset(PostProcessors) | None = Field(
        None,
        description="The list of post-processors to apply to the image, in the order to be applied."
    )
    karras: bool = Field(
        False,
        description="Set to True to enable karras noise scheduling tweaks."
    )
    tiling: bool = Field(
        False,
        description="Set to True to create images that stitch together seamlessly."
    )
    hires_fix: bool = Field(
        False,
        description="Set to True to process the image at base resolution before upscaling and re-processing."
    )
    clip_skip: conint(ge=1, le=12) | None = Field(
        None,
        description="The number of CLIP language processor layers to skip."
    )
    control_type: ControlType | None = None
    image_is_control: bool = Field(
        False,
        description="Set to True if the image submitted is a pre-generated control map for ControlNet use."
    )
    return_control_map: bool = Field(
        False,
        description="Set to True if you want the ControlNet map returned instead of a generated image."
    )
    facefixer_strength: confloat(ge=0, le=1) | None = None
    loras: list[LoRA] | None = None
    tis: list[TextualInversion] | None = None

    # This is, according to db0, used for private models hosted by the likes of stability.ai
    # Thus, it's not documented and shouldn't (?) need to be implemented by me,
    # a plain dict is fine if it's really needed
    special: dict | None = None

    steps: conint(ge=1, le=500) = 30
    n: conint(ge=1, le=20) = Field(
        1,
        description="The amount of images to generate."
    )


# noinspection LongLine
class GenerationCheck(BaseModel):
    finished: int = Field(description="The amount of finished jobs in this request.")
    processing: int = Field(description="The amount of still processing jobs in this request.")
    restarted: int = Field(
        description="The amount of jobs that timed out and had to be restarted or were reported as failed by a worker.")
    waiting: int = Field(description="The amount of jobs waiting to be picked up by a worker.")
    done: bool = Field(description="True when all jobs in this request are done. Else False.")
    faulted: bool = Field(False,
                          description="True when this request caused an internal server error and could not be completed.")
    wait_time: int = Field(description="The expected amount to wait (in seconds) to generate all jobs in this request.")
    queue_position: int = Field(
        description="The position in the requests queue. This position is determined by relative Kudos amounts.")
    kudos: float = Field(description="The amount of total Kudos this request has consumed until now.")
    is_possible: bool = Field(
        description="If False, this request will not be able to be completed with the pool of workers currently available.")


class GenerationState(Enum):
    OK = "ok"
    CENSORED = "censored"


class StableGeneration(BaseModel):
    worker_id: str = Field(description="The UUID of the worker which generated this image.")
    worker_name: str = Field(description="The name of the worker which generated this image.")
    model: str = Field(description="The model which generated this image.")
    state: GenerationState = Field(GenerationState.OK, description="The state of this generation.")
    img: str = Field(description="The generated image as a Base64-encoded .webp file.")
    seed: str = Field(description="The seed which generated this image.")
    id: str = Field(description="The ID for this image.")
    censored: bool = Field(description="When true this image has been censored by the worker's safety filter.")


class GenerationStatus(GenerationCheck):
    generations: list[StableGeneration]
    shared: bool = Field(description="If True, These images have been shared with LAION.")


class QueuedGeneration(BaseModel, extra=Extra.allow):
    def __init__(self, *, session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    id: str = Field(description="The UUID of the request. Use this to retrieve the request status in the future.")
    kudos: int = Field(description="The expected kudos consumption for this request.")
    message: str | None = Field(None, description="Any extra information from the horde about this request.")

    async def _fetch_status(self, full: bool) -> GenerationCheck | GenerationStatus:
        api_endpoint = f"{HORDE_API_BASE}/generate/" + ("status" if full else "check") + f"/{self.id}"
        async with self.session.get(api_endpoint) as response:
            response_json = await response.json()
            if response.status != 200:
                raise RequestError(response_json["message"], status_code=response.status)
            return GenerationStatus(**response_json) if full else GenerationCheck(**response_json)

    async def check(self) -> GenerationCheck:
        """Checks the status of this request."""
        return await self._fetch_status(full=False)

    async def status(self) -> GenerationStatus:
        """Checks the full status of this request."""
        return await self._fetch_status(full=True)


# noinspection SpellCheckingInspection, LongLine
class GenerationRequest(BaseModel, use_enum_values=True, extra=Extra.allow):
    def __init__(self, *, session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    positive_prompt: constr(min_length=1) = Field(
        description="The positive prompt which will be sent to Stable Diffusion to generate an image.",
        exclude=True
    )
    negative_prompt: str = Field(
        "",
        description="The negative prompt which will be sent to Stable Diffusion to generate an image.",
        exclude=True
    )

    @computed_field
    @property
    def prompt(self) -> str:
        """The prompt which will be sent to Stable Diffusion to generate an image."""
        return self.positive_prompt + (f"###{self.negative_prompt}" if self.negative_prompt else "")

    @prompt.setter
    def prompt(self, value: str) -> None:
        if "###" in value:
            self.positive_prompt, self.negative_prompt = value.split("###", maxsplit=1)
        else:
            self.positive_prompt = value
            self.negative_prompt = ""

    params: GenerationParams | None = None

    nsfw: bool = Field(
        False,
        description="Set to true if this request is NSFW. This will skip workers which censor images."
    )
    trusted_workers: bool = Field(
        False,
        description=(
            "When true, only trusted workers will serve this request. "
            "When False, Evaluating workers will also be used which can increase speed but adds more risk!"
        ),
    )
    slow_workers: bool = Field(
        True,
        description="When True, allows slower workers to pick up this request. Disabling this incurs an extra kudos cost.",
    )
    censor_nsfw: bool = Field(
        False,
        description="If the request is SFW, and the worker accidentally generates NSFW, it will send back a censored image.",
    )
    workers: conlist(str, min_length=1, max_length=5) | None = Field(
        None,
        description="Specify up to 5 workers which are allowed to service this request.",
    )
    worker_blacklist: bool = Field(
        False,
        description="If true, the worker list will be treated as a blacklist instead of a whitelist."
    )
    models: list[str] | None = Field(
        None,
        description="Specify which models are allowed to be used for this request."
    )
    source_image: str | bytes | None = Field(
        None,
        description="The Base64-encoded webp to use for img2img.",
    )
    source_processing: SourceProcessors | None = Field(
        None,
        description="If source_image is provided, specifies how to process it."
    )
    source_mask: str | None = Field(
        None,
        description=(
            "If source_processing is set to 'inpainting' or 'outpainting', this parameter can be optionally "
            "provided as the Base64-encoded webp mask of the areas to inpaint. "
            "If this arg is not passed, the inpainting/outpainting mask has to be embedded as alpha channel."
        ),
    )
    r2: bool = Field(
        True,
        description="If True, the image will be sent via cloudflare r2 download link"
    )
    shared: bool = Field(
        False,
        description=(
            "If True, The image will be shared with LAION for improving their dataset. "
            "This will also reduce your kudos consumption by 2. For anonymous users, this is always True."
        ),
    )
    replacement_filter: bool = Field(
        True,
        description="If enabled, suspicious prompts are sanitized through a string replacement filter instead True.",
    )
    dry_run: bool = Field(
        False,
        description="When false, the endpoint will simply return the cost of the request in kudos and exit."
    )

    def dump_json_dict(
        self,
        *,
        include: set[str] | list[str] | None = None,
        exclude: set[str] | list[str] | None = None
    ) -> dict:
        return json.loads(self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            include=set(include) if include else None,
            exclude=set(exclude) if exclude else "session",
        ))

    async def _make_request(self, dry_run: bool = False) -> QueuedGeneration | int:
        async with self.session.post(
            f"{HORDE_API_BASE}/generate/async",
            json=self.dump_json_dict() | {"dry_run": dry_run}
        ) as response:
            response_json: dict = await response.json()

            match tuple(response_json.keys()):
                case ("kudos", ):
                    return response_json["kudos"]
                case ("id", "kudos") | ("id", "kudos", "message"):
                    return QueuedGeneration(
                        id=response_json["id"],
                        kudos=response_json["kudos"],
                        message=response_json.get("message"),
                        session=self.session
                    )
                case _:
                    print(response_json)
                    raise RequestError(response_json.get("message", "Unknown error."), status_code=response.status)

    async def fetch_kudo_cost(self) -> int:
        return await self._make_request(dry_run=True)

    async def request_generation(self) -> QueuedGeneration:
        return await self._make_request()


class ModelType(Enum):
    TEXT = "text"
    IMAGE = "image"


class ActiveModel(BaseModel):
    name: str = Field(description="The Name of a model available by workers in this horde.")
    count: int = Field(description="How many of workers in this horde are running this model.")
    performance: float = Field(description="The average speed of generation for this model.")
    queued: int = Field(description="The amount waiting to be generated by this model.")
    jobs: float = Field(description="The job count waiting to be generated by this model.")
    eta: int = Field(description="Estimated time in seconds for this model's queue to be cleared.")
    type: ModelType = Field(description="The model type (text or image).")


class InterrogationStatuses(Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    DONE = "done"


class InterrogationFormStatus(BaseModel):  # I swear these names are only getting less logical
    form: str | None = Field(None, description="The type of interrogation this is.")
    state: str | None = Field(None, description="The overall status of this interrogation.")

    # TODO: Figure out what this is
    # It's somewhat similar to GenerationParams.special, but I don't know what that does either
    result: dict | None = None  # ?????????


class InterrogationStatus(BaseModel):
    state: InterrogationStatuses = Field(description="The overall status of this interrogation.")
    forms: list[InterrogationFormStatus]


class QueuedInterrogation(BaseModel, extra=Extra.allow):
    def __init__(self, *, session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    id: str = Field(description="The UUID of the request. Use this to retrieve the request status in the future.")
    message: str | None = Field(None, description="Any extra information from the horde about this request.")

    async def fetch_status(self) -> InterrogationStatus:
        # FYI: despite what the docs want you to believe, there is no check endpoint!
        api_endpoint = f"{HORDE_API_BASE}/interrogate/status/{self.id}"
        async with self.session.get(api_endpoint) as response:
            response_json = await response.json()
            if response.status != 200:
                raise RequestError(response_json["message"], status_code=response.status)
            return InterrogationStatus(**response_json)


# noinspection SpellCheckingInspection
class InterrogationType(Enum):
    CAPTION = "caption"
    INTERROGATION = "interrogation"
    NSFW = "nsfw"
    GFPGAN = "GFPGAN"
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    NMKD_SIAX = "NMKD_Siax"
    FOURX_ANIMESHARP = "4x_AnimeSharp"
    CODEFORMERS = "CodeFormers"
    STRIP_BACKGROUND = "strip_background"


class InterrogationForm(BaseModel, use_enum_values=True):
    name: InterrogationType = Field(description="The type of interrogation this is.")

    # TODO: Figure out what this is
    # It's somewhat similar to GenerationParams.special, but I don't know what that does either
    payload: dict | None = None  # ?????????


# Corresponds to "ModelInterrogationInputStable" in the official docs
# was renamed for consistency with "GenerationRequest"
# noinspection LongLine
class InterrogationRequest(BaseModel, extra=Extra.allow):
    def __init__(self, *, session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    forms: list[InterrogationForm]
    source_image: str = Field(description="The public URL of the image to interrogate.")
    slow_workers: bool = Field(
        True,
        description="When True, allows slower workers to pick up this request. Disabling this incurs an extra kudos cost.",
    )

    def dump_json_dict(self) -> dict:
        return json.loads(self.model_dump_json(
            exclude_none=True,
            exclude_defaults=True,
            exclude="session"  # type: ignore
        ))

    async def request_interrogation(self):
        async with self.session.post(
            f"{HORDE_API_BASE}/interrogate/async",
            json=self.dump_json_dict()
        ) as response:
            response_json: dict = await response.json()

            match tuple(response_json.keys()):
                case ("id", ) | ("id", "message"):
                    return QueuedInterrogation(
                        id=response_json["id"],
                        message=response_json.get("message"),
                        session=self.session
                    )
                case _:
                    raise RequestError(response_json.get("message", "Unknown error."), status_code=response.status)


@dataclass
class CivitAIData:
    loras: list[LoRA]
    textual_inversions: list[TextualInversion]


