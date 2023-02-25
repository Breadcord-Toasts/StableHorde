from typing import TypedDict

from .utils import GeneratedImage


class RequestFail(TypedDict):
    message: str


class ImageRequestResponse(TypedDict):
    id: str
    message: str


class GenerationCheckResponse(TypedDict):
    finished: int
    processing: int
    restarted: int
    waiting: int
    done: bool
    faulted: bool
    wait_time: int
    queue_position: int
    kudos: float
    is_possible: bool


class GenerationStatusResponse(GenerationCheckResponse):
    shared: bool
    generations: list[GeneratedImage]
