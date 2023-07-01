import contextlib
import json
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, Any

import aiofiles
import aiohttp

from .constants import STYLES_URL, STYLE_CATEGORIES_URL
from .types import GenerationParams, GenerationRequest, LoRA


async def _request_with_cache(
    *,
    session: aiohttp.ClientSession,
    request_args: dict,
    cache_file_path: Path,
    # Objects supported by json.JSONDecoder
    data_parser : Callable[
        [dict | list | str | int | float | bool | None],
        tuple[Any, dict | list | str | int | float | bool | None]
    ] = lambda x: x,
    dump_with_indent: bool = False
) -> Any:
    # If a valid cache is found, we return whatever value it holds, otherwise an exception is raised and we continue
    if cache_file_path.is_file():
        try:
            async with aiofiles.open(cache_file_path, "r", encoding="utf-8") as cache_file:
                cache_json: dict = json.loads(await cache_file.read())
            assert cache_json.get("saved_at", 0) + 60*60*24*2 > time.time()
            assert (data := cache_json.get("data")) is not None
        except (AssertionError, JSONDecodeError):
            pass
        else:
            # If it errors we just move on to fetching new data
            with contextlib.suppress(Exception):
                return_data, json_data = data_parser(data)
                return return_data

    async with session.get(**request_args) as response:
        data: dict = await response.json()
        # Using a user-supplied function here is a bit messy, but it gives us the most flexibility
        # but since it's a private function mainly meant to be used by functions that fetch specific things, it's fine
        return_data, json_data = data_parser(data)

        async with aiofiles.open(cache_file_path, "w", encoding="utf-8") as cache_file:
            await cache_file.write(json.dumps(
                {
                    "saved_at": time.time(),
                    "data": json_data
                },
                indent=4 if dump_with_indent else None
            ))
        return return_data


async def fetch_styles(*, session: aiohttp.ClientSession, storage_file_path: Path) -> dict[str, GenerationRequest]:
    def parse(style_data: Any) -> GenerationRequest:
        prompt: str = style_data.pop("prompt") # type: ignore
        if "{np}" in prompt and "###" not in prompt:
            prompt = prompt.replace("{np}", "###{np}")

        prompts = prompt.split("###", maxsplit=1)
        request = GenerationRequest(
            positive_prompt=prompts[0],
            negative_prompt=prompts[1] if len(prompts) == 2 else "",
            models=[style_data.pop("model")] if "model" in style_data else None,
            session=session
        )

        for key, value in style_data.items():
            if key in GenerationRequest.model_json_schema()["properties"].keys():
                if key == "params":
                    setattr(request, key, GenerationParams(**value))
                else:
                    setattr(request, key, value)
            elif key in GenerationParams.model_json_schema()["properties"].keys():
                if request.params is None:
                    request.params = GenerationParams()
                if key == "loras":
                    if request.params.loras is None:
                        request.params.loras = []
                    for lora in value:
                        request.params.loras.append(LoRA(**lora))
            else:
                raise KeyError(f"Unknown style key: {key}")

        return request

    def data_parser(data: dict) -> tuple[Any, dict | list | str | int | float | bool | None]:
        if not isinstance(data, dict):
            raise ValueError("Github's API did not respond with a dictionary.")
        if "message" in data:
            raise RuntimeError(f"Error fetching styles from github: {data.get('message', '')}")

        return_data = {key: parse(value) for key, value in data.items()}
        json_data = {key: value.dump_json_dict() for key, value in return_data.items()}
        return return_data, json_data

    return await _request_with_cache(
        session=session,
        request_args={
            "url": STYLES_URL,
            "headers": {"Accept": "application/vnd.github.raw+json"}
        },
        cache_file_path=storage_file_path,
        data_parser=data_parser,
        dump_with_indent=True
    )

async def fetch_style_categories(*, session: aiohttp.ClientSession, storage_file_path: Path) -> dict[str, list[str]]:
    def data_parser(data: dict) -> tuple[Any, dict | list | str | int | float | bool | None]:
        if not isinstance(data, dict):
            raise ValueError("Github's API did not respond with a dictionary.")
        if not all(
            isinstance(key, str) and all(
                isinstance(value, str)
                for value in values
            )
            for key, values in data.items()
        ):
            raise ValueError("The style categories supplied do not follow the format expected.")
        return data, data

    return await _request_with_cache(
        session=session,
        request_args={
            "url": STYLE_CATEGORIES_URL,
            "headers": {"Accept": "application/vnd.github.raw+json"}
        },
        cache_file_path=storage_file_path,
        data_parser=data_parser,
        dump_with_indent=True
    )


def modify_with_style(
    generation: GenerationRequest | GenerationParams,
    style: GenerationRequest | GenerationParams,
) -> GenerationRequest | GenerationParams:
    new_generation = generation

    # Overwrite generation values with style values
    for key, value in style.model_dump(
        exclude_defaults=True  # We don't want to overwrite stuff that's not specified in the style
    ).items():
        if key == "prompt":
            new_generation.prompt = style.prompt.format(
                p=generation.positive_prompt,
                np=generation.negative_prompt.removeprefix(", ")
            )
        elif key == "params":
            params = new_generation.params.model_dump() | value

            # Convert lora dicts into LoRA objects
            if loras := params.get("loras"):
                params["loras"] = []
                for lora in loras:
                    params["loras"].append(LoRA(**lora))
            new_generation.params = GenerationParams(**params)

    return new_generation




