import asyncio
import contextlib
import inspect
import io
import json
import re
import time
from base64 import b64encode, b64decode

import aiofiles
import aiohttp
import discord
import pydantic
from discord import app_commands
from discord.ext import tasks, commands

import breadcord
from .helpers.constants import HORDE_API_BASE
from .helpers.errors import (
    RequestError,
    HordeAPIError,
    MaintenanceModeError,
    InsufficientKudosError,
    FaultedGenerationError,
    MissingGenerationsError
)
from .helpers.types import *

available_models: list[ActiveModel] = []
# civitai_data: CivitAIData = CivitAIData(models=[], last_indexed_at=0)
# available_styles: dict[str, dict] = {} #TODO: use this
# available_style_categories: dict[str, list] = {} #TODO: use this


class DeleteButtonView(discord.ui.View):
    def __init__(self, *, required_votes: int, author_id: int):
        super().__init__()
        self.required_votes = required_votes
        self.author_id = author_id
        self.votes = []

    @discord.ui.button(emoji="\N{WASTEBASKET}", label="Delete", style=discord.ButtonStyle.red)
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if (user := interaction.user.id) not in self.votes:
            self.votes.append(user)
        else:
            self.votes.remove(user)

        vote_count = len(self.votes)
        button.label = f"Delete ({vote_count}/{self.required_votes})" if vote_count else "Delete"

        if vote_count >= self.required_votes or interaction.user.id == self.author_id:
            for button in self.children:
                button.disabled = True
            await interaction.response.edit_message(attachments=[], view=self)
            return
        await interaction.response.edit_message(view=self)


def is_diffusion_model(model: ActiveModel) -> bool:
    return model.type == ModelType.IMAGE


class DiffusionModelTransformer(app_commands.Transformer):
    def transform(self, interaction: discord.Interaction, value: str, /) -> ActiveModel | None:
        value = re.sub(" \(\d+ workers\)$", "", value.strip())
        for model in available_models:
            if model.name.strip() == value:
                return model

    async def autocomplete(self, interaction: discord.Interaction, value: str, /) -> list[app_commands.Choice[str]]:
        def get_choice(model: ActiveModel) -> app_commands.Choice:
            return app_commands.Choice(
                name=txt
                if len(txt := f"{model.name} ({model.count} workers)") <= 100
                else model.name,
                value=model.name,
            )

        if not value:
            return [
                get_choice(model)
                for model in sorted(
                    filter(is_diffusion_model, available_models),
                    key=lambda m: m.count,
                    reverse=True
                )[:25]
            ]

        return [
            get_choice(model)
            for model in breadcord.helpers.search_for(
                query=value,
                objects=list(available_models),
                key=lambda m: m.name,
                threshold=60
            )
        ]


class LoRATransformer(app_commands.Transformer):
    def transform(self, interaction: discord.Interaction, value: str, /) -> LoRA | None:
        value = value.strip()
        for lora in civitai_data.models:
            if lora.name.strip() == value:
                return lora

    async def autocomplete(self, interaction: discord.Interaction, value: str, /) -> list[app_commands.Choice[str]]:
        def get_choice(lora: LoRA) -> app_commands.Choice:
            return app_commands.Choice(
                name=lora.name.strip(),
                value=lora.name,
            )

        if not value:
            return [get_choice(lora) for lora in civitai_data.models[:25]]

        return [
            get_choice(lora)
            for lora in breadcord.helpers.search_for(
                query=value,
                objects=list(civitai_data.models),
                key=lambda m: m.name,
                threshold=60
            )
        ]


class StableHorde(breadcord.module.ModuleCog):
    def __init__(self, module_id: str):
        super().__init__(module_id)

        self.session: aiohttp.ClientSession | None = None
        self.update_data.start()

    async def cog_load(self) -> None:
        api_key: str = self.settings.stable_horde_api_key.value
        self.session = aiohttp.ClientSession(headers={"apikey": api_key})

    async def cog_unload(self):
        self.update_data.cancel()
        if self.session is not None:
            await self.session.close()

    @tasks.loop(hours=24)
    async def update_data(self, *, force_update: bool = False) -> None:
        async with self.session.get(f"{HORDE_API_BASE}/status/models", params={"type": "image"}) as response:
            global available_models
            available_models = list(map(
                lambda m: ActiveModel(**m),
                await response.json()
            ))
            self.logger.debug("Fetched horde models")

        return
        # TODO: make this nonsense less awful
        # noinspection PyUnreachableCode

        async with self.session.get(
            "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-Styles/main/styles.json"
        ) as response:
            global available_styles
            available_styles = json.loads(await response.read())
            # Why are these two formatted differently from all the others? Who knows!
            with contextlib.suppress(KeyError):
                del available_styles["raw"]
            with contextlib.suppress(KeyError):
                del available_styles["raw2"]
            self.logger.debug("Fetched styles")
        async with self.session.get(
            "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-Styles/main/categories.json"
        ) as response:
            global available_style_categories
            available_style_categories = json.loads(await response.read())
            self.logger.debug("Fetched style categories")

        global civitai_data
        civitai_models_file = self.module.storage_path / "civitai_model_cache.json"
        with contextlib.suppress(json.JSONDecodeError):
            if civitai_models_file.is_file():
                self.logger.debug("Loading civitai model cache")
                async with aiofiles.open(civitai_models_file, "r", encoding="utf8") as file:
                    data = json.loads(await file.read())
                    civitai_data = CivitAIData(
                        models=[LoRA(name=lora["name"], inject_trigger="any") for lora in data["models"]],
                        last_indexed_at=data.get("last_indexed_at", 0)
                    )
                self.logger.debug("Loaded civitai model cache")

        if civitai_data.last_indexed_at + (60 * 60 * 24) < time.time() or force_update:
            civitai_data = CivitAIData(models=[], last_indexed_at=round(time.time()))
            next_page = "https://civitai.com/api/v1/models?limit=100&&types=LORA&page=1"
            while next_page:
                async with self.session.get(next_page) as response:
                    response_json = await response.json()
                    civitai_data.models.extend(
                        LoRA(name=lora["name"], inject_trigger="any", nsfw=lora.get("nsfw", False))
                        for lora in response_json.get("items", [])
                    )
                    self.logger.debug(f"Fetched page from civitai: {next_page}")
                    next_page = response_json.get("metadata", {}).get("nextPage")
                    await asyncio.sleep(1)
            self.logger.debug("Finished fetching civitai pages")

            async with aiofiles.open(civitai_models_file, "w", encoding="utf8") as file:
                self.logger.debug("Saving civitai model cache")
                await file.write(civitai_data.model_dump_json())
                self.logger.debug(f"Saved civitai model cache with {len(civitai_data.models)} models")

    async def _generate_and_send(
        self,
        interaction: discord.Interaction,
        generation: GenerationRequest,
        *,
        source_image: discord.Attachment | None = None,
    ) -> None:
        lora = generation.params.loras[0] if generation.params.loras else None
        control_type = generation.params.control_type

        await interaction.response.send_message("Starting to generate image...")

        kudo_cost = await generation.fetch_kudo_cost()
        if kudo_cost > float(self.settings.max_kudo_cost.value) and not await self.bot.is_owner(interaction.user):
            raise InsufficientKudosError()
        queued_generation = await generation.request_generation()

        # A generation request times out after 10 minutes
        for _ in range((10 * 60) // int(self.settings.time_between_updates.value)):
            check = await queued_generation.check()
            if check.faulted:
                raise FaultedGenerationError()
            if check.done:
                break
            await interaction.edit_original_response(
                content="",
                embed=discord.Embed(
                    title="Generation status",
                    description=inspect.cleandoc(
                        f"""
                        Estimated to be done <t:{round(time.time() + check.wait_time)}:R>
                        Position in queue: {check.queue_position}
                        Estimated kudo cost: {kudo_cost}

                        **Generation settings**
                        Prompt: {generation.positive_prompt}
                        Negative prompt: {generation.negative_prompt or None}
                        Marked as NSFW: {generation.nsfw}
                        """
                    )
                ).set_thumbnail(
                    url=source_image.url if source_image else None
                )
            )
            await asyncio.sleep(int(self.settings.time_between_updates.value))

        generation_status = await queued_generation.status()
        if not generation_status.generations:
            raise MissingGenerationsError()

        required_deletion_votes = int(self.settings.required_deletion_votes.value)
        await interaction.edit_original_response(
            content="",
            embeds=[
                discord.Embed(
                    title="Generation complete",
                    description=inspect.cleandoc(
                        f"""
                        **Prompt:** {generation.positive_prompt}
                        **Negative prompt:** {generation.negative_prompt or None}
                        **Seed:** {finished_generation.seed}
                        **Model:** {finished_generation.model}
                        **Marked as NSFW:** {generation.nsfw}
                        **Lora:** {lora.name if lora else None}
                        **ControlNet type:** {control_type.value.lower() if control_type else None}

                        ### **Horde metadata**
                        **Finished by worker:** {finished_generation.worker_name} (`{finished_generation.worker_id}`)
                        **Total kudo cost:** {generation_status.kudos}
                        """
                    ),
                ).set_author(
                    name=interaction.user.display_name,
                    icon_url=interaction.user.display_avatar.url
                ).set_thumbnail(
                    url=source_image.url if source_image else None
                )
                for finished_generation in generation_status.generations
            ],
            attachments=[
                discord.File(
                    fp=io.BytesIO(b64decode(finished_generation.img)),
                    filename=f"{'SPOILER_' if generation.nsfw else ''}{finished_generation.id}.webp"
                )
                for finished_generation in generation_status.generations
            ],
            view=DeleteButtonView(
                required_votes=min(
                    required_deletion_votes,
                    len(set(filter(lambda m: not m.bot, interaction.channel.members)))
                ) if hasattr(interaction.channel, "members") else required_deletion_votes,
                author_id=interaction.user.id
            )
        )

    @app_commands.command(description="Generates an image using Stable Diffusion",)
    @app_commands.rename(
        nsfw="is_nsfw",
        n="image_count",
        control_type="controlnet_model",
        return_control_map="return_controlnet_map",
    )
    @app_commands.describe(
        positive_prompt="A prompt describing what you want in your image",
        negative_prompt="A prompt describing what you don't want in your image",
        model="The model to use for generation",
        nsfw="Whether or not the generation reqeust is NSFW",
        width="The width of the generated image. Has to be a multiple of 64 and between 64 and 3072",
        height="The height of the generated image. Has to be a multiple of 64 and between 64 and 3072",
        steps="The number of steps to run for",
        seed="The seed to use for generation",
        cfg_scale="How closely the AI should adhere to your prompt (cfg = classifier free guidance)",
        n="The number of images to generate",
        tiling="Weather or not to attempt to make the image tillable",
        hires_fix="Weather or not to process the image at base resolution before upscaling and re-processing",
        post_processing="The post processing to apply to the image, this can be ",
        source_image="The base image to use for generation. Must be a webp",
        source_processing="What to use the source image for",
        source_mask="The mask to use for in/outpainting. If left blank the source image's alpha layer will be used",
        control_type="If specified, this dictates what ControlNet model will be used on the source image",
        lora="What LoRA model to use. LoRAs allow putting custom concepts into the prompt",
        return_control_map="Requires controlnet_model to be specified. Whether or not to return the controlnet map instead of a generated image",
    )
    @app_commands.checks.cooldown(1, 10)
    async def imagine(
        self,
        interaction: discord.Interaction,
        positive_prompt: str,
        negative_prompt: str = "",
        model: app_commands.Transform[ActiveModel | None, DiffusionModelTransformer] = None,
        nsfw: bool = False,
        width: int = 512,
        height: int = 512,
        steps: int = 25,
        seed: str | None = None,
        cfg_scale: float = 7.5,
        post_processing: PostProcessors | None = None, # Technically supports a list of postprocessors, but too bad!
        lora: app_commands.Transform[LoRA | None, LoRATransformer] = None,
        source_image: discord.Attachment | None = None,
        source_processing: SourceProcessors | None = None,
        source_mask: discord.Attachment | None = None,
        control_type: ControlType | None = None,
        return_control_map: bool = False,
        tiling: bool = False,
        hires_fix: bool = False,
        n: int = 1,
        #TODO: add style support using https://github.com/Haidra-Org/AI-Horde-Styles
        # Why does it need to use the quirkiest format known to man?
        # Do i seriously need to write a function that converts a style to a GenerationRequest object?
        # Why is the format of the prompt field of the `raw` and `raw2` styles different from all others?
        # How does the official implementations manage to parse this mess? Oh wait, I know! Spaghetti!
        # style: app_commands.Transform[str | None, StyleTransformer] = None
    ) -> None:
        nsfw = nsfw or lora.nsfw if lora else nsfw
        # Cut off due to embed limits. There's no real reason to get this high with most normal prompts anyway
        positive_prompt, negative_prompt = positive_prompt[:1536], negative_prompt[:1536]
        try:
            generation = GenerationRequest(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                nsfw=nsfw,
                models=[model.name] if model else None,
                source_image=b64encode(await source_image.read()) if source_image else None,
                source_processing=source_processing if source_image else None,
                source_mask=b64encode(await source_mask.read()) if source_processing in [
                    SourceProcessors.INPAINTING,
                    SourceProcessors.OUTPAINTING
                ] else None,
                params=GenerationParams(
                    cfg_scale=cfg_scale,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    n=n,
                    tiling=tiling,
                    hires_fix=hires_fix,
                    post_processing=post_processing,
                    control_type=control_type,
                    loras=[lora.model_dump()] if lora else None,
                    return_control_map=return_control_map
                ),
                shared=True,
                r2=False,

                session=self.session
            )
        # This is the only validation error we want to ignore, since it happening depends on the user's reading skills
        except pydantic.ValidationError:
            await interaction.response.send_message("Invalid parameter(s).", ephemeral=True)
            return

        await self._generate_and_send(interaction, generation, source_image=source_image)

    @commands.command(name="update_data")
    @commands.is_owner()
    async def update_data_command(self, ctx: commands.Context) -> None:
        await ctx.message.add_reaction("\N{Thumbs Up Sign}")
        await self.update_data(force_update=True)

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError
    ) -> None:
        async def send_error_message(*args, **kwargs) -> None:
            if interaction.response.is_done():
                await interaction.edit_original_response(*args, **kwargs)
            else:
                await interaction.response.send_message(*args, **kwargs)
        generic_error_args = dict(content="Failed to generate image(s).")

        if isinstance(error, app_commands.CommandOnCooldown):
            await send_error_message("Command is on cooldown, please wait before trying again.", ephemeral=True)
            return

        if isinstance(error, InsufficientKudosError):
            await send_error_message(content="Failed to generate, operation was too expensive.")
            self.logger.debug("Aborted image generation due to excessive kudo cost.")
            return

        if isinstance(error, FaultedGenerationError):
            await send_error_message(**generic_error_args)
            self.logger.warn("Image generation failed, generation faulted.")
            return

        if isinstance(error, MissingGenerationsError):
            await send_error_message(**generic_error_args)
            self.logger.warn("Image generation failed. API didn't respond with any images.")
            return

        if isinstance(error, MaintenanceModeError):
            await send_error_message(content="Failed to generate, the API is in maintenance mode.")
            self.logger.debug("Aborted image generation due to maintenance mode.")
            return

        if isinstance(error, HordeAPIError):
            await send_error_message(**generic_error_args)
            self.logger.warn(
                f"Image generation failed. {error.__class__.__name__}"
                + (f": {error}" if str(error) else "")
            )
            return

    #TODO: Context menu entry (@app_commands.context_menu()) to "remix" a user's PFP using controlnet
    # first I'd want to break up the above monster method into individual parts
    # none of which should require an interaction to function


async def setup(bot: breadcord.Bot):
    await bot.add_cog(StableHorde("stable_horde"))
