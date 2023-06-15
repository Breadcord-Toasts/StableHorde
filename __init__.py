import asyncio
import base64
import inspect
import io
import json
import time

import aiohttp
import discord
import pydantic
from discord import app_commands
from discord.ext import tasks

import breadcord
from .helpers.types import (
    GenerationRequest,
    GenerationParams,
    APIError,
    QueuedGeneration,
    GenerationCheck,
    GenerationStatus,
    ActiveModel,
    ModelType
)

available_models: list[ActiveModel] = []

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
        value = value.strip()
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


class StableHorde(breadcord.module.ModuleCog):
    def __init__(self, module_id: str):
        super().__init__(module_id)
        self.api_base = "/api/v2"

        self.session: aiohttp.ClientSession | None = None
        self.update_models.start()

    async def cog_load(self) -> None:
        self.session = aiohttp.ClientSession("https://stablehorde.net")

    async def cog_unload(self):
        self.update_models.cancel()
        if self.session is not None:
            await self.session.close()

    @tasks.loop(hours=24)
    async def update_models(self) -> None:
        async with self.session.get("/api/v2/status/models", params={"type": "image"}) as response:
            global available_models
            available_models = tuple(map(
                lambda m: ActiveModel(**m),
                await response.json()
            ))

    async def check_generation(
        self, queued_generation: QueuedGeneration, *, full_status: bool
    ) -> GenerationCheck | GenerationStatus:
        async with self.session.get(
            f"/api/v2/generate/{'status' if full_status else 'check'}/{queued_generation.id}"
        ) as response:
            response_json = await response.json()
            if response.status != 200:
                raise APIError(
                    f"API returned status code {response.status} "
                    + (f": {msg}" if (msg := response_json.get("message")) else "")
                )
            return GenerationStatus(**response_json) if full_status else GenerationCheck(**response_json)


    @app_commands.command()
    @app_commands.rename(
        nsfw="is_nsfw",
        n="image_count",
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
        n="The number of images to generate",
    )
    @app_commands.checks.cooldown(1, 10)
    async def imagine(
        self,
        interaction: discord.Interaction,
        positive_prompt: str,
        negative_prompt: str = "",
        model: app_commands.Transform[ActiveModel, DiffusionModelTransformer] = None,
        nsfw: bool = False,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: str | None = None,
        n: int = 1
        #TODO: add fields for the source image and mask
        #TODO: add support for ControlNet
        #TODO: add field for lora with a transformer
    ):
        positive_prompt, negative_prompt = positive_prompt[:1536], negative_prompt[:1536]
        try:
            generation = GenerationRequest(
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                nsfw=nsfw,
                models=[model.name] if model else None,
                params=GenerationParams(
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    n=n,
                ),
                shared=True,
                r2=False
            )
        except pydantic.ValidationError:
            await interaction.response.send_message("Invalid parameters.", ephemeral=True)
            return

        generation_json = json.loads(generation.model_dump_json(exclude_none=True, exclude_defaults=True))
        self.logger.debug(f"Generating image with JSON: {generation_json}")

        await interaction.response.send_message("Starting to generate image(s)...")
        async with self.session.post(
            "/api/v2/generate/async",
            headers={"apikey": self.settings.stable_horde_api_key.value},
            json=generation_json | {"dry_run": True}
        ) as response:
            response_json = await response.json()
            if (kudo_cost := response_json.get("kudos")) is None or response.status not in (200, 202):
                self.logger.warn(f"Image generation failed, got response: {response_json}")
                await interaction.edit_original_response(content="Failed to generate image(s).")
                return
            self.logger.debug(f"Trying to generate an image using {kudo_cost} kudos.")
            if kudo_cost > float(self.settings.max_kudo_cost.value) and not await self.bot.is_owner(interaction.user):
                await interaction.edit_original_response(content="Failed to generate, operation too expensive.")
                self.logger.debug("Aborted image generation due to excessive kudo consumption.")
                return

        async with self.session.post(
            "/api/v2/generate/async",
            headers={"apikey": self.settings.stable_horde_api_key.value},
            json=generation_json
        ) as response:
            if response.status != 202:
                self.logger.warn(f"Image generation failed, api responded with status code: {response.status}")
                await interaction.edit_original_response(content="Failed to generate image(s).")
                return
            response_json = await response.json()
            kudo_cost = response_json["kudos"]
            queued_generation = QueuedGeneration(
                id=response_json["id"],
                kudos=response_json["kudos"],
                message=response_json.get("message"),
            )

        # A generation request times out after 10 minutes
        for _ in range((10 * 60) // int(self.settings.time_between_updates.value)):
            try:
                check = await self.check_generation(queued_generation, full_status=False)
            except APIError as err:
                self.logger.warn(f"Image generation failed, api responded with: {err}")
                await interaction.edit_original_response(content="Failed to generate image(s).")
                return
            self.logger.debug(f"Generation check for {queued_generation.id}: {check}")
            if check.faulted:
                self.logger.warn("Image generation failed, generation faulted.")
                await interaction.edit_original_response(content="Failed to generate image(s).")
                return
            if check.done:
                break
            await interaction.edit_original_response(
                content="",
                embed=discord.Embed(
                    title="Generation Status",
                    description=inspect.cleandoc(
                        f"""
                        Estimated to be done <t:{round(time.time() + check.wait_time)}:R>
                        Position in queue: {check.queue_position}
                        Estimated kudo cost: {kudo_cost}
                        
                        **Generation settings**
                        Prompt: {generation.positive_prompt}
                        Negative prompt: {generation.negative_prompt}
                        Marked as NSFW: {generation.nsfw}
                        """
                    )
                )
            )
            await asyncio.sleep(int(self.settings.time_between_updates.value))

        generation_status: GenerationStatus = await self.check_generation(queued_generation, full_status=True)
        if not generation_status.generations:
            self.logger.warn("Image generation failed, API didn't respond with any images.")
            await interaction.edit_original_response(content="Failed to generate image(s).")
            return

        required_deletion_votes = int(self.settings.required_deletion_votes.value)
        await interaction.edit_original_response(
            content="",
            embeds=[
                discord.Embed(
                    title="Generation Status",
                    description=inspect.cleandoc(
                        f"""
                        Prompt: {generation.positive_prompt}
                        Negative prompt: {generation.negative_prompt}
                        Seed: {finished_generation.seed}
                        Model: {finished_generation.model}
                        Marked as NSFW: {generation.nsfw}
                        
                        **Horde metadata**
                        Finished by worker: {finished_generation.worker_name} (`{finished_generation.worker_id}`)
                        Total kudo cost: {generation_status.kudos}
                        """
                    ),
                ).set_image(
                    url=f"attachment://{'SPOILER_' if generation.nsfw else ''}{finished_generation.id}.webp"
                ).set_author(
                    name=interaction.user.display_name,
                    icon_url=interaction.user.display_avatar.url
                )
                for finished_generation in generation_status.generations
            ],
            attachments=[
                discord.File(
                    fp=io.BytesIO(base64.b64decode(finished_generation.img)),
                    filename=f"{'SPOILER_' if generation.nsfw else ''}{finished_generation.id}.webp"
                )
                for finished_generation in generation_status.generations
            ],
            view=DeleteButtonView(
                required_votes=min(
                    required_deletion_votes, len(set(filter(lambda m: not m.bot, interaction.channel.members)))
                ) if hasattr(interaction.channel, "members") else required_deletion_votes,
                author_id=interaction.user.id
            )
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(StableHorde("stable_horde"))
