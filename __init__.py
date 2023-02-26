import asyncio
import re
from inspect import cleandoc

import aiohttp
import discord
from discord import app_commands
from discord.ext import tasks
from pydantic.utils import deep_update

import breadcord
from .helpers.types import (
    RequestFail,
    ImageRequestResponse,
    GenerationCheckResponse,
    GenerationStatusResponse,
)
from .helpers.utils import ImageGenerationInputParams, ImageGenerationInput, remove_payload_none_values, GeneratedImage


class DeleteButton(discord.ui.View):
    def __init__(self, required_votes: int):
        super().__init__()
        self.required_votes = required_votes
        self.votes = []

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.red, emoji="\N{WASTEBASKET}")
    async def delete(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if (user := interaction.user.id) not in self.votes:
            self.votes.append(user)

        vote_count = len(self.votes)
        button.label = f"Delete ({vote_count}/{self.required_votes})"

        if vote_count >= self.required_votes:
            return await interaction.response.edit_message(attachments=[], view=None)

        await interaction.response.edit_message(view=self)


class StableHorde(breadcord.module.ModuleCog):
    def __init__(self, module_id: str):
        super().__init__(module_id)
        self.module_settings = self.bot.settings.get_child(module_id)
        self.api_base = "/api/v2"

        self.session = None
        self.available_models: list[dict] = []
        self.update_models.start()

    async def cog_load(self) -> None:
        self.session = aiohttp.ClientSession("https://stablehorde.net")

    def cog_unload(self):
        self.update_models.cancel()
        if self.session is not None:
            self.session.close()

    @tasks.loop(hours=24)
    async def update_models(self) -> None:
        async with self.session.get(f"{self.api_base}/status/models", params={"type": "image"}) as response:
            self.available_models = await response.json()

    # noinspection PyUnusedLocal
    async def model_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        matches = []
        for model in self.available_models:
            name: str = f"{model['name']} ({model['count']} available)"
            if current.lower() in name.lower():
                matches.append(name)

        return [app_commands.Choice(name=match, value=match) for match in matches[:25]]

    async def request_image(self, input_params: ImageGenerationInput) -> ImageRequestResponse | RequestFail:
        payload_base = {
            "shared": True,
            "r2": False,
            "params": {
                "samples": 50,
                "cfg_scale": 7.5,
            },
        }
        payload = deep_update(payload_base, remove_payload_none_values(dict(input_params)))

        async with self.session.post(
            f"{self.api_base}/generate/async",
            headers={"apikey": self.module_settings.stable_horde_api_key.value},
            json=payload,
        ) as response:
            return await response.json()

    async def request_generation_status(
        self, generation_id: str, /, *, with_images: bool = False
    ) -> RequestFail | GenerationCheckResponse | GenerationStatusResponse:
        api_endpoint = f"{self.api_base}/generate/{'status' if with_images else 'check'}/{generation_id}"
        async with self.session.get(api_endpoint) as response:
            return await response.json()

    @staticmethod
    async def create_generating_embed(
        generation_data: GenerationCheckResponse,
        input_data: ImageGenerationInput,
        author: discord.User | discord.Member,
    ) -> discord.Embed:
        avatar = author.avatar

        embed = discord.Embed(
            title="Generating image...",
            description=cleandoc(
                f"""
                **Prompt:** {input_data.prompt}
                **Model:** {input_data.models[0] if input_data.models else 'any'}
                **Queue Position:** {generation_data["queue_position"]}
                **Estimated time left:** {generation_data['wait_time']}s
                """
            ),
        )
        embed.set_footer(text=f"Requested by: {author.name}", icon_url=avatar.url if avatar else None)
        return embed

    @staticmethod
    async def create_finished_embed(
        image_data: GeneratedImage,
        input_data: ImageGenerationInput,
        author: discord.User | discord.Member,
    ) -> discord.Embed:
        avatar = author.avatar

        embed = discord.Embed(
            title="Image generated.",
            description=cleandoc(
                f"""
                **Prompt:** {input_data.prompt}
                **Seed:** {image_data.seed}
                **Model:** {image_data.model.replace('_', ' ').replace('-', ' ').title()}
                **NSFW:** {input_data.nsfw or False}
                """
            ),
        )
        embed.set_footer(text=f"Requested by: {author.name}", icon_url=avatar.url if avatar else None)

        return embed

    async def generate_images(
        self,
        interaction: discord.Interaction,
        input_params: ImageGenerationInput,
    ) -> list[GeneratedImage, ...] | None:
        image_request = await self.request_image(input_params)
        if "id" not in image_request:
            return None
        uuid = image_request["id"]

        done = False
        while not done:
            await asyncio.sleep(3)
            data = await self.request_generation_status(uuid)
            if "message" in data or data["faulted"] or not data["is_possible"]:
                return None
            if done := data["done"]:
                data = await self.request_generation_status(uuid, with_images=True)

                return (
                    None
                    if "generations" not in data
                    else [GeneratedImage(**generated_image) for generated_image in data["generations"]]  # type: ignore
                )

            await interaction.edit_original_response(
                embed=await self.create_generating_embed(data, input_params, interaction.user)
            )

    @app_commands.command(description="Generate an image using AI")
    @app_commands.autocomplete(model=model_autocomplete)  # type: ignore
    @app_commands.describe(
        prompt="The prompt to feed to the image.",
        model="What model to use when generating the image",
        seed="The random seed the AI should use",
        cfg_scale="How much the Ai should follow your prompt, higher values means more accurate, but less reactive",
        should_tile="If the generated image should be tillable, mostly useful for textures",
        is_nsfw="If you think the image will be NSFW. THis will spoiler the output image",
        use_gfpgan="GFPGAN helps improve faces, mostly useful if you're going for a realistic look",
        steps="How many steps should be taken when generating the image",
    )
    async def ai_gen(
        self,
        interaction: discord.Interaction,
        prompt: str,
        model: str = "",
        seed: str | None = None,
        cfg_scale: float | None = None,
        should_tile: bool | None = None,
        is_nsfw: bool = False,
        use_gfpgan: bool | None = None,
        steps: int = 50,
    ):
        model = re.sub(r" \([0-9]+ available\)$", "", model)

        generation_input = ImageGenerationInput(
            prompt=prompt,
            models=[model] if model else None,
            nsfw=is_nsfw,
            censor_nsfw=False,
            params=ImageGenerationInputParams(
                steps=min(50, steps),
                seed=seed,
                cfg_scale=cfg_scale,
                tiling=should_tile,
                post_processing=["GFPGAN"] if use_gfpgan else None,
            ),
        )

        await interaction.response.send_message("Starting generation...")
        images = await self.generate_images(interaction, generation_input)
        if images is None:
            await interaction.edit_original_response(
                embed=discord.Embed(title="Generation failed.", colour=discord.Colour.red())
            )
            return

        embed = await self.create_finished_embed(images[0], generation_input, interaction.user)
        files = [
            discord.File(image.img, filename=f"{'SPOILER_' if generation_input.nsfw else ''}generated_image.webp")
            for image in images
        ]

        await interaction.edit_original_response(
            content="",
            embed=embed,
            attachments=files,
            view=DeleteButton(self.module_settings.required_deletion_votes.value),
        )


async def setup(bot: breadcord.Bot):
    await bot.add_cog(StableHorde("stable_horde"))
