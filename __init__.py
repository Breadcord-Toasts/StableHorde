import asyncio
import io
import math
import random
import re
import time
from base64 import b64encode, b64decode
from collections import namedtuple

import aiohttp
import discord
import pydantic
from discord import app_commands
from discord.ext import tasks, commands
from discord.utils import escape_markdown

import breadcord
from .helpers.constants import HORDE_API_BASE
from .helpers.errors import (
    HordeAPIError,
    MaintenanceModeError,
    InsufficientKudosError,
    FaultedGenerationError,
    MissingGenerationsError,
    GenerationTimeoutError
)
from .helpers.types import *
from .helpers.utils import (
    fetch_styles,
    fetch_style_categories,
    modify_with_style,
    embed_desc_from_dict,
    fetch_loras,
    cb,
    clean_indented_string,
)

available_models: list[ActiveModel] = []
available_styles: dict[str, GenerationRequest] = {}
available_style_categories: dict[str, list[str]] = {}
available_loras: list[LoRA] = []


class DiffusionModelTransformer(app_commands.Transformer):
    def transform(self, interaction: discord.Interaction, value: str, /) -> ActiveModel | None:
        value = re.sub(r" \(\d+ workers\)$", "", value.strip())
        for model in available_models:
            if model.name.strip() == value:
                return model

    async def autocomplete(self, interaction: discord.Interaction, value: str, /) -> list[app_commands.Choice[str]]:
        def to_choice(model: ActiveModel) -> app_commands.Choice:
            versions = (
                f"{model.name} ({model.count} workers)",
                f"{model.name} ({model.count})",
                model.name,
                "⚠️ Model name too long ⚠️"
            )

            return app_commands.Choice(
                name=next(version for version in versions if len(version) <= 100),
                value=model.name,
            )

        processed_models = tuple(sorted(
            filter(is_diffusion_model, available_models),
            key=lambda m: m.count,
            reverse=True
        ))

        if not value:
            return [to_choice(model) for model in processed_models[:25]]

        models_matching_query = breadcord.helpers.search_for(
            query=value.strip(),
            objects=processed_models,
            key=lambda m: m.name,
            threshold=50
        )

        if not models_matching_query:
            return [app_commands.Choice(name="No models found", value="")]

        return [to_choice(model) for model in models_matching_query]


class StyleTransformer(app_commands.Transformer):
    def transform(self, interaction: discord.Interaction, value: str, /) -> str:
        value = value.removesuffix(" (category)").strip()
        if value in available_style_categories:
            value = random.choice(available_style_categories.get(value, []))
        return value

    async def autocomplete(self, interaction: discord.Interaction, value: str, /) -> list[app_commands.Choice[str]]:
        style_choices = (
            list(map(
                lambda style: (style, style),
                available_styles.keys()
            )) + list(map(
                lambda style: (f"{style} (category)", style),
                available_style_categories.keys()
            ))
        )

        return [
            app_commands.Choice(name=style[0], value=style[1])
            for style in breadcord.helpers.search_for(
                query=value,
                objects=style_choices,
                key=lambda style: style[0],
            )
        ]


class LoRATransformer(app_commands.Transformer):
    def transform(self, interaction: discord.Interaction, value: str, /) -> LoRA | None:
        value = value.strip()
        for lora in available_loras:
            if value in (lora.name, lora.actual_name.strip()):
                return lora

    async def autocomplete(self, interaction: discord.Interaction, value: str, /) -> list[app_commands.Choice[str]]:
        return [
            app_commands.Choice(name=lora.actual_name.strip(), value=lora.name.strip())
            for lora in breadcord.helpers.search_for(
                query=value,
                objects=available_loras,
                key=lambda lora: lora.actual_name
            )
        ]


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


class StableHorde(breadcord.module.ModuleCog):
    def __init__(self, module_id: str):
        super().__init__(module_id)

        self.session: aiohttp.ClientSession | None = None
        self.update_data.start()

        self.ctx_menu = app_commands.ContextMenu(
            name='Remix Avatar',
            callback=self.remix_user_avatar,
        )
        self.bot.tree.add_command(self.ctx_menu)

    async def cog_load(self) -> None:
        api_key: str = self.settings.stable_horde_api_key.value
        self.session = aiohttp.ClientSession(headers={"apikey": api_key})

    async def cog_unload(self):
        self.update_data.cancel()
        if self.session is not None:
            await self.session.close()
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)

    @tasks.loop(hours=12)
    async def update_data(self) -> None:
        self.logger.debug("Fetching horde models")
        async with self.session.get(f"{HORDE_API_BASE}/status/models", params={"type": "image"}) as response:
            global available_models
            available_models = list(map(
                lambda m: ActiveModel(**m),
                await response.json()
            ))
        self.logger.debug("Fetched horde models")

        global available_styles
        self.logger.debug("Loading styles")
        try:
            available_styles = await fetch_styles(
                session=self.session,
                storage_file_path=self.module.storage_path / "styles_cache.json",
            )
        except Exception as e:
            self.logger.error(f"Failed to load styles: {e}")
        else:
            self.logger.debug("Loaded styles")

        global available_style_categories
        self.logger.debug("Loading style categories")
        try:
            available_style_categories = await fetch_style_categories(
                session=self.session,
                storage_file_path=self.module.storage_path / "style_categories_cache.json",
            )
        except Exception as e:
            self.logger.error(f"Failed to load style categories: {e}")
        else:
            self.logger.debug("Loaded style categories")

        global available_loras
        self.logger.debug("Loading loras")
        try:
            available_loras = await fetch_loras(
                session=self.session,
                storage_file_path=self.module.storage_path / "lora_cache.json",
                logger=self.logger,
                max_count=int(self.settings.lora_limit.value)
            )
        except Exception as e:
            self.logger.error(f"Failed to load loras: {e}")
        else:
            self.logger.debug("Loaded loras")


    async def _generate_and_send(
        self,
        interaction: discord.Interaction,
        generation: GenerationRequest,
        *,
        source_image: discord.Attachment | discord.Asset | None = None,
    ) -> None:
        lora = generation.params.loras[0] if generation.params.loras else None
        control_type = generation.params.control_type
        generation.workers = await self.exclude_unavailable_workers(
            generation.workers or self.settings.workers.value
        ) or None

        if interaction.response.is_done():
            await interaction.edit_original_response(content="Starting to generate image...")
        else:
            await interaction.response.send_message("Starting to generate image...")

        kudo_cost = await generation.fetch_kudo_cost()
        if kudo_cost > float(self.settings.max_kudo_cost.value) and not await self.bot.is_owner(interaction.user):
            raise InsufficientKudosError()
        queued_generation = await generation.request_generation()

        # A generation request times out after 10 minutes
        for _ in range((10 * 60) // int(self.settings.horde_api_wait_time.value)):
            check = await queued_generation.check()
            if check.faulted:
                raise FaultedGenerationError()
            if check.done:
                break
            await interaction.edit_original_response(
                content="",
                embed=discord.Embed(
                    title="Generation status",
                    description=clean_indented_string(
                        f"""
                        **Estimated to be done <t:{round(time.time() + check.wait_time)}:R>**
                        **Position in queue:** {check.queue_position}
                        **Estimated kudo cost:** {kudo_cost}

                        ### Generation settings
                        **Prompt:** {escape_markdown(generation.positive_prompt)}
                        **Negative prompt:** {escape_markdown(generation.negative_prompt) or None}
                        **Marked as NSFW:** {generation.nsfw}
                        """
                    )
                ).set_thumbnail(
                    url=source_image.url if source_image else None
                )
            )
            await asyncio.sleep(int(self.settings.horde_api_wait_time.value))
        else:
            raise GenerationTimeoutError()

        generation_status = await queued_generation.status()
        if not generation_status.generations:
            raise MissingGenerationsError()

        required_deletion_votes = int(self.settings.required_deletion_votes.value)
        await interaction.edit_original_response(
            content="",
            embeds=[
                discord.Embed(
                    title="Generation complete",
                    description="\n".join(
                        line.lstrip() for line in f"""
                        {embed_desc_from_dict({
                            "Prompt": escape_markdown(generation.positive_prompt),
                            "Negative prompt": escape_markdown(generation.negative_prompt) or None,
                            "Seed": finished_gen.seed,
                            "Model": escape_markdown(finished_gen.model),
                            "Marked as NSFW": generation.nsfw,
                            "Lora": f"{escape_markdown(lora.actual_name)} ({lora.name})" if lora else None,
                            "ControlNet type": str(control_type).lower() if control_type else None,
                        }, bold_keys=True)}
                        ### **Horde metadata**
                        {embed_desc_from_dict({
                            "Finished by worker": f"{escape_markdown(finished_gen.worker_name)} "
                                                  f"(`{finished_gen.worker_id}`)",
                            "Total kudo cost": generation_status.kudos
                        }, bold_keys=True)}
                        """.splitlines()
                    ),
                ).set_author(
                    name=interaction.user.display_name,
                    icon_url=interaction.user.display_avatar.url
                ).set_thumbnail(
                    url=source_image.url if source_image else None
                )
                for finished_gen in generation_status.generations
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

    # noinspection LongLine
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
        style='A "preset" to use for generation',
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
        lora="What LoRA model to use. LoRAs allow embedding custom concepts into a generation",
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
        style: app_commands.Transform[str, StyleTransformer] = None,
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
        worker: str | None = None,  #TODO: Add a transformer
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
                workers=[worker] if worker else None,
                source_image=b64encode(await source_image.read()) if source_image else None,
                source_processing=source_processing if source_image else None,
                source_mask=b64encode(await source_mask.read()) if source_processing in [
                    SourceProcessors.INPAINTING,
                    SourceProcessors.OUTPAINTING
                ] and source_mask else None,
                params=GenerationParams(
                    cfg_scale=cfg_scale,
                    seed=seed,
                    steps=steps,
                    width=width,
                    height=height,
                    n=min(10, n), # Max attachments in a discord message
                    tiling=tiling,
                    hires_fix=hires_fix,
                    post_processing=post_processing,
                    control_type=control_type,
                    loras=[lora] if lora else None,
                    return_control_map=return_control_map
                ),
                shared=True,
                r2=False,
                replacement_filter=True,

                session=self.session
            )
        # This is the only validation error we want to ignore, since it happening depends on the user's reading skills
        except pydantic.ValidationError:
            await interaction.response.send_message("Invalid parameter(s).", ephemeral=True)
            return

        if style:
            generation = modify_with_style(generation, available_styles[style])

        await self._generate_and_send(interaction, generation, source_image=source_image)

    async def remix_user_avatar(self, interaction: discord.Interaction, user: discord.Member) -> None:
        self.logger.debug(f"Remixing {user.display_name}'s avatar")
        await interaction.response.send_message(f"Remixing {user.display_name}'s avatar...")

        avatar_resolution = 512
        avatar = user.display_avatar.with_format("webp").with_size(avatar_resolution)
        interrogation = InterrogationRequest(
            forms=[InterrogationForm(name=InterrogationType.CAPTION)],
            source_image=avatar.url,
            session=self.session
        )
        queued_interrogation = await interrogation.request_interrogation()

        # An interrogation request times out after 20 minutes
        for _ in range((20 * 60) // int(self.settings.horde_api_wait_time.value)):
            status = await queued_interrogation.fetch_status()
            if status.state == InterrogationStatuses.DONE:
                break
            await asyncio.sleep(int(self.settings.horde_api_wait_time.value))
        else:
            #TODO: Rename error so it fits better with interrogations AND image generations
            raise GenerationTimeoutError()
        prompt = next(iter(status.forms[0].result.values()))

        controlnet_model: ControlType = random.choice(list(map(
            ControlType,
            self.settings.avatar_remix_controlnet_models.value
        )))

        generation = GenerationRequest(
            positive_prompt=prompt,
            # models=[model.name] if model else None,
            source_image=b64encode(await avatar.read()),
            params=GenerationParams(
                steps=20,
                width=avatar_resolution,
                height=avatar_resolution,
                control_type=controlnet_model.value,
            ),
            shared=True,
            r2=False,
            replacement_filter=True,

            session=self.session
        )
        await self._generate_and_send(interaction, generation, source_image=avatar)

    @commands.hybrid_group(name="horde_info", description="Stable horde info commands")
    async def horde_info_group(self, ctx: commands.Context) -> None:
        await ctx.send_help(ctx.command)

    @horde_info_group.command(name="user")
    async def horde_user_info(self, ctx: commands.Context, user: str | None = None) -> None:
        try:
            if user is None:
                user_api_endpoint = "find_user"
            else:
                match = re.search(r"(\d+)$", user.strip())
                assert match is not None
                user_api_endpoint = f"users/{match[1]}"

            async with self.session.get(f"{HORDE_API_BASE}/{user_api_endpoint}") as response:
                assert response.status == 200
                user_data: dict = await response.json()
        except AssertionError:
            await ctx.reply("Could not find that user. Make sure that you are including the correct ID.")
            return

        kudo_details: dict = user_data["kudos_details"]
        records: dict = user_data["records"]
        created_at = int(time.time() - user_data['account_age'])

        await ctx.reply(
            embed=discord.Embed(
                title="Info about " + (f"the bot host ({user_data['username']})"
                                       if user is None else user_data["username"]),
                colour=discord.Colour.random(seed=user_data["id"]),
                description=clean_indented_string(f"""
                    {embed_desc_from_dict({
                        "Created at": f"<t:{created_at}:F> (<t:{created_at}:R>)",
                        "Trusted": cb(user_data["trusted"]),
                        "Flagged": cb(user_data["flagged"]) or None,
                        "Is a moderator": cb(user_data["moderator"]),
                        "Uses OAuth": cb(False) if user_data["pseudonymous"] else None,
                        "Max concurrent image requests": cb(user_data["concurrency"]),
                    })}
                
                    **Kudos**
                    {embed_desc_from_dict({
                        "Total": cb(int(user_data["kudos"])),
                        "Evaluating": cb(int(evaluating_kudos))
                                      if (evaluating_kudos := user_data.get("evaluating_kudos")) else None,
                        "Accumulated": cb(int(kudo_details["accumulated"])),
                        "Gifted": cb(int(kudo_details["gifted"])),
                        "Received": cb(int(kudo_details["received"])),
                        "Awarded": cb(int(kudo_details["awarded"])),
                        "Gifted by admins": cb(int(kudo_details["admin"])) or None,
                        "Monthly": cb(int(monthly))
                                   if (monthly := user_data.get("monthly_kudos", {}).get("amount")) else None,
                    })}
                
                    **Requested**
                    {embed_desc_from_dict({
                        "Images": f"{cb(records['request']['image'])} "
                                  f"({cb(records['usage']['megapixelsteps'])} megapixelsteps)",
                        "Texts": f"{cb(records['request']['text'])} "
                                 f"({cb(records['usage']['tokens'])} tokens)",
                        "Interrogations": cb(records["request"]["interrogation"]),
                    })}
                    
                    **Contributed**
                    {embed_desc_from_dict({
                        "Images": f"{cb(records['fulfillment']['image'])} "
                                  f"({cb(records['contribution']['megapixelsteps'])} megapixelsteps)",
                        "Texts": f"{cb(records['fulfillment']['text'])} "
                                 f"({cb(records['contribution']['tokens'])} tokens)",
                        "Interrogations": cb(records["fulfillment"]["interrogation"]),
                    })}
                    
                    **Workers**
                    {embed_desc_from_dict({
                        "Count": cb(user_data["worker_count"]),
                        "Invited": cb(user_data["worker_invited"]),
                        "Workers": ", ".join(str(cb(worker)) for worker in user_data.get("worker_ids", [])) or None,
                    })}
                """)
            )
        )

    @horde_info_group.command(name="worker")
    async def horde_worker_info(self, ctx: commands.Context, worker_id: str) -> None:
        async with self.session.get(f"{HORDE_API_BASE}/workers/{worker_id}") as response:
            if response.status != 200:
                await ctx.reply("Could not find that worker.")
                return
            worker_data: dict = await response.json()

        worker_performance = re.match(r"^(\d+\.\d+)", worker_data["performance"])[1]
        bridge_agent = namedtuple("BridgeAgent", ["name", "version", "url"])(
            *worker_data["bridge_agent"].split(":", maxsplit=2)
        )

        worker_type = worker_data["type"]
        if worker_type == "image":
            max_pixels = worker_data["max_pixels"]
            max_square_sides = math.floor(math.sqrt(max_pixels))

        # It complains about is accessing max_pixels and max_square_sides before assignment
        # But they will only be accessed if worker_type == "image"
        # noinspection PyUnboundLocalVariable
        await ctx.reply(embed=discord.Embed(
            title=f"Info about worker {worker_data['name']}",
            colour=discord.Colour.random(seed=worker_data["id"]),
            description=clean_indented_string(f"""
                {embed_desc_from_dict({
                    "ID": cb(worker_data["id"]),
                    "Is online": cb(worker_data["online"]),
                    "Accepts NSFW": cb(worker_data["nsfw"]),
                    "Trusted": cb(worker_data["trusted"]),
                    "Flagged": cb(worker_data["flagged"]) or None,
                    "Under maintenance": cb(worker_data["maintenance_mode"]) or None,
                    "Bridge agent": f"`{bridge_agent.name }` `v{bridge_agent.version}`",
                    "Worker type": cb(worker_data["type"]),
                    "Models": ", ".join(map(lambda model: str(cb(model)), worker_data["models"])) or "none",
                })}
                
                **Stats**
                {embed_desc_from_dict({
                    "Requests fulfilled": cb(worker_data["requests_fulfilled"]),
                    "Kudos rewarded": cb(worker_data["kudos_rewards"]),
                    "Megapixels generated": cb(worker_data["megapixelsteps_generated"]),
                    "Megapixelsteps per second": cb(worker_performance),
                    "Max pixels": cb(max_pixels) + f" (`{max_square_sides}x{max_square_sides}`)",
                    "Uncompleted jobs": cb(worker_data["uncompleted_jobs"]),
                }) if worker_type == "image" else embed_desc_from_dict({
                    "Requests fulfilled": cb(worker_data["requests_fulfilled"]),
                    "Kudos rewarded": cb(worker_data["kudos_rewards"]),
                    "Tokens generated": cb(worker_data.get("tokens_generated")),
                    "Tokens per second": cb(worker_performance),
                    "Max length": cb(worker_data["max_length"]),
                    "Max context length": cb(worker_data["max_context_length"]),
                    "Uncompleted jobs": cb(worker_data["uncompleted_jobs"]),
                })}
            """ + ("" if worker_type != "image" else f"""
                **Enabled features**
                {embed_desc_from_dict({
                    "Image to image": cb(worker_data["img2img"]),
                    "Inpainting": cb(worker_data["painting"]),
                    "LoRAs": cb(worker_data["lora"]),
                    "Post-processing": cb(worker_data["post-processing"]),
                })}
            """)),
        ))

    async def exclude_unavailable_workers(self, worker_ids: list[str]) -> list[str]:
        if not worker_ids:
            return []
        async with self.session.get(f"{HORDE_API_BASE}/workers") as response:
            available_workers = map(
                lambda worker: worker.get("name"),
                await response.json()
            )
        return list(filter(
            lambda worker: worker in available_workers,
            worker_ids
        ))

    async def cog_app_command_error(
        self,
        interaction: discord.Interaction,
        error: Exception
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

        if isinstance(error, GenerationTimeoutError):
            await send_error_message(content="Failed to generate, generation timed out.")
            self.logger.debug("Aborted image generation due to timeout.")
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
                + (f": {error}" if str(error) else "") + f" {error.status_code}"
            )
            return


async def setup(bot: breadcord.Bot):
    await bot.add_cog(StableHorde("stable_horde"))
