"""
title: Image Router Image Generator Pipe
author: Professor Patterns
description: Generate images using any Image Router model and display them in chat
version: 1.0.0
license: MIT
"""

import json
import traceback
import httpx
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Callable, Awaitable, AsyncGenerator
from enum import Enum

IMAGE_ROUTER_API_BASE_URL = "https://api.imagerouter.io/v1/openai"

class ImageQuality(str, Enum):
    auto = "auto"
    high = "high"
    medium = "medium"
    low = "low"


class Pipe:
    class Valves(BaseModel):
        IMAGE_ROUTER_API_KEY: str = Field(
            default="",
            title="Image Router API Key",
            description="Your Image Router API key for authentication"
        )
        QUALITY: ImageQuality = Field(
            default=ImageQuality.auto,
            title="Quality",
            description="Quality of the image (auto, high, medium, low)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.id = "imagerouter"
        self.name = "Image Router"
        self.emitter = None

    def pipes(self) -> List[dict]:
        """Return a pipe entry for every model returned by the Image Router API.

        The API response is a JSON object where the keys are model identifiers,
        e.g. "google/gemini-2.0-flash-exp:free".  We convert every key into a pipe
        description.  The *id* of each pipe is slug-ified so that it is safe to
        use in HTML and WebUI contexts, while the *name* keeps the original
        model identifier for better readability.

        If the request fails we fall back to a single default pipe so the
        extension keeps working offline.
        """

        # Simple in-memory cache to avoid querying the API on every call
        if hasattr(self, "_pipes_cache") and self._pipes_cache:
            return self._pipes_cache

        api_url = "https://api.imagerouter.io/v1/models"

        try:
            response = httpx.get(api_url, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            def _slugify(model_id: str) -> str:
                # Replace characters that might break ids ("/", " ", "::", etc.)
                return (
                    "imagerouter-" + model_id.lower().replace("/", "-").replace(" ", "-").replace(":", "-")
                )

            # Build list of pipes (only for IMAGE models) and a reverse lookup table for id -> model
            pipes_list = []
            self._id_to_model = {}

            for model_key, model_info in data.items():
                # The API includes both image and video models.  We only expose
                # those that are able to generate *images* so that they show up
                # in the WebUI image picker.  A model is considered an "image"
                # model if its "output" field (a list) contains the string
                # "image".

                outputs = model_info.get("output") or model_info.get("outputs")

                # Skip anything that does not explicitly list support for
                # images (e.g. video-only models)
                if outputs and "image" not in outputs:
                    continue

                pipe_id = _slugify(model_key)
                pipes_list.append({"id": pipe_id, "name": f" - {model_key}"})
                self._id_to_model[pipe_id] = model_key

            # Cache the result for subsequent calls
            self._pipes_cache = pipes_list

            return pipes_list

        except Exception as e:
            # Gracefully fall back to a default entry if the remote request fails
            print(f"[Image Router] Failed to fetch models list: {e}")
            return [
                {
                    "id": "test/test",
                    "name": " - test/test",
                }
            ]

    def _get_last_user_message(self, messages):
        """Extract the last user message from the conversation"""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    # Limit content to 30000 characters to stay safely under the limit
                    return content[:30000]
                elif isinstance(content, list):
                    # Handle content that might be a list of parts
                    for part in content:
                        if part.get("type") == "text":
                            # Limit content to 30000 characters to stay safely under the limit
                            return part.get("text", "")[:30000]
        return ""

    async def _emit_status(
        self, description: str, done: bool = False
    ) -> Awaitable[None]:
        """Send status updates"""
        if self.emitter:
            return await self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": description,
                        "done": done,
                    },
                }
            )
        return None

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate images using Image Router API and display them in chat"""
        self.emitter = __event_emitter__

        try:
            # Validate API key
            if not self.valves.IMAGE_ROUTER_API_KEY:
                yield json.dumps(
                    {"error": "Image Router API key is not configured. Get an API key: https://imagerouter.io/api-keys. Enter your API key in the Admin Panel -> Functions -> ImageRouter.io settings."}, ensure_ascii=False
                )
                return

            # Extract the prompt from the last user message
            last_user_message = self._get_last_user_message(body.get("messages", []))
            if not last_user_message:
                yield json.dumps({"error": "No user message found"}, ensure_ascii=False)
                return

            # Skip prompts that begin with the sentinel "### Task:" (often system-level instructions)
            if last_user_message.lstrip().startswith("### Task:"):
                # Silently ignore such prompts and exit early
                return

            # Send status update
            await self._emit_status(
                "Starting Image Router image generation..."
            )

            # Log what we're about to do
            print(f"Preparing to generate image with prompt: '{last_user_message}'")

            # Prepare request parameters
            headers = {
                "Authorization": f"Bearer {self.valves.IMAGE_ROUTER_API_KEY}",
                "Content-Type": "application/json",
            }

            # Resolve the target model.
            # 1. The caller SHOULD pass a canonical model string like "google/gemini-2.0-flash-exp:free".
            # 2. For backwards-compatibility we also accept slug-ified pipe ids such as
            #    "imagerouter-google-gemini-2.0-flash-exp-free" or those prefixed with
            #    "ir1." (e.g. "ir1.imagerouter-google-gemini-2.0-flash-exp-free").

            raw_model: Optional[str] = body.get("model")

            if not raw_model or not isinstance(raw_model, str):
                raise ValueError(
                    "No 'model' field found in the request body. "
                    "Please supply a valid model identifier (e.g. 'google/gemini-2.0-flash-exp:free')."
                )

            # If the supplied string contains a slug id, callers may prepend an arbitrary
            # prefix (e.g. "ir1.") before the actual slug that starts with "imagerouter-".
            # Remove anything before the first occurrence of "imagerouter-" so that we
            # are left with the pure slug id that exists in the cache.
            if "imagerouter-" in raw_model and not raw_model.startswith("imagerouter-"):
                raw_model = raw_model[raw_model.find("imagerouter-") :]

            if not hasattr(self, "_id_to_model") or not getattr(self, "_id_to_model", {}):
                # Attempt to populate the mapping; this can raise if the model list endpoint fails
                self.pipes()

            model_to_use = self._id_to_model.get(raw_model)

            if not model_to_use:
                raise ValueError(
                    f"Unknown model or pipe id '{body.get('model')}'. "
                    "Please provide a canonical model string or a recognised pipe id."
                )

            payload = {
                "prompt": last_user_message,
                "model": model_to_use,
                "quality": self.valves.QUALITY.value,
                "response_format": "url",
            }

            # Log the request that's being sent (removing the API key)
            debug_headers = headers.copy()
            if "Authorization" in debug_headers:
                debug_headers["Authorization"] = "Bearer [REDACTED]"
            print(
                f"Sending request to Image Router API: {IMAGE_ROUTER_API_BASE_URL}/images/generations"
            )
            print(f"Headers: {debug_headers}")
            print(f"Payload: {payload}")

            # Send request to Image Router API
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{IMAGE_ROUTER_API_BASE_URL}/images/generations",
                        json=payload,
                        headers=headers,
                        timeout=200.0,
                    )

                    print(f"Response status: {response.status_code}")
                    if response.status_code != 200:
                        error_text = response.text
                        print(f"Error response: {error_text}")
                        yield json.dumps(
                            {"error": f"Error generating image: {error_text}"},
                            ensure_ascii=False,
                        )
                        return
                except httpx.TimeoutException:
                    yield json.dumps(
                        {"error": "Request to Image Router API timed out. Please try again."},
                        ensure_ascii=False,
                    )
                    return
                except httpx.RequestError as e:
                    yield json.dumps(
                        {"error": f"Request error: {str(e)}"},
                        ensure_ascii=False,
                    )
                    return

                response_data = response.json()

                # For URL responses (shouldn't happen with our configuration)
                image_url = response_data["data"][0]["url"]
                yield f"🎨 Image generated successfully'\n\n"
                yield f"![Generated Image]({image_url})\n\n"
                # yield f"Note: This URL will expire in 60 minutes.\n\n"

                # Final status update
                await self._emit_status("Image generation complete", done=True)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            error_details = f"Exception type: {type(e).__name__}"
            stack_trace = traceback.format_exc()

            yield json.dumps(
                {
                    "error": error_message,
                    "details": error_details,
                    "trace": stack_trace,
                },
                ensure_ascii=False,
            )
            print(f"Error in Image Router Image pipe: {error_message} - {error_details}")
            print(f"Stack trace: {stack_trace}")
            await self._emit_status(f"Error: {str(e)}", done=True)
