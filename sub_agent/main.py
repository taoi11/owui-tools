"""
title: Sub-Agent Tools
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import aiohttp
import asyncio


class Tools:
    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()

    class Valves(BaseModel):
        """Admin-configurable settings."""

        base_url: str = Field(
            "http://litellm:4000/v1",
            description="LiteLLM API base URL (e.g., http://litellm:4000/v1)",
        )
        api_key: str = Field(
            "", description="API key for the LiteLLM endpoint (if needed)"
        )
        timeout_seconds: int = Field(60, description="Timeout for API calls in seconds")

    async def sub_agent(
        self,
        query: str,
        system_message: str = "You are a helpful assistant.",
        model: Optional[str] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Makes a call to big LLMs off premise to off load complex and beyond your capabilities tasks.

        :param query: The query/prompt to send to the LLM
        :param system_message: System message provides the agent with high level instructions.
        :param model: One of the following `sonnet-3.7` - Code generation agent; `sonar-pro` - internet search Agent for complex research, Fact-checking and fresh knowledge based tasks.
        """
        # Construct the complete chat completions API endpoint
        base_url = self.valves.base_url.rstrip("/")
        api_endpoint = f"{base_url}/chat/completions"
        api_key = self.valves.api_key
        timeout = self.valves.timeout_seconds

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Preparing to call external LLM...",
                        "done": False,
                    },
                }
            )

        # Prepare the API request
        headers = {"Content-Type": "application/json"}

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            "temperature": 0.6,
            "stream": False,
        }

        # Only add the model if it's provided
        if model:
            payload["model"] = model
            # Add high search context size for sonar models
            if model.startswith("sonar"):
                payload["web_search_options"] = {"search_context_size": "high"}
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Using model: {model}", "done": False},
                    }
                )

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Sending request to external LLM...",
                            "done": False,
                        },
                    }
                )

            # Use async HTTP client instead of synchronous requests
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_endpoint, headers=headers, json=payload, timeout=timeout
                ) as response:
                    if response.status == 200:
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": "Successfully received response from external LLM",
                                        "done": True,
                                    },
                                }
                            )

                        result = await response.json()
                        response_text = (
                            result.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        return response_text
                    else:
                        error_text = await response.text()
                        error_msg = f"API Error: {response.status} - {error_text}"

                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Error: {error_msg}",
                                        "done": True,
                                    },
                                }
                            )
                        return f"Error calling sub-agent: {error_msg}"

        except asyncio.TimeoutError:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: Request timed out",
                            "done": True,
                        },
                    }
                )
            return "Error: Request timed out"
        except aiohttp.ClientError as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"Error: {str(e)}", "done": True},
                    }
                )
            return f"Error: Request failed: {str(e)}"
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"An unexpected error occurred: {str(e)}",
                            "done": True,
                        },
                    }
                )
            return f"Error: An unexpected error occurred: {str(e)}"
