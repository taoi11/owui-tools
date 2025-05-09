"""
title: Sub-Agent Tools
"""

from pydantic import BaseModel, Field
import aiohttp


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
        model: str,
        system_message: str,
        __event_emitter__=None,
    ) -> str:
        """
        Makes a call to big LLMs off premise to offload complex and beyond your capabilities tasks.

        :param query: The query/prompt to send to the LLM (required)
        :param model: Model to use, one of, 'sonnet-3.7' code generation agent; 'sonar-pro' internet search agent (required)
        :param system_message: System message provides the agent with high level instructions (required)
        :return: The response from the external LLM
        """
        # Construct the complete chat completions API endpoint
        api_endpoint = f"{self.valves.base_url.rstrip('/')}/chat/completions"
        
        # Simple status update for the user
        if __event_emitter__:
            await __event_emitter__({"type": "status", "data": {"description": f"Calling external LLM ({model})...", "done": False}})

        # Prepare the API request
        headers = {"Content-Type": "application/json"}
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            "temperature": 0.6,
            "model": model,
        }
        
        # Add high search context size for sonar models
        if model and model.startswith("sonar"):
            payload["web_search_options"] = {"search_context_size": "high"}

        try:
            # Use async HTTP client
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_endpoint, headers=headers, json=payload, 
                    timeout=self.valves.timeout_seconds
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Simple completion status
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": "Done", "done": True}})
                        
                        return result
                    else:
                        error_text = await response.text()
                        
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": "Error", "done": True}})
                        
                        return f"API Error: {response.status} - {error_text}"

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Error", "done": True}})
            return f"Request failed: {str(e)}"
