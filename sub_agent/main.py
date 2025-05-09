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
        Sends queries to powerful external large language models for tasks beyond your capabilities.
        
        Use this tool when you:
        - Need to write complex code or solve advanced programming problems use sonnet-3.7
        - Need to search the web for current information, use sonar-pro with a simple question in the prompt
        - Need a well researched answer from the internet, use sonar-pro with multiple questions in the prompt
        
        Best practices:
        - Keep system_message concise but specific about the desired output format and approach
        - For code generation, specify language, frameworks, and expected functionality
        - For web searches, include specific keywords and time-sensitive context if relevant
        - Avoid chaining multiple unrelated topics in a single query

        :param query: The detailed instructions or question for the external model
        :param model: one of `sonar-pro`, `sonar-3.7`
        :param system_message: Instructions that guide the external model's behavior and approach
        :return: The complete response from the external model
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
