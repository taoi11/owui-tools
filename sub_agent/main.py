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
            "https://openrouter.ai/api/v1",
            description="OpenRouter API base URL",
        )
        api_key: str = Field(
            "", description="API key for the LiteLLM endpoint (if needed)"
        )
        timeout_seconds: int = Field(60, description="Timeout for API calls in seconds")

    async def sub_agent(
        self,
        query: str,
        model: str,
        __event_emitter__=None,
    ) -> str:
        """
        Sends queries to powerful external large language models for tasks beyond your capabilities.

        Model Selection Guide:
        - Need to write complex code or solve advanced programming problems use "anthropic/claude-sonnet-4.5"
        - Need to search the web for current information, use "perplexity/sonar" with a simple question in the prompt
        - Need a well researched answer from the internet, use "perplexity/sonar-pro-search" with multiple simple but related questions or one complex question

        Best practices:
        - For code generation, specify language, frameworks, and expected functionality
        - For web searches, include specific keywords and time-sensitive context if relevant
        - Avoid chaining multiple unrelated topics in a single query

        :param query: The detailed instructions or question for the external model (required)
        :param model: OpenRouter model identifier (required)
        :return: The complete response from the external model
        """

        # Safety check: only allow predefined models
        allowed_models = [
            "anthropic/claude-sonnet-4.5",
            "perplexity/sonar",
            "perplexity/sonar-pro-search",
        ]
        
        if model not in allowed_models:
            return f"Error: Model '{model}' is not in the allowed list. Allowed models: {', '.join(allowed_models)}"
        
        # Define system messages based on the model
        if model == "anthropic/claude-sonnet-4.5":
            system_message = "You are an expert code writing llm sub_agent. You are called upon by a architect agent to produce code snippets. Provide clear, well-structured code solutions with concise and simple explanations. Follow best practices and write idiomatic code."
        elif model == "perplexity/sonar":
            system_message = "You are a helpful research llm sub_agent. You are called upon by a architect agent to provide accurate, current information from the web. Keep your responses concise, to the point, and token-efficient."
        elif model == "perplexity/sonar-pro-search":
            system_message = "You are an advanced research llm sub_agent. You are called upon by a architect agent to provide comprehensive, well-researched answers with detailed analysis. Keep your responses concise, to the point, and token-efficient."
        else:
            system_message = "You are a helpful AI assistant."
        
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
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
        }

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
