""" 
title: Sub-Agent Tools
"""

from typing import Dict, List
from pydantic import BaseModel, Field
import requests

class Tools:
    def __init__(self):
        """Initialize the Tool."""
        self.valves = self.Valves()

    class Valves(BaseModel):
        """Admin-configurable settings."""
        base_url: str = Field("http://litellm:4000/v1", 
                              description="LiteLLM API base URL (e.g., http://litellm:4000/v1)")
        api_key: str = Field("", description="API key for the LiteLLM endpoint (if needed)")
        timeout_seconds: int = Field(60, description="Timeout for API calls in seconds")

    def sub_agent(
        self, 
        query: str, 
        system_message: str = "You are a helpful assistant.", 
        model: str = Field(
            None, 
            description="One of the following `sonnet-3.7`, `r1-1776`, `sonar-pro`, `sonar-deep-research`. "
        )
    ) -> str:
        """
        Makes a call to big LLMs off premise to off load complex and beyond your capabilities tasks.
        
        :param query: The query/prompt to send to the LLM
        :param system_message: System message to provide context to the model
        :param model: One of the following `sonnet-3.7` - Code generation tasks; `r1-1776` - Deep thoughtful reasoning; `sonar-pro` - Grounded by internet search. Fact-checking and fresh knowledge based tasks; `sonar-deep-research` - Grounded by internet search. optimized for deep research, analysis, and scholarly content.
        :return: The LLM's response text
        """
        # Construct the complete chat completions API endpoint
        base_url = self.valves.base_url.rstrip('/')
        api_endpoint = f"{base_url}/chat/completions"
        api_key = self.valves.api_key
        timeout = self.valves.timeout_seconds
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ],
            "temperature": 0.6,
            "stream": False
        }
        
        # Only add the model if it's provided
        if model:
            payload["model"] = model
        
        try:
            # Standard request
            response = requests.post(
                api_endpoint, 
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return response_text
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return f"Error calling sub-agent: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        
        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {str(e)}"
