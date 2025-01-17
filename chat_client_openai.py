from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve OpenAI endpoint, API key, model, and API version from environment variables
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")  # Default if not set

if not openai_endpoint or not openai_key:
    raise ValueError("Azure OpenAI endpoint or key is not set in environment variables.")

# Wrapper class to mimic the behavior of the AISERVICE chat client
class ChatsClient:
    def __init__(self):
        # Initialize the Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=openai_endpoint,
            api_version=api_version,
            api_key=openai_key
        )

    def complete(self, model: str, messages: list):
        """
        Generate a chat completion response.

        Parameters:
            model (str): Azure OpenAI model name (e.g., "gpt-35-turbo").
            messages (list): List of message dictionaries for the conversation.

        Returns:
            dict: Response from Azure OpenAI's chat completions API.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response

# Initialize the wrapper class
chat = ChatsClient()
