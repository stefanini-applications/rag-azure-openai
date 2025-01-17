import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Load environment variables
load_dotenv()


# Load OpenAI endpoint and API key from environment variables
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_KEY")

if not openai_endpoint or not openai_key:
    raise ValueError("Azure OpenAI endpoint or key not set in environment variables.")

# Create embeddings client manually
embeddings_client = AzureOpenAI(
    azure_endpoint=openai_endpoint,
    api_version="2024-02-01",
    api_key=openai_key
)

# Create a wrapper to generate embeddings (to mimic previous behavior)
class EmbeddingsClient:
    def embed(self, input: str, model: str):
        response = embeddings_client.embeddings.create(model=model, input=[input])
        return {"data": [{"embedding": response.data[0].embedding}]}


