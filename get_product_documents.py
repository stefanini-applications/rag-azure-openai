import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from config import ASSET_PATH, get_logger
from azure.ai.inference.prompts import PromptTemplate
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize logging and tracing objects
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# Create a vector embeddings client and chat client
# --- Original lines ---
# chat = project.inference.get_chat_completions_client()
# embeddings = project.inference.get_embeddings_client()

# --- Updated for OpenAI compatibility ---
from chat_client_openai import ChatsClient
from embedding_client_openai import EmbeddingsClient

# Initialize the OpenAI-compatible clients
chat = ChatsClient()
embeddings = EmbeddingsClient()

# Use the project client to get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# Create a search index client using the search connection
search_client = SearchClient(
    index_name=os.environ["AISEARCH_INDEX_NAME"],
    endpoint=search_connection.endpoint_url,
    credential=AzureKeyCredential(key=search_connection.key),
)

from query_index import query_index  # Import the query_index function
@tracer.start_as_current_span(name="get_product_documents")
def get_product_documents(
    messages: list,
    context: dict = None,
    index_name: str = None,
    vector_field: str = "contentVector",
    select_fields: list = None,
    top: int = 5
) -> list[dict]:
    """
    Retrieves relevant documents based on the chat messages.

    Args:
        messages (list): Chat messages.
        context (dict): Additional context.
        index_name (str): Name of the index to query.
        vector_field (str): Field for vector search.
        select_fields (list): Fields to include in the result.
        top (int): Number of top results to return.

    Returns:
        list[dict]: List of retrieved documents.
    """
    if context is None:
        context = {}

    # Generate the search query from chat messages
    intent_prompty = PromptTemplate.from_prompty(Path(ASSET_PATH) / "grounded_chat.prompty")
    intent_mapping_response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=intent_prompty.create_messages(conversation=messages)
    )

    search_query = intent_mapping_response.choices[0].message.content
    logger.debug(f"🧠 Intent mapping: {search_query}")

    # Call the query_index function to perform the vector search
    documents = query_index(
        index_name=index_name,
        query=search_query,
        vector_field=vector_field,
        select_fields=select_fields,
        top=top
    )

    # Add results to the context
    context["thoughts"] = context.get("thoughts", [])
    context["thoughts"].append({
        "title": "Generated search query",
        "description": search_query,
    })

    context["grounding_data"] = context.get("grounding_data", [])
    context["grounding_data"].append(documents)

    logger.debug(f"📄 Retrieved {len(documents)} documents: {documents}")
    return documents




if __name__ == "__main__":
    import logging
    import argparse

    # Set logging level to debug when running this module directly
    logger.setLevel(logging.DEBUG)

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        help="Query to use to search product",
        default="I need a new tent for 4 people, what would you recommend?",
    )

    args = parser.parse_args()
    query = args.query

    result = get_product_documents(messages=[{"role": "user", "content": query}])
