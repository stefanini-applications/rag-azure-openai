import os
from pathlib import Path
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from config import ASSET_PATH, get_logger, enable_telemetry
from get_product_documents import get_product_documents
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

# Create a chat client
# --- Original line ---
# chat = project.inference.get_chat_completions_client()

# --- Updated for OpenAI compatibility ---
from chat_client_openai import ChatsClient
chat = ChatsClient()

from azure.ai.inference.prompts import PromptTemplate



@tracer.start_as_current_span(name="chat_with_products")
def chat_with_products(
    messages: list,
    context: dict = None,
    index_name: str = None,
    vector_field: str = "contentVector",
    select_fields: list = None,
    top: int = 5
) -> dict:
    """
    Processa uma conversa com base em documentos relevantes de um índice.

    Args:
        messages (list): Lista de mensagens do chat.
        context (dict): Contexto adicional.
        index_name (str): Nome do índice. Default: None.
        vector_field (str): Campo do vetor. Default: 'contentVector'.
        select_fields (list): Campos a serem retornados. Default: None.
        top (int): Número de documentos. Default: 5.

    Returns:
        dict: Resposta do modelo de chat.
    """
    if context is None:
        context = {}

    # Recupera documentos relevantes
    documents = get_product_documents(
        messages=messages,
        context=context,
        index_name=index_name,
        vector_field=vector_field,
        select_fields=select_fields,
        top=top,
    )

    # Gera o prompt para o modelo de chat
    grounded_chat_prompt = PromptTemplate.from_prompty(Path(ASSET_PATH) / "grounded_chat.prompty")
    system_message = grounded_chat_prompt.create_messages(documents=documents, context=context)

    # Realiza a conclusão do chat
    response = chat.complete(
        model=os.environ["CHAT_MODEL"],
        messages=system_message + messages
    )

    #logger.info(f"Chat response: {response.choices[0].message}")

    return response,context



if __name__ == "__main__":
    import argparse

    # Load command line arguments
    parser = argparse.ArgumentParser(description="Chat with products based on a search index.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to use to search documents (e.g., 'Quem é o relator do processo?')."
    )
    parser.add_argument(
        "--index-name",
        type=str,
        required=True,
        help="Name of the index to query (e.g., 'stf-pdf-index')."
    )
    parser.add_argument(
        "--vector-field",
        type=str,
        default="contentVector",
        help="Field name for vector embeddings in the index (default: 'contentVector')."
    )
    parser.add_argument(
        "--select-fields",
        type=str,
        default="id,content,page_number",
        help="Comma-separated list of fields to retrieve from the index (default: 'id,content,page_number')."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top documents to retrieve (default: 5)."
    )
    parser.add_argument(
        "--enable-telemetry",
        action="store_true",
        help="Enable sending telemetry back to the project."
    )
    args = parser.parse_args()

    if args.enable_telemetry:
        enable_telemetry(True)

    # Parse select_fields into a list
    select_fields = [field.strip() for field in args.select_fields.split(",")]

    # Run chat with products
    response,context = chat_with_products(
        messages=[{"role": "user", "content": args.query}],
        index_name=args.index_name,
        vector_field=args.vector_field,
        select_fields=args.select_fields.split(","),
        top=args.top
    )


    chat_response_content = response.choices[0].message.content
    
    print("Resposta:\n", chat_response_content)
    #print(context)

