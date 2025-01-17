import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from config import get_logger
from embedding_client_openai import EmbeddingsClient
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential



# Load environment variables
load_dotenv()

# Initialize logging
logger = get_logger(__name__)

# Search connection
endpoint = os.getenv("SEARCH_ENDPOINT")
key = os.getenv("SEARCH_ADMIN_KEY")

# Initialize embedding client
embeddings = EmbeddingsClient()

def query_index(
    index_name: str,
    query: str,
    vector_field: str,
    select_fields: list[str],
    embedding_model: str = "text-embedding-3-large",
    top: int = 5
):
    """
    Generic function to query an Azure Search index using vector embeddings.

    Args:
        index_name (str): Name of the search index.
        query (str): User query.
        vector_field (str): Name of the vector field for the index.
        select_fields (list[str]): List of fields to return in the results.
        embedding_model (str): Embedding model to generate the query vector.
        top (int): Number of top results to return.

    Returns:
        list[dict]: List of documents matching the query.
    """
    #logger.info(f"Querying index '{index_name}' with query: '{query}'")

    # Initialize SearchClient for the specified index
    #search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=key)
    # Create a search index client using the search connection
    search_client = SearchClient(
        index_name=index_name,
        endpoint=endpoint,
        credential=AzureKeyCredential(key=key),
    )


  

    # Perform vector search
    #logger.info("Performing vector search...")
    if select_fields is None:
        select_fields = ["id", "content"]

    # Log the query
    #logger.info(f"Querying index '{index_name}' with query: '{query}'")

    # Generate the vector embedding for the query
    # Generate embedding for the query
    #logger.info(f"Generating embedding for query: {query}")
    embedding = embeddings.embed(model=embedding_model, input=query)
    search_vector = embedding["data"][0]["embedding"]

    # Perform the vector search
    vector_query = VectorizedQuery(vector=search_vector, k_nearest_neighbors=top, fields=vector_field)
    search_results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=select_fields,
        top=top
    )

    # Format the results into a list of documents
    documents = [
        {field: result.get(field, None) for field in select_fields}
        for result in search_results
    ]
    #logger.info(f"Retrieved {len(documents)} documents.")

    """
    for result in documents:
        print(f"Result: {result}")
        print("\n")
    """
    

    return documents


if __name__ == "__main__":
    import argparse

    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Query an Azure Search index with vector embeddings.")

    # Required arguments
    parser.add_argument("--index-name", type=str, required=True, help="Name of the search index.")
    parser.add_argument("--query", type=str, required=True, help="Query to search the index.")
    parser.add_argument("--vector-field", type=str, required=True, help="Name of the vector field in the index.")
    parser.add_argument("--select-fields", type=str, required=True, help="Comma-separated list of fields to retrieve.")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large", help="Embedding model to use.")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to retrieve (default: 5).")

    # Parse arguments
    args = parser.parse_args()

    # Parse select fields into a list
    select_fields_list = [field.strip() for field in args.select_fields.split(",")]

    # Query the index
    results = query_index(
        index_name=args.index_name,
        query=args.query,
        vector_field=args.vector_field,
        select_fields=select_fields_list,
        embedding_model=args.embedding_model,
        top=args.top
    )

    if not results:
        print("No results found.")
