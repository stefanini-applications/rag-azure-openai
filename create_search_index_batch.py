import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger

# Initialize logging object
logger = get_logger(__name__)

# Create a project client using environment variables
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

# Create a vector embeddings client
from embedding_client_openai import EmbeddingsClient
embeddings = EmbeddingsClient()

# Get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)

# Create a search index client
index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url, credential=AzureKeyCredential(key=search_connection.key)
)

# Import other dependencies
import pandas as pd
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    SearchIndex,
)


def create_index_definition(index_name: str, fields: list[str], vector_field: str, model: str) -> SearchIndex:
    """
    Generic function to create a search index definition.
    """
    dimensions = 1536  # Default for text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # Build fields dynamically
    search_fields = [SimpleField(name="id", type=SearchFieldDataType.String, key=True)]
    for field in fields:
        search_fields.append(SearchableField(name=field, type=SearchFieldDataType.String))

    # Add the vector field
    search_fields.append(
        SearchField(
            name=vector_field,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="myHnswProfile",
        )
    )

    # Vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4, ef_construction=1000, ef_search=1000, metric=VectorSearchAlgorithmMetric.COSINE
                ),
            )
        ],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    return SearchIndex(name=index_name, fields=search_fields, vector_search=vector_search)


def create_docs_from_file(path: str, content_column: str, fields: list[str], vector_field: str, model: str) -> list[dict]:
    """
    Generic function to create documents for indexing.
    """
    items = []

    if path.endswith(".csv"):
        data = pd.read_csv(path)
        for idx, row in data.iterrows():
            content = row[content_column]  # Content for embeddings
            emb = embeddings.embed(input=content, model=model)

            # Construct document dynamically
            rec = {"id": str(idx)}
            for field in fields:
                rec[field] = row.get(field, "")

            rec[vector_field] = emb["data"][0]["embedding"]
            items.append(rec)

    elif path.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        pdf_content = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = [pdf_content[i:i + 1000] for i in range(0, len(pdf_content), 1000)]

        for idx, chunk in enumerate(chunks):
            emb = embeddings.embed(input=chunk, model=model)
            rec = {
                "id": str(idx),
                content_column: chunk,
                vector_field: emb["data"][0]["embedding"]
            }
            items.append(rec)
    else:
        raise ValueError("Unsupported file format. Only CSV and PDF are supported.")

    return items


def upload_documents_in_batches(search_client, documents, batch_size=100):
    """
    Uploads documents to Azure Cognitive Search in batches to avoid size limits.
    """
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        try:
            result = search_client.upload_documents(batch)
            logger.info(f"Uploaded batch {i // batch_size + 1}: {len(batch)} documents successfully.")
        except Exception as e:
            logger.error(f"Batch {i // batch_size + 1} failed: {str(e)}")
            raise e


def create_index_from_file(index_name: str, file_path: str, content_column: str, fields: list[str], vector_field: str):
    """
    Creates an index and uploads documents from a generic file.
    """
    # Delete index if it exists
    try:
        index_client.delete_index(index_name)
        logger.info(f"Deleted existing index '{index_name}'")
    except Exception:
        pass

    # Create index definition
    index_definition = create_index_definition(index_name, fields, vector_field, os.environ["EMBEDDINGS_MODEL"])
    index_client.create_index(index_definition)

    # Create documents
    docs = create_docs_from_file(file_path, content_column, fields, vector_field, os.environ["EMBEDDINGS_MODEL"])

    # Upload documents in batches
    search_client = SearchClient(
        endpoint=search_connection.endpoint_url,
        index_name=index_name,
        credential=AzureKeyCredential(key=search_connection.key),
    )
    upload_documents_in_batches(search_client, docs)


if __name__ == "__main__":
    import argparse

    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Create a search index and upload documents to Azure AI Search.")

    # Required arguments
    parser.add_argument("--index-name", type=str, required=True, help="Name of the search index to create.")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the input file (CSV or PDF).")
    parser.add_argument("--content-column", type=str, required=True, help="Column or content field for generating embeddings.")
    parser.add_argument("--fields", type=str, required=True, help="Comma-separated list of fields to include in the index.")
    parser.add_argument("--vector-field", type=str, default="contentVector", help="Field name for vector embeddings. Default: contentVector")

    # Parse the arguments
    args = parser.parse_args()

    # Split fields into a list
    fields_list = [field.strip() for field in args.fields.split(",")]

    # Call the main function
    logger.info("Starting index creation process...")
    create_index_from_file(
        index_name=args.index_name,
        file_path=args.file_path,
        content_column=args.content_column,
        fields=fields_list,
        vector_field=args.vector_field,
    )
    logger.info("Index creation process completed successfully!")
