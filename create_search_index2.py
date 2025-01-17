
import os
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger

# initialize logging object
logger = get_logger(__name__)

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)






# create a vector embeddings client that will be used to generate vector embeddings
#Esta como AI SERVICE, Mas queremos mudar para OPENAISERVICE, por isso tive que criar uma função auxiliar, embeddingc_client openai
#embeddings = project.inference.get_embeddings_client()
from embedding_client_openai import EmbeddingsClient
embeddings = EmbeddingsClient()






# use the project client to get the default search connection
search_connection = project.connections.get_default(
    connection_type=ConnectionType.AZURE_AI_SEARCH, include_credentials=True
)



# Create a search index client using the search connection
# This client will be used to create and delete search indexes
index_client = SearchIndexClient(
    endpoint=search_connection.endpoint_url, credential=AzureKeyCredential(key=search_connection.key)
)




## Defining search index

import pandas as pd
from azure.search.documents.indexes.models import (
    SemanticSearch,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
)






def create_index_definition(index_name: str, fields: list[str], vector_field: str, model: str) -> SearchIndex:
    """
    Generic function to create a search index definition.

    Args:
        index_name (str): Name of the index.
        fields (list[str]): List of field names to include in the index.
        vector_field (str): Name of the field for vector embeddings.
        model (str): Embedding model to determine dimensions.

    Returns:
        SearchIndex: The index definition.
    """
    dimensions = 1536  # Default for text-embedding-ada-002
    if model == "text-embedding-3-large":
        dimensions = 3072

    # Build fields dynamically
    search_fields = [SimpleField(name="id", type=SearchFieldDataType.String, key=True)]

    for field in fields:
        if field == "page_number":
            search_fields.append(SimpleField(name=field, type=SearchFieldDataType.Int32))  # Page number as integer
        else:
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

    # Define the vector search configuration
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
        profiles=[
            VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")
        ],
    )

    return SearchIndex(name=index_name, fields=search_fields, vector_search=vector_search)




from PyPDF2 import PdfReader

def create_docs_from_file(path: str, content_column: str, fields: list[str], vector_field: str, model: str) -> list[dict]:
    """
    Generic function to create documents for indexing.

    Args:
        path (str): Path to the input file (CSV or PDF).
        content_column (str): Column or content to generate embeddings.
        fields (list[str]): List of fields to include.
        vector_field (str): Name of the vector embeddings field.
        model (str): Embedding model to use.

    Returns:
        list[dict]: List of documents for Azure Search.
    """
    items = []

    if path.endswith(".csv"):
        data = pd.read_csv(path)
        for idx, row in data.iterrows():
            content = row[content_column]  # Content for embeddings
            emb = embeddings.embed(input=content, model=model)

            # Generate a valid id
            doc_id = f"{idx}".replace(".", "_").replace(" ", "_")

            # Construct document dynamically
            rec = {"id": doc_id}
            for field in fields:
                rec[field] = row.get(field, "")

            rec[vector_field] = emb["data"][0]["embedding"]
            items.append(rec)

    elif path.endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages, start=1):
            page_content = page.extract_text() or ""
            chunks = [page_content[i:i+1000] for i in range(0, len(page_content), 1000)]

            for chunk_idx, chunk in enumerate(chunks, start=1):
                emb = embeddings.embed(input=chunk, model=model)
                
                # Generate a valid id
                doc_id = f"{os.path.basename(path)}_page{page_num}_chunk{chunk_idx}".replace(".", "_").replace(" ", "_")

                # Create the document
                rec = {
                    "id": doc_id,
                    "page_number": page_num,  # Store the page number
                    content_column: chunk,
                    vector_field: emb["data"][0]["embedding"]
                }
                items.append(rec)
    else:
        raise ValueError("Unsupported file format. Only CSV and PDF are supported.")

    return items





def create_index_from_file(index_name: str, file_path: str, content_column: str, fields: list[str], vector_field: str):
    """
    Creates an index and uploads documents from a generic file.

    Args:
        index_name (str): Name of the index.
        file_path (str): Path to the input file.
        content_column (str): Column to generate embeddings.
        fields (list[str]): Fields to include in the index.
        vector_field (str): Name of the vector embeddings field.
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

    # Upload documents
    search_client = SearchClient(
        endpoint=search_connection.endpoint_url,
        index_name=index_name,
        credential=AzureKeyCredential(key=search_connection.key),
    )
    search_client.upload_documents(docs)
    logger.info(f"Uploaded {len(docs)} documents to '{index_name}' index")


if __name__ == "__main__":
    import argparse

    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Create a search index and upload documents to Azure AI Search.")

    # Required arguments
    parser.add_argument(
        "--index-name", type=str, required=True, help="Name of the search index to create."
    )
    parser.add_argument(
        "--file-path", type=str, required=True, help="Path to the input file (CSV or PDF)."
    )
    parser.add_argument(
        "--content-column", type=str, required=True, help="Column or content field for generating embeddings."
    )
    parser.add_argument(
        "--fields", type=str, required=True, help="Comma-separated list of fields to include in the index."
    )
    parser.add_argument(
        "--vector-field", type=str, default="contentVector", help="Field name for vector embeddings. Default: contentVector"
    )

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
