
########### AINDA NÃO FUNCIONA, TEM QUE FAZER PELO PORTAL ########################




import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import *

load_dotenv()

projects_endpoint = os.environ["PROJECTS_ENDPOINT"]
project_name = os.environ["PROJECT_NAME"]

search_endpoint = os.environ["SEARCH_ENDPOINT"]
search_admin_key = os.environ["SEARCH_ADMIN_KEY"]
index_name = os.environ["INDEX_NAME"]

subscription_id = os.environ["SUBSCRIPTION_ID"]
resource_group_name = os.environ["RESOURCE_GROUP_NAME"]

# Use DefaultAzureCredential or another TokenCredential.
# Make sure 'az login' is done or environment variables for client credentials are set.
credential = DefaultAzureCredential()

# Instantiate the AiProjectClient
client = AIProjectClient(credential=credential, endpoint=projects_endpoint,subscription_id=subscription_id,resource_group_name=resource_group_name,project_name = project_name)

# Check existing data sources
data_sources = client.list_data_sources(project_name=project_name)
for ds in data_sources:
    if ds.get("type") == "AzureCognitiveSearch":
        print("An Azure Cognitive Search data source is already connected.")
        break
else:
    # Create a new Azure Cognitive Search data source
    connection_details = {
        "endpoint": search_endpoint,
        "apiKey": search_admin_key,
        "indexName": index_name
    }

    created_data_source = client.create_data_source(
        project_name=project_name,
        name="my-azure-search-data-source",
        connection_type="AzureCognitiveSearch",
        connection_details=connection_details
    )
    print(f"Data source '{created_data_source['name']}' created successfully.")
