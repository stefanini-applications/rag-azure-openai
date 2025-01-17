import streamlit as st
import hashlib
from cost_calculation import (
        calculate_total_cost,
        calculate_cost,
        COMPLETION_MODELS,
        EMBEDDING_MODELS,
        SEARCH_PLANS,
        calculate_pdf_tokens,
        calculate_tokens
    )
import subprocess
import os
import json
from change_model import get_current_models, update_models
from environment_settings import get_all_env_vars, update_env



from dotenv import load_dotenv
load_dotenv()


# Set page configuration
st.set_page_config(
    page_title="Azure RAG Interface",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# User authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Sidebar Cost Initialization
if "completion_cost" not in st.session_state:
    st.session_state["completion_cost"] = 0.0
if "embedding_cost" not in st.session_state:
    st.session_state["embedding_cost"] = 0.0
if "search_cost_hourly" not in st.session_state:
    st.session_state["search_cost_hourly"] = 0.0
if "search_cost_monthly" not in st.session_state:
    st.session_state["search_cost_monthly"] = 0.0
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0

# Ensure chat history is initialized
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def load_users(file_path="users.json"):
    if not os.path.exists(file_path):
        st.error("Users file not found. Please create a valid users.json file.")
        return []
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error parsing users.json file. Please check the format.")
        return []

# Hash function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Check if a user exists and validate their credentials
def validate_user(users, username, password):
    for user in users:
        if user["user"] == username:
            hashed_input_pass = hash_password(password)
            stored_hashed_pass = hash_password(user["password"])  # Hash the stored password
            return hashed_input_pass == stored_hashed_pass
    return False

# Load users from JSON file
users = load_users()

# Login screen
if not st.session_state.get("authenticated"):
    st.title("Login to Azure RAG Interface")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        # Validate user credentials
        if validate_user(users, username, password):
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
else:
    # Logout option
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    # Model configuration
    st.title("Azure RAG Interface 📄")
    option = st.sidebar.radio(
        "Choose Action",
        ["Create Search Index", "Chat", "Query", "Evaluate", "List Indexes", "Environment Settings"]
    )

    # Sidebar Model Configuration
    st.sidebar.title("Model Configuration")
    st.sidebar.markdown("Configure the models used by the application.")

    # Inputs for model configuration
    current_models = get_current_models()
    completion_model = st.sidebar.selectbox(
        "Completion Model",
        COMPLETION_MODELS.keys(),
        index=list(COMPLETION_MODELS.keys()).index(current_models["completion_model"]),
    )
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        EMBEDDING_MODELS.keys(),
        index=list(EMBEDDING_MODELS.keys()).index(current_models["embedding_model"]),
    )
    search_plan = st.sidebar.selectbox(
        "Azure Search Plan",
        SEARCH_PLANS.keys(),
        index=list(SEARCH_PLANS.keys()).index(current_models["search_plan"]),
    )

    # Update models dynamically when user confirms
    if st.sidebar.button("Save Model Configuration"):
        update_models(completion_model, embedding_model, search_plan)
        st.success("Model configuration updated successfully!")

    # Display current configuration in the sidebar
    st.sidebar.markdown("### Current Configuration:")
    st.sidebar.text(f"Completion Model: {completion_model}")
    st.sidebar.text(f"Embedding Model: {embedding_model}")
    st.sidebar.text(f"Search Plan: {search_plan}")


    # Dynamic Cost Updates
    # Dynamic Cost Updates with Metrics
# Dynamic Cost Updates with Metrics in a Sidebar Container
# Sidebar Cost Updates (with a single container)
    def update_sidebar_cost(tokens_used=0, query_tokens=0, update_search=False,indexing=False):
        # Calculate Costs
        costs = calculate_total_cost(
            tokens=tokens_used + query_tokens,
            completion_model=completion_model,
            embedding_model=embedding_model,
            search_plan=search_plan,
            search_hours=24,  # Example value
        )

        # Update Session State (search costs are updated only once)
        if not indexing:
            st.session_state["completion_cost"] += costs["completion_cost"]
        st.session_state["embedding_cost"] += costs["embedding_cost"]
        if update_search:
            st.session_state["search_cost_hourly"] = costs["search_cost_hourly"]
            st.session_state["search_cost_monthly"] = costs["search_cost_monthly"]

        # Total cost should dynamically reflect the sum of completion and embedding costs
        st.session_state["total_cost"] = (
            st.session_state["completion_cost"]
            + st.session_state["embedding_cost"]
        )

        # Render the Sidebar with a Single Container
        with st.sidebar.container():
            st.subheader("Cost Estimation (Updated)")
            st.metric("Completion Cost", f"${st.session_state['completion_cost']:.4f}")
            st.metric("Embedding Cost", f"${st.session_state['embedding_cost']:.4f}")
            st.metric("Querying Cost", f"${(st.session_state['embedding_cost']+st.session_state['completion_cost']):.4f}")
            st.metric("Search Cost (Hourly)", f"${st.session_state['search_cost_hourly']:.2f}/hour")
            st.metric("Search Cost (Monthly)", f"${st.session_state['search_cost_monthly']:.2f}/month")
            st.metric("Total Cost", f"${st.session_state['total_cost']:.2f}")




    # "Create Search Index" Section
    if option == "Create Search Index":
        st.header("Create Search Index")
        index_name = st.text_input("Index Name", "rag-pdf-index")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        content_column = st.text_input("Content Column", "content")

        # Advanced options within "DEV Options"
        with st.expander("DEV Options"):
            fields = st.text_input("Fields (comma-separated)", "content,page_number")
            vector_field = st.text_input("Vector Field", "contentVector")

        tokens_used = 0  # Default
        if uploaded_file:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join("temp_files", uploaded_file.name)
            os.makedirs("temp_files", exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Estimate tokens based on the content length using the helper function
            try:
                tokens_used = calculate_pdf_tokens(temp_file_path)
                #st.info(f"Estimated Tokens for Embedding: {tokens_used}")
                
            except Exception as e:
                st.error(f"Error reading the PDF: {e}")

        if st.button("Create Index"):
            command = [
                "python", "create_search_index2.py",
                "--index-name", index_name,
                "--file-path", temp_file_path,
                "--content-column", content_column,
                "--fields", fields,
                "--vector-field", vector_field
            ]
            try:
                subprocess.run(command, check=True, text=True)
                st.success("Index created successfully!")
                update_sidebar_cost(tokens_used=tokens_used,indexing=True)  # Recalculate after success
            except subprocess.CalledProcessError as e:
                st.error(f"Error while creating index:\n{e.stderr}")

    # "Chat Interface" Section
    elif option == "Chat":
        st.header("Chat Interface")
        index_name = st.text_input("Index Name", "rag-pdf-index")
        query = st.text_input("Enter your query:", "Who is the relator of the process?")
        top = st.slider("Top Results", 1, 10, 5)
        st.markdown(f"**Selected Top Results:** {top}")

        # Advanced options within "DEV Options"
        with st.expander("DEV Options"):
            vector_field = st.text_input("Vector Field", "contentVector")
            select_fields = st.text_input("Select Fields (comma-separated)", "content,page_number")

        # Ensure chat history is initialized
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Calculate tokens for query and embedding
        query_tokens = calculate_tokens(query)
        #st.info(f"Estimated Tokens for Query: {query_tokens}")

        # Embedding cost for the query
        embedding_cost_query = calculate_cost(query_tokens, EMBEDDING_MODELS[embedding_model])

        # Display estimated embedding cost
        #st.info(f"Estimated Embedding Cost for Query: ${embedding_cost_query:.4f}")



        # Handle query submission
        if st.button("Send Query"):
            # Update sidebar cost with both query embedding and search costs
            update_sidebar_cost(query_tokens=query_tokens, tokens_used=query_tokens)
            command = [
                "python", "chat_with_products.py",
                "--index-name", index_name,
                "--query", query,
                "--vector-field", vector_field,
                "--select-fields", select_fields,
                "--top", str(top)
            ]
            try:
                # Get response from the subprocess
                result = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)

                # Add user query and assistant response to chat history
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": result.strip()})

                # Display query result
                st.markdown(f"### Assistant Response:\n{result.strip()}")

            except subprocess.CalledProcessError as e:
                st.error(f"Command failed with error:\n{e.output}")

        # Display chat history with improved styling
        st.subheader("Chat History")
        for message in st.session_state.chat_history:
            if message.get("role") == "user":
                st.markdown(
                    f"""
                    <div style="background-color: #f9f9f9; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <strong style="color: #333;">User:</strong>
                        <span style="color: #1a1a1a;">{message['content']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif message.get("role") == "assistant":
                st.markdown(
                    f"""
                    <div style="background-color: #dff5e1; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <strong style="color: #2c662d;">Assistant:</strong>
                        <span style="color: #1a1a1a;">{message['content']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # "Query" Section
    elif option == "Query":
        st.header("Query Interface")
        index_name = st.text_input("Index Name", "rag-pdf-index")
        query = st.text_input("Enter your query:", "Who is the relator of the process?")
        top = st.slider("Top Results", 1, 10, 1)
        st.markdown(f"**Selected Top Results:** {top}")

        with st.expander("DEV Options"):
            vector_field = st.text_input("Vector Field", "contentVector")
            select_fields = st.text_input("Select Fields (comma-separated)", "content,page_number")

        if st.button("Search"):
            command = [
                "python", "query_index.py",
                "--index-name", index_name,
                "--query", query,
                "--vector-field", vector_field,
                "--select-fields", select_fields,
                "--top", str(top)
            ]
            try:
                result = subprocess.check_output(command, text=True)
                st.markdown("### Query Results")
                st.markdown(f"📄 **Results:**\n{result}")
            except subprocess.CalledProcessError as e:
                st.error(f"Command failed with error:\n{e.stderr}")

    elif option == "Evaluate":
        st.header("Evaluate Chat Responses")

        # Inputs for evaluation
        dataset_path = st.text_input("Path to Evaluation Dataset", "assets/chat_eval_data.jsonl")
        evaluation_name = st.text_input("Evaluation Name", "evaluate_chat_with_products")
        output_path = st.text_input("Output Path for Results", "./myevalresults.json")

        if st.button("Run Evaluation"):
            try:
                # Prepare the command to run evaluate.py
                command = [
                    "python", "evaluate.py",
                    "--dataset-path", dataset_path,
                    "--evaluation-name", evaluation_name,
                    "--output-path", output_path,
                ]

                # Run the command and capture the output
                result = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)

                # Parse and display the results
                st.success("Evaluation completed successfully!")
                st.subheader("Evaluation Results")
                st.text(result)

            except subprocess.CalledProcessError as e:
                st.error(f"Evaluation failed with error:\n{e.output}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


    elif option == "List Indexes":
        st.header("List Search Indexes")

        if st.button("Fetch Indexes"):
            try:
                from azure.search.documents.indexes import SearchIndexClient
                from azure.core.credentials import AzureKeyCredential

                # Fetch environment variables
                search_admin_key = os.getenv("SEARCH_ADMIN_KEY")
                search_endpoint = os.getenv("SEARCH_ENDPOINT")

                if not search_admin_key or not search_endpoint:
                    st.error("SEARCH_ADMIN_KEY or SEARCH_ENDPOINT is not configured.")
                else:
                    # Initialize SearchIndexClient
                    index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_admin_key))

                    # Fetch all indexes
                    indexes = index_client.list_indexes()
                    index_names = [index.name for index in indexes]

                    # Display the index names
                    if index_names:
                        st.success(f"Found {len(index_names)} indexes:")
                        st.write(index_names)
                    else:
                        st.info("No indexes found.")
            except Exception as e:
                st.error(f"Error fetching indexes: {e}")
    




# Environment Settings Section
    elif option == "Environment Settings":
        st.header("Environment Settings")

        # Fetch current environment variables
        env_vars = get_all_env_vars()

        # Allow updating environment variables within an expandable section
        with st.expander("Environment Variables (Hidden by Default)"):
            st.subheader("Current Settings")
            updated_env_vars = {}
            for key in [
                "AIPROJECT_CONNECTION_STRING",
                "SEARCH_ADMIN_KEY",
                "SEARCH_ENDPOINT",
                "PROJECTS_ENDPOINT",
                "PROJECTS_KEY",
                "PROJECT_NAME",
                "SUBSCRIPTION_ID",
                "RESOURCE_GROUP_NAME",
                "AISEARCH_INDEX_NAME",
                "EMBEDDINGS_MODEL",
                "INTENT_MAPPING_MODEL",
                "CHAT_MODEL",
                "EVALUATION_MODEL",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_KEY",
            ]:
                current_value = env_vars.get(key, "")
                if "KEY" in key or "PASSWORD" in key or "CONNECTION_STRING" in key:
                    updated_env_vars[key] = st.text_input(key, current_value, type="password")
                else:
                    updated_env_vars[key] = st.text_input(key, current_value)

            if st.button("Update Settings"):
                try:
                    # Update the environment variables in the .env file
                    update_env(updated_env_vars)
                    st.success("Environment variables updated successfully!")
                except Exception as e:
                    st.error(f"Failed to update environment variables: {e}")

        st.markdown("---")

        # Display current models
        st.subheader("Models in Use")
        st.text(f"Embedding Model: {env_vars.get('EMBEDDINGS_MODEL', 'Not Set')}")
        st.text(f"Chat Model: {env_vars.get('CHAT_MODEL', 'Not Set')}")
        st.text(f"Intent Mapping Model: {env_vars.get('INTENT_MAPPING_MODEL', 'Not Set')}")
        st.text(f"Evaluation Model: {env_vars.get('EVALUATION_MODEL', 'Not Set')}")

        # Test and validate sensitive variables
        st.subheader("Test Connection")
        if st.button("Test Azure Connection"):
            try:
                from azure.search.documents.indexes import SearchIndexClient
                from azure.core.credentials import AzureKeyCredential

                # Initialize SearchIndexClient
                search_admin_key = env_vars.get("SEARCH_ADMIN_KEY")
                search_endpoint = env_vars.get("SEARCH_ENDPOINT")
                if not search_admin_key or not search_endpoint:
                    raise ValueError("SEARCH_ADMIN_KEY or SEARCH_ENDPOINT is missing.")

                index_client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_admin_key))

                # Test by listing indexes
                indexes = index_client.list_indexes()
                st.success("Azure connection is working! 🎉")
            except Exception as e:
                st.error(f"Azure connection test failed: {e}")


