import os

# Function to load environment variables from the .env file
def load_env(file_path=".env"):
    env_vars = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore comments and empty lines
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

# Function to update specific environment variables in the .env file
def update_env(updated_vars, file_path=".env"):
    env_vars = load_env(file_path)
    env_vars.update(updated_vars)  # Update only the specified keys
    with open(file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

# Function to get the current model configuration
def get_current_models(file_path=".env"):
    env_vars = load_env(file_path)
    return {
        "completion_model": env_vars.get("CHAT_MODEL", "gpt-4o"),  # Default to gpt-4-32k
        "embedding_model": env_vars.get("EMBEDDINGS_MODEL", "text-embedding-large"),  # Default to text-embedding-large-002
        "search_plan": env_vars.get("SEARCH_PLAN", "Basic"),
    }

# Function to update models
def update_models(completion_model, embedding_model, search_plan, file_path=".env"):
    updated_vars = {
        "CHAT_MODEL": completion_model,
        "EMBEDDINGS_MODEL": embedding_model,
        "SEARCH_PLAN": search_plan,
    }
    update_env(updated_vars, file_path)
