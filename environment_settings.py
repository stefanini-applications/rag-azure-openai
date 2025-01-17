import os

# Load environment variables from the .env file
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

# Update specific environment variables in the .env file
def update_env(updated_vars, file_path=".env"):
    env_vars = load_env(file_path)
    env_vars.update(updated_vars)  # Update only the specified keys
    with open(file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

# Get all current environment variables
def get_all_env_vars(file_path=".env"):
    return load_env(file_path)
