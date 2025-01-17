import argparse
import os
import sys
import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.ai.evaluation import evaluate, GroundednessEvaluator
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pathlib import Path
from chat_with_products import chat_with_products
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Force UTF-8 encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Load environment variables
load_dotenv()

def validate_file_encoding(file_path):
    """
    Validate and fix the encoding of the input dataset file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
    except UnicodeDecodeError:
        logger.warning(f"File {file_path} is not UTF-8 encoded. Attempting to fix encoding.")
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"File {file_path} encoding fixed to UTF-8.")
        except Exception as e:
            logger.error(f"Failed to fix encoding for {file_path}: {e}")
            sys.exit(1)

def evaluate_chat_with_products(query):
    """
    Evaluate chat responses using the target function.
    """
    try:
        response = chat_with_products(messages=[{"role": "user", "content": query}])
        return {"response": response["message"].content, "context": response["context"]["grounding_data"]}
    except Exception as e:
        logger.error(f"Error in evaluate_chat_with_products: {e}")
        raise

def validate_connection():
    """
    Validate the Azure connection and ensure all environment variables are properly set.
    """
    required_envs = [
        "AIPROJECT_CONNECTION_STRING",
        "EVALUATION_MODEL",
    ]
    for env in required_envs:
        if not os.getenv(env):
            logger.error(f"Environment variable {env} is not set. Please configure it.")
            sys.exit(1)

def main():
    """
    Main function to run evaluation.
    """
    # Validate Azure connection and environment variables
    validate_connection()

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Chat Responses")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the evaluation dataset (JSONL file).")
    parser.add_argument("--evaluation-name", type=str, required=True, help="Name for the evaluation.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the evaluation results.")
    args = parser.parse_args()

    # Validate dataset encoding
    validate_file_encoding(args.dataset_path)

    # Create a project client
    try:
        project = AIProjectClient.from_connection_string(
            conn_str=os.getenv("AIPROJECT_CONNECTION_STRING"), credential=DefaultAzureCredential()
        )
        connection = project.connections.get_default(
            connection_type=ConnectionType.AZURE_OPEN_AI, include_credentials=True
        )
    except Exception as e:
        logger.error(f"Failed to create project client: {e}")
        sys.exit(1)

    evaluator_model = {
        "azure_endpoint": connection.endpoint_url,
        "azure_deployment": os.getenv("EVALUATION_MODEL"),
        "api_version": "2024-06-01",
        "api_key": connection.key,
    }

    groundedness = GroundednessEvaluator(evaluator_model)

    # Run evaluation
    try:
        result = evaluate(
            data=Path(args.dataset_path),
            target=evaluate_chat_with_products,
            evaluation_name=args.evaluation_name,
            evaluators={
                "groundedness": groundedness,
            },
            evaluator_config={
                "default": {
                    "query": {"${data.query}"},
                    "response": {"${target.response}"},
                    "context": {"${target.context}"},
                }
            },
            azure_ai_project=project.scope,
            output_path=args.output_path,
        )

        # Output summarized metrics and results
        tabular_result = pd.DataFrame(result.get("rows"))
        print("-----Summarized Metrics-----")
        print(result["metrics"])
        print("-----Tabular Result-----")
        print(tabular_result)
        print(f"View evaluation results in AI Studio: {result['studio_url']}")
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decoding error during evaluation: {e}. Check dataset encoding.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
