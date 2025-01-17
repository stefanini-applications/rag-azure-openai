# cost_calculation.py

# Preços dos modelos de Completion e Embedding
COMPLETION_MODELS = {
    "gpt-4o": 0.03,
    "gpt-4o-mini": 0.015,
    "gpt-o1": 0.01,
}

EMBEDDING_MODELS = {
    "text-embedding-ada-002": 0.0004,
    "text-embedding-3-large": 0.0008,
    "text-embedding-3-medium": 0.0006,
}

# Preços dos planos de Azure Search por unidade
SEARCH_PLANS = {
    "Basic": {"hourly": 0.125, "monthly": 91.25},
    "Standard S1": {"hourly": 0.50, "monthly": 365},
    "Standard S2": {"hourly": 1.0, "monthly": 730},
    "Standard S3": {"hourly": 2.0, "monthly": 1460},
    "Storage Optimized L1": {"hourly": 3.0, "monthly": 2190},
    "Storage Optimized L2": {"hourly": 4.0, "monthly": 2920},
}


def calculate_cost(tokens, model_price):
    """
    Calcula o custo para modelos de Completion ou Embedding.
    :param tokens: Número de tokens processados.
    :param model_price: Preço do modelo por 1.000 tokens.
    :return: Custo total.
    """
    return (tokens / 1000) * model_price


def calculate_search_cost(plan, hours):
    """
    Calcula o custo para o Azure Search com base no plano escolhido.
    :param plan: Plano escolhido (ex: "Basic", "Standard S1").
    :param hours: Número de horas usadas.
    :return: Custo total por hora.
    """
    if plan not in SEARCH_PLANS:
        raise ValueError(f"Plano '{plan}' não encontrado.")
    hourly_cost = SEARCH_PLANS[plan] * hours
    monthly_cost = SEARCH_PLANS[plan] * 730  # ~730 horas por mês
    return hourly_cost, monthly_cost


def calculate_total_cost(tokens, completion_model, embedding_model, search_plan, search_hours):
    # Compute costs
    completion_cost = calculate_cost(tokens, COMPLETION_MODELS[completion_model])
    embedding_cost = calculate_cost(tokens, EMBEDDING_MODELS[embedding_model])
    search_cost_hourly = SEARCH_PLANS[search_plan]["hourly"]
    search_cost_monthly = SEARCH_PLANS[search_plan]["monthly"]
    total_cost = completion_cost + embedding_cost + (search_cost_hourly * search_hours)

    # Return costs as a dictionary
    return {
        "completion_cost": completion_cost,
        "embedding_cost": embedding_cost,
        "search_cost_hourly": search_cost_hourly,
        "search_cost_monthly": search_cost_monthly,
        "total_cost": total_cost,
    }

# cost_calculation.py

import PyPDF2

# Token calculation function
def calculate_tokens(text):
    """
    Calculates the approximate number of tokens based on the text length.
    Assuming 4 characters per token as a rough estimate.
    :param text: Text input.
    :return: Approximate number of tokens.
    """
    return len(text) / 4

# Function to extract text from a PDF and calculate tokens
def calculate_pdf_tokens(file_path):
    """
    Extract text from a PDF file and calculate the number of tokens.
    :param file_path: Path to the PDF file.
    :return: Number of tokens.
    """
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text()
        # Calculate tokens from extracted text
        tokens = calculate_tokens(full_text)
        return tokens
    except Exception as e:
        raise ValueError(f"Error reading the PDF file: {e}")
