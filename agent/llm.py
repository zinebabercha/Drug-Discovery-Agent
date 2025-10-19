
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # OpenRouter-compatible LLM

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# Setup OpenRouter LLM (Strategist)
# -------------------------------
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from environment variables.")

strategist_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model_name="qwen/qwen3-235b-a22b-2507",  # Free/OpenRouter model
    temperature=0.1
)

# -------------------------------
# Setup Synthesizer LLM (Open-source / free)
# -------------------------------
# For fully free/open-source usage, you can also use OpenRouter smaller free models or local models
# Here we reuse Qwen 3 for demonstration (fully free if you have a free OpenRouter account)
synthesizer_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model_name="qwen/qwen3-32b",  # Smaller free model
    temperature=0.7
)

# -------------------------------
# Fallback LLM for Synthesizer
# -------------------------------
# In case synthesizer fails, fallback to strategist
synthesizer_fallback_llm = strategist_llm


# pip install python-dotenv
