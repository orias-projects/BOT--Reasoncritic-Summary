# Libraries/Processor_Models.py

from pathlib import Path

from . import Client_Llama

# ==============================

def llm_initialize(config: dict, llama_cpp_params: dict, base_models_dir: Path):
    """
    Initialize Local Llama HTTP Client instead of loading GGUF locally.
    """
    print("🔗 Using Local Llama HTTP server...")
    llm_client = Client_Llama.LocalLlamaClient("http://localhost:8080")
    return llm_client
