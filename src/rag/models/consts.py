import logging
from enum import StrEnum

logging_level_map = {
	"debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

class EnvKeys(StrEnum):
    LLM_API_KEY = "OPENAI_API_KEY"
    LLM_BASE_URL = "OPENAI_BASE_URL"
    LLM_MODEL = "OPENAI_MODEL"
    CHUNK_SIZE = "CHUNK_SIZE"
    CHUNK_OVERLAP = "CHUNK_OVERLAP"
    LOGGING_LEVEL = "LOGGING_LEVEL"
    EMBEDDING_API_KEY = "EMBEDDING_API_KEY"
    EMBEDDING_BASE_URL = "EMBEDDING_BASE_URL"
    EMBEDDING_MODEL = "EMBEDDING_MODEL"