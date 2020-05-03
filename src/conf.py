from dotenv import load_dotenv
from pydantic import BaseSettings

class Config(BaseSettings):
    ALLNLI_DEV_PATH: str
    ALLNLI_TRAIN_PATH: str
    CORS_ORIGIN: str
    DATA_FOLDER: str
    ENTAILMENT_MODEL: str
    ES_ENDPOINT: str
    LOG_REG_MODEL: str
    RTE_TEST_PATH: str
    RTE_TRAIN_PATH: str
    SPACY_NLP_MODEL: str
    WORD_EMBEDDINGS: str

    @staticmethod
    def load_config(env_path: str):
        load_dotenv(dotenv_path=env_path)
        return Config(_env_file=env_path)
