from dotenv import load_dotenv
from pydantic import BaseSettings

class Config(BaseSettings):
    LOG_REG_MODEL: str
    NLP_MODEL: str
    ES_ENDPOINT: str

    @staticmethod
    def load_config(env_path: str):
        load_dotenv(dotenv_path=env_path)
        return Config(_env_file=env_path)
