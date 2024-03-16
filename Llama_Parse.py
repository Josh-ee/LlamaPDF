import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env

li_api_key = os.getenv('LLAMA_INDEX_API_KEY')

parser = LlamaParse(
    api_key=li_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True
)

file_extractor = {".pdf": parser}



documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
print(documents)
