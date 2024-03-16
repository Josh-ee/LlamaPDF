import nest_asyncio
nest_asyncio.apply()

import os
os.environ["LLAMA_CLOUD_API_KEY"] = ""

from llama_parse import LlamaParse
documents = LlamaParse(result_type="text").load_data("./attention.pdf")

print(documents[0].text[6000:7000])