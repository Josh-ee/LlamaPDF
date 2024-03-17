import chainlit as cl
import asyncio
import re

# from llama_index import VectorStoreIndex, ServiceContext

import asyncio
import chromadb

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Prompt

from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager

from llama_index.llms.anthropic import Anthropic


from llama_index.core import Settings

import pandas as pd
import datetime
import re
import torch 

from dotenv import load_dotenv
import os
import nest_asyncio
nest_asyncio.apply()

from llama_parse import LlamaParse

load_dotenv()  # This loads the variables from .env

api_key = os.getenv('ANTHROPIC_API_KEY')

li_api_key = os.getenv('LLAMA_INDEX_API_KEY')




# Run Code: "chainlit run PDF_Pal2.py" in this directory

import yaml

# Load the configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Accessing the configuration items

EMBEDDING_MODEL_NAME = config['EMBEDDING_MODEL_NAME']
VECTOR_DB_PATH = config['VECTOR_DB_PATH']


class ChatBot:
    def __init__(self, memory_prompt, base_answer_prompt, rag_prompt):
        self.memory_prompt = memory_prompt
        self.base_answer_prompt = base_answer_prompt
        self.rag_prompt = rag_prompt
        

        self.chat_history = []
        self.question_count = 0

        # Detect hardware acceleration device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self._gpu_layers = 50 if self.device == 'cuda' else 1 if self.device == 'mps' else 0

        # Load Base LLM
        self.llm = Anthropic(temperature=0.0, model='claude-3-opus-20240229')


        
        self.embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, device=self.device, normalize='True')

        #Load Vector DB
        db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.vector_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("DB_collection"))
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        #Initiate the log file
        columns = ["Question", "LLM", "LLM with Sources", "SOURCE1", "SOURCE2",  "SOURCE3"]
        df = pd.DataFrame(columns=columns)
        self.out_file = 'output.csv' 
        df.to_csv(self.out_file, index=False)


    
    def new_chat(self):
        print("New Chat")
        self.chat_history = []
        self.question_count = 0

# Use MEM_PROMPT create a Standalone Question by combining USER QUESTION with the relevant CHAT HISTORY.
MEM_PROMPT = PromptTemplate("""
Your objective is to take in the USER QUESTION and fill in the missing context from the CHAT HISTORY.             
The question is always about the current pdf loaded in memory, do not modify acronyms and use FIRST and LAST Name.
DO NOT CHANGE THE QUESTION WORD: who, what, when, where, why, how.

                        
Here is an example of your task:
<EXAMPLE START>
CHAT HISTORY: [ChatMessage(role=<messagerole.user: 'user'>, content='Who was the first President of the US?'), chatmessage(role=<messagerole.assistant: 'assistant'>, content='\n  George Washington.')]
USER QUESTION: Where did he live?
STANDALONE QUESTION: Where did George Washington (first President of the US), live?
<EXAMPLE END>

<CHAT HISTORY>
{chat_history}
                            
<USER QUESTION>
{question}


<STANDALONE QUESTION>
""")



# If API cannot be reached
HYDE_PROMPT = PromptTemplate(""" 
The user will ask you about a academic paper loaded in memory, which you cannot see.
Your goal is provide a realistic sounding answer, which will then be used to search the paper
Provide a concise answer for the USER QUESTION about the paper
Only answer the user's exact question.
Keep your answer short, concise, and to the point.
Include as much context as possible in your answer, do not modify acronyms and use FIRST and LAST Name.

USER QUESTION: {question}

Concise Answer: """)


# QA Prompt pt 1: Courses API


# QA Prompt pt 2: RAG
RAG_PROMPT = PromptTemplate(""" 
You are PDF Pal, an AI Assistant for understanding academic papers, assume all questions are about the uploaded paper\n
Your goal is to answer the USER QUESTION with the most useful context below.
Only answer the user's exact question.
Keep your answer short, concise, and to the point.
                    
SOURCES:
{context_str}
            

USER QUESTION: {query_str}

ANSWER: """)

    

cal_chat_bot = ChatBot(MEM_PROMPT, HYDE_PROMPT, RAG_PROMPT)
print("Model Loaded")

@cl.on_chat_start
async def factory():
    Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()]) # This needs to be first
    Settings.llm = cal_chat_bot.llm
    
    Settings.embed_model = cal_chat_bot.embedding_model
    # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900


    files = None


    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["pdf"], max_size_mb=10
        ).send()

    #print((files))
    pdf_file = files[0]

    parser = LlamaParse(
        api_key=li_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4, # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en" # Optionaly you can define a language, default=en
    )

    # sync
    documents = parser.load_data(pdf_file.path)
    # print(documents)

    # nodes = parser.get_nodes_from_documents(documents)

    # index = VectorStoreIndex(nodes,service_context=Settings)

    index = VectorStoreIndex.from_documents(documents, service_context=Settings)



    # index = VectorStoreIndex.from_vector_store(
    #     cal_chat_bot.vector_store,
    #     storage_context = cal_chat_bot.storage_context,
    #     service_context=Settings,
    #     vector_store_query_mode="mmr",
    #     vector_store_kwargs={"mmr_threshold": 0.2}
        
    # )
    
    # Return the top 3 sources 
    retrieve_engine = index.as_retriever(similarity_top_k = 2,
                                node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.8, embed_model=cal_chat_bot.embedding_model)])   

    cal_chat_bot.new_chat()

    

    cl.user_session.set("llm", cal_chat_bot.llm)
    cl.user_session.set("retrieve_engine", retrieve_engine)
    # cl.user_session.set("ft_llm", cal_chat_bot.ft_llm)


@cl.on_message
async def main(message: cl.Message):
    #global HYDE_PROMPT, MEM_PROMPT, TEMPLATE_STR, COL_NAME
    llm = cl.user_session.get("llm")

    question = (message.content).strip()

    ### Find arxiv papers
    pattern = '(?:https?:\/\/)?arxiv\.org\/abs\/(\d{4}\.\d{4,5})(?:v\d+)?$'
    match = re.search(pattern, question)
    if match:
        arxiv_id = match.group(1)  # Return the matched arXiv ID
    else:
        arxiv_id = None
    
    if arxiv_id is not None:
        search = arxiv.Search(id_list=match)
        paper = next(arxiv.Client().results(search))
        paper_path = paper.download_pdf() 
        print(paper.title)

    
    if cal_chat_bot.question_count != 0:
        # print("rephrasing with MEM")
        rephrase_prompt = cal_chat_bot.memory_prompt.format(chat_history = cal_chat_bot.chat_history, question=question)
        question = await cl.make_async(llm.complete)(rephrase_prompt)
        question = (question.text).strip()

    full_question = cal_chat_bot.base_answer_prompt.format(question = question)
    # print(f"\nFULL QUESTION {full_question}\n")
    
    response = await cl.make_async(llm.complete)(full_question)

    raw_answer = response.text
    # print(f'Raw LLM answer: {raw_answer} \n')
    
    

    
    # If the model returned an empty string, search with the user question
    if len(raw_answer.strip()) == 0:
        raw_answer = question

    search_db_string = raw_answer # Here we are searching with the hypothetical embedding

    print(f'Searching with:\n {search_db_string}\n')
    retrieve_engine = cl.user_session.get("retrieve_engine") 
    retrieved_data = await cl.make_async(retrieve_engine.retrieve)(search_db_string)

    # print(retrieved_data)
    sources_dict = {} 
    for i in range(len(retrieved_data)):
        sources_dict[i] = retrieved_data[i].node.text
        # print(retrieved_data[i].node.text)
    # for i in range(len(retrieved_data)):
    #     text = f"Source Date: {retrieved_data[i].metadata['Updated']} \n {retrieved_data[i].text}\n\n"
    #     sources_dict[i] = {'URI': retrieved_data[i].metadata['URI'],
    #                         'topic': retrieved_data[i].metadata['Topic'],
    #                         'Updated': retrieved_data[i].metadata['Updated'],
    #                         'text': text
    #                         }
    
    # #below sorts sources by most recent to the top
    # for key, value in sources_dict.items():
    #     value["Updated"] = datetime.datetime.strptime(value["Updated"], "%Y-%m-%d")

    # # Sort the dictionary by date in descending order
    # sorted_sources = sorted(sources_dict.values(), key=lambda x: x["Updated"], reverse=True)

    sorted_sources_text = ''
    for k, v in sources_dict.items():
        sorted_sources_text += v
    
    rag_question = cal_chat_bot.rag_prompt.format(query_str = question, context_str = sorted_sources_text )        


    # response_string = ''
    
    
    response_message = cl.Message(content="")
    for response in await cl.make_async(llm.stream_complete)(rag_question):
        token = response.delta
        # print(token.replace('  ', ''), end='')
        await response_message.stream_token(token=token.replace('  ', ''))
        

    source_msg = '\n\n Sources: \n'
    printed_sources = []
    # for v in sorted_sources:
    #     print(v["URI"])
    #     if v["URI"] not in printed_sources:
    #         source_msg += f'\u00A0 \- [{v["topic"]}]({v["URI"]})\n'
    #     printed_sources.append(v["URI"])

    # for char in str(source_msg):
    #     await asyncio.sleep(0.01)
    #     await response_message.stream_token(token=char)

    # try:
    #     source1 = sorted_sources[0]['text']
    #     source2 = sorted_sources[1]['text']
    # #     source3 = sorted_sources[2]['text']
    # except IndexError:
    #     source1 = '**Empty DB used. Run Create_Vector_DB.py to load the DB**'
    #     for char in str(source1):
    #         await asyncio.sleep(0.01)
    #         await response_message.stream_token(token=char)
    #     source2 = source1
        # source3 = source2

    cal_chat_bot.chat_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content = question
            )
        )
    
    cal_chat_bot.chat_history.append(
        ChatMessage(
        role=MessageRole.ASSISTANT,
        content = response
        )
    )

    # Save to CSV
    new_row_df = pd.DataFrame({
        "Question": [question],
        "LLM": [raw_answer.strip()],
        "LLM with Sources": [response],
        # "SOURCE1": [source1],
        # "SOURCE2": [source2],
        # "SOURCE3": [source3]
    })
    new_row_df.to_csv(cal_chat_bot.out_file, mode='a', header=False, index=False)

    cal_chat_bot.question_count += 1

    # await response_message.send()

