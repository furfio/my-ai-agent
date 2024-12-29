from dotenv import load_dotenv
import os

load_dotenv()

Azure_OpenAI_Endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
Azure_OpenAI_Key = os.getenv("AZURE_INFERENCE_CREDENTIAL")

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="gpt-35-turbo-16k",
    api_key=Azure_OpenAI_Key,
    azure_endpoint=Azure_OpenAI_Endpoint,
    api_version="2024-08-01-preview",
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=Azure_OpenAI_Key,
    azure_endpoint=Azure_OpenAI_Endpoint,
    api_version="2023-05-15",
)

# configure Azure openai
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

# from llama_index.llms.openai import OpenAI
# llm = OpenAI(model="gpt-3.5-turbo-0613")

# 加载电商财报数据
from llama_index.core import SimpleDirectoryReader

# A_docs = SimpleDirectoryReader(
#     input_files=["C:\\testProjects\\ai-agents\\08-llamaindex-RAG-agent\\电商A-Third Quarter 2023 Results.pdf"]
# ).load_data()
# B_docs = SimpleDirectoryReader(
#     input_files=["C:\\testProjects\\ai-agents\\08-llamaindex-RAG-agent\\电商B-Third Quarter 2023 Results.pdf"]
# ).load_data()



# 从文档中创建索引
# from llama_index.core import VectorStoreIndex
# A_index = VectorStoreIndex.from_documents(A_docs)
# B_index = VectorStoreIndex.from_documents(B_docs)

# 持久化索引（保存到本地）
from llama_index.core import StorageContext
# A_index.storage_context.persist(persist_dir="./storage/A")
# B_index.storage_context.persist(persist_dir="./storage/B")


# 从本地读取索引
from llama_index.core import load_index_from_storage
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/A"
    )
    A_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/B"
    )
    B_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False


# 创建查询引擎
A_engine = A_index.as_query_engine(similarity_top_k=3)
B_engine = B_index.as_query_engine(similarity_top_k=3)


# 配置查询工具
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=A_engine,
        metadata=ToolMetadata(
            name="A_Finance",
            description=(
                "Finance info about Sea limited"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=B_engine,
        metadata=ToolMetadata(
            name="B_Finance",
            description=(
                "Finance info about Alibaba Group"
            ),
        ),
    ),
]

# 创建ReAct Agent
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)


# 让Agent完成任务
agent.chat("compare the revenue of the 2 company")
