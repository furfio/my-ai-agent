# 导入环境变量
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
load_dotenv()

# 初始化大模型
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k"
)

api_wrapper = BingSearchAPIWrapper(k=1)
bingSearchAPI = BingSearchResults(api_wrapper=api_wrapper)

# 设置工具
from langchain.agents import load_tools
baseTools = load_tools(["serpapi","llm-math"], llm=llm)
# tools = [bingSearchAPI]
tools = [
    # Tool(
    #     name="BingSearchAPI",
    #     func=bingSearchAPI.run,
    #     description="Search for information using Bing Search API."
    # )
] + baseTools

# 设置提示模板
from langchain.prompts import PromptTemplate
template = ('''
    'Try your best to answer below questions。If you are not capable enough you can use the following tools:\n\n'
    '{tools}\n\n
    Use the following format:\n\n'
    'Question: the input question you must answer\n'
    'Thought: you should always think about what to do\n'
    'Action: the action to take, should be one of [{tool_names}]\n'
    'Action Input: the input to the action\n'
    'Observation: the result of the action\n'
    '... (this Thought/Action/Action Input/Observation can repeat N times)\n'
    'Thought: I now know the final answer\n'
    'Final Answer: the final answer to the original input question\n\n'
    'Begin!\n\n'
    'Question: {input}\n'
    'Thought:{agent_scratchpad}' 
    '''
)
prompt = PromptTemplate.from_template(template)

# 初始化Agent
agent = create_react_agent(llm, tools, prompt)

# 构建AgentExecutor
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               handle_parsing_errors=True,
                               verbose=True)

# 执行AgentExecutor
agent_executor.invoke({"input": 
                       """What is the price of bitcoin now?\n
                       What is the price if it raise 5%"""})
