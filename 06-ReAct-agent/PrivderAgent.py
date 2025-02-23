import json
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.agents import load_tools

load_dotenv()

# Initialize the LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo-16k",
)

api_wrapper = BingSearchAPIWrapper(k=1)
bingSearchAPI = BingSearchResults(api_wrapper=api_wrapper)

# Set up tools
baseTools = load_tools([], llm=llm)
tools = [
    Tool(
        name="BingSearchAPI",
        func=bingSearchAPI.run,
        description="Search for information using Bing Search API."
    )
] + baseTools

# Set up the prompt template
template = ('''
    Try your best to answer below questions. If you are not capable enough you can use the following tools:\n\n
    {tools}\n\n
    Use the following format:\n\n
    Question: the input question you must answer\n
    Thought: you should always think about what to do\n
    Action: the action to take, should be one of [{tool_names}]\n
    Action Input: the input to the action\n
    Observation: the result of the action\n
    ... (this Thought/Action/Action Input/Observation can repeat N times)\n
    Thought: I now know the final answer\n
    Final Answer: the final answer to the original input question\n\n
    Begin!\n\n
    Question: {input}\n
    Thought:{agent_scratchpad}
''')
prompt = PromptTemplate.from_template(template)

# Initialize the agent
agent = create_react_agent(llm, tools, prompt)

# Build the AgentExecutor
agent_executor = AgentExecutor(agent=agent, 
                               tools=tools, 
                               handle_parsing_errors=True,
                               verbose=True)

def process_providers(providers):
    updated_providers = []

    for provider in providers:
        provider_name = provider['providerName']
        emails = provider['email']
        emailsHashSet = set()
        for email in emails:
            emailsHashSet.add(email.split('@')[-1])           

        # Use the agent to search for the company name
        result = agent_executor.invoke({"input": f"What is the company name for the domain {emailsHashSet}? You can only return the company name, without any other information."})
        new_provider_name = result['output']

        # Use the agent to compare old and new provider names
        comparison_result = agent_executor.invoke({"input": f"Is '{provider_name}' refers to the same company with '{new_provider_name}'?, they don't have to be totally the same, just return yes if they are very similar in words and most likely refer to the same company, you can only return yes or no. If there is spelling error in old name, you should return no. \
                                                   If they just have different suffixes, like ABC vs abc,inc, you should return yes."})
        is_same_company = comparison_result['output'].strip().lower() == 'yes'

        # Update provider name if they are not the same
        if not is_same_company:
            provider['providerName'] = new_provider_name

        updated_providers.append(provider)

    return json.dumps(updated_providers, ensure_ascii=False, indent=4)

# Example usage
input_json = [
    {"providerName": "Arm Cloud Services", "email": ['abuse@pelion.com']},
    {"providerName": "Rogers", "email": ['ip.management@rci.rogers.com', 'ipmanage@rogers.wave.ca', 'labros.bakogiannis@rci.rogers.com', 'abuse@rogers.com', 'disna.kumanayake@rci.rogers.com', 'adam.pattenden@rci.rogers.com', 'muadh.ali@rci.rogers.com']}
]

output_json = process_providers(input_json)
print(output_json)