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
    deployment_name="gpt-4o",
)

api_wrapper = BingSearchAPIWrapper(k=1)
bingSearchAPI = BingSearchResults(api_wrapper=api_wrapper)

# Set up tools
# baseTools = load_tools([], llm=llm)
# tools = [
#     Tool(
#         name="BingSearchAPI",
#         func=bingSearchAPI.run,
#         description="Search for information using Bing Search API."
#     )
# ] + baseTools
tools = [bingSearchAPI]

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

from utils import Provider

def process_providers(provider: Provider) -> Provider:

    provider_name = provider['providerName']
    emails = provider['emails']
    emailsHashSet = set()
    for email in emails:
        emailsHashSet.add(email.split('@')[-1])           

    # Use the agent to search for the company name
    result = agent_executor.invoke({"input": f"I'll give you a list of Email addresses: {emailsHashSet}, which belongs to a company or organization. \
                                    Then I'll give you a most-likely company name: {provider_name}, which might be out-of-date or incorrect.\
                                    You need to return the correct and update-to-date company/org name based on Email addresses and the most-likely answer. Do not include any extra text, explanations, or formatting—just return the company name as a single phrase."})

    new_provider_name = result['output']
    provider['newProviderName'] = new_provider_name

    # Use the agent to compare old and new provider names
    comparison_result = agent_executor.invoke({
        "input": f"Do '{provider_name}' and '{new_provider_name}' the same company name? \
                They don't have to be an exact match, return 'yes' if they are very similar (e.g., case differences, suffix changes like 'Inc.', 'Ltd.'). \
                However, return 'no' if they are completely different, or there is rebranding or acquisition between old and new, or the old name has spelling errors. \
                Only return 'yes' or 'no', with no extra text."
    })
    isNameChanged = comparison_result['output'].strip().lower() == 'no'

    provider['isNameChanged'] = isNameChanged

    # Update provider name if they are not the same
    if isNameChanged:
        provider['finalName'] = new_provider_name
    else:
        provider['finalName'] = provider_name

    return provider

# Example usage
input_json: Provider =  {
    "internalId": "f1f60605-9225-46e8-8f74-805e9e555763",
    "providerName": "Headquarters, USAISC",
    "asn": "1493",
    "emails": ["disa.columbus.ns.mbx.arin-registrations@mail.mil"]
}
output_json = process_providers(input_json)
print(output_json)