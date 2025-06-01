
import os 
from dotenv import load_dotenv
from typing import Any

from exa_py import Exa
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

load_dotenv()


exa = Exa(api_key=os.environ['EXA_API_KEY'])
llm = ChatOpenAI(model="gpt-4o-mini")
memory = MemorySaver()

@tool
def search_and_content(query: str) -> Any:
    """Search and return content"""
    
    return exa.search_and_contents(
        query, use_autoprompt=True, num_results=5, text=True, highlights=True
    )


@tool 
def find_similar_and_contents(url:str) -> Any:
    """Search for webpages similar to a given URL and retrieve their contents"""
    
    return exa.find_similar_and_contents(
        url, use_autoprompt=True, num_results=5, text=True, highlights=True
    )

tools = [search_and_content, find_similar_and_contents]

llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="you're helpfull assistant")


class MessagesState(MessagesState):
    pass 

def assistant(state: MessagesState)-> MessagesState:
    """
    LLM function that processes user input and generates AI responses
    """

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)

builder.add_node('assistant',assistant)
builder.add_node('tools',ToolNode(tools))

builder.add_edge(START, 'assistant')
builder.add_conditional_edges('assistant', tools_condition)
builder.add_edge('tools', 'assistant')

graph = builder.compile()

print(graph.get_graph().draw_mermaid())

val = graph.invoke({"messages": [HumanMessage(content="What is cricket score IPL?")]})
print(val)