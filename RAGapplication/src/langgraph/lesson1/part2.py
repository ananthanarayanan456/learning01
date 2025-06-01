
import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langchain_openai.chat_models import ChatOpenAI
from typing import TypedDict, Annotated
from loguru import logger

load_dotenv()

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Anantha"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Anantha"))

# llm = ChatOpenAI(model="gpt-4o-mini")
# result = llm.invoke(messages)

# logger.info(f"Result - content")
# print(result.content)

# logger.info(f"Result - metadata")
# print(result.response_metadata)

llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools([multiply])
tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Anantha")])
logger.info(f"Tool call - content")

print(tool_call.tool_calls)

class MessageState(TypedDict):
    """
    A class representing the state of a message.
    """
    messages: Annotated[list[AnyMessage], add_messages]





def tool_calling_llm(state: MessageState)-> MessageState:
    """
    A function that takes a state and returns the name of the state.
    """
    logger.info(f"tool_calling_llm called with state: {state}")
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}


builder = StateGraph(MessagesState)
builder.add_node(tool_calling_llm, "tool_calling_llm")

builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

graph = builder.compile()

val = graph.invoke({"messages": [HumanMessage(content="What is 2 multiplied by 3")]})
print(val)