

import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai.chat_models import ChatOpenAI
from typing import TypedDict, Annotated
from loguru import logger

load_dotenv()

class MessageState(TypedDict):
    """
    A class representing the state of a message.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def node1(state: MessageState)-> MessageState:
    """
    A function that takes a state and returns its name.
    """
    logger.info(f"node1 called with state: {state}")
    return {"messages": state["messages"] + [AIMessage(content="node1 called after tool node", name="Anantha")]}


def tools_condition_llm(state: MessageState) -> str:
    """
    A function that takes a state and returns the name of the state.
    """
    logger.info(f"tools_condition called with state: {state}")
    messages = state["messages"]
    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls"):
        return "tools"
    return "node1"

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])


def tool_calling_llm(state: MessageState)-> MessageState:
    """
    A function that takes a state and returns the name of the state.
    """
    logger.info(f"tool_calling_llm called with state: {state}")
    messages = state["messages"]
    tool_call = llm_with_tools.invoke(messages)
    return {"messages": tool_call}

builder = StateGraph(MessageState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("node1", node1)
builder.add_node("tools", ToolNode([multiply]))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "node1") # my own node
builder.add_edge("node1", END)

graph = builder.compile()    

print(graph.get_graph().draw_mermaid())
val = graph.invoke({"messages": [HumanMessage(content="What is 2 multiplied by 3")]})
print(val)