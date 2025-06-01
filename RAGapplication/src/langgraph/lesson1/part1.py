
import sys
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
import random

from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="TRACE")

class State(TypedDict):
    """
    A class representing the name of a state.
    """
    name: str

def node1(state: State)-> State:
    """
    A function that takes a state and returns its name.
    """
    logger.info(f"node1 called with state: {state}")
    return {"name": state["name"] + " is the name of the state."}


def node2(state: State)-> State:
    """
    A function that takes a state and returns its name in uppercase.
    """
    logger.info(f"node2 called with state: {state}")
    return {"name": state["name"].upper() + " is the name of the state in uppercase."}

def node3(state: State)-> State:
    """
    A function that takes a state and returns its name in lowercase.
    """
    logger.info(f"node3 called with state: {state}")
    return {"name": state["name"].lower() + " is the name of the state in lowercase."}


def decide(state: State) -> Literal["node2", "node3"]:
    """
    A function that takes a state and returns a random node name.
    """
    logger.info(f"decide called with state: {state}")
    return random.choice(["node2", "node3"])

builder = StateGraph(State)
builder.add_node(node1, "node1")
builder.add_node(node2, "node2")
builder.add_node(node3, "node3")

builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

graph = builder.compile()

val = graph.invoke({"name" : "Hi, this is Anantha."})
print(val)