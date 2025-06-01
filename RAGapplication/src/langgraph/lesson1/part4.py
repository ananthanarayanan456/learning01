from dotenv import load_dotenv
from typing import Literal, Any
from loguru import logger

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END


load_dotenv()

class MessagesState(MessagesState):
    """
    A class representing the state of a message.
    """
    ai_answer: str

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract a and b.

    Args:
        a: first int
        b: second int
    """
    return abs(a - b)

tools = [multiply, add, subtract]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_message = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def should_continue(state: MessagesState) -> Literal["tools", "assistant_tool"]:
    """
    Determines whether to use tools or proceed to assistant_tool
    """
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, 'tool_calls', None):
        return "tools"
    return "assistant_tool"

def assistant(state: MessagesState) -> MessagesState:
    """
    A function that processes user input and generates AI responses.
    """
    messages = state["messages"]
    logger.info(f"assistant called with {len(messages)} messages")
    
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        msg_content = getattr(msg, 'content', 'No content')
        logger.debug(f"  Message {i}: {msg_type} - {msg_content[:50]}...")
    

    result = llm_with_tools.invoke([sys_message] + messages)
    logger.debug(f"LLM returned: {result.content[:50]}... with tool_calls: {getattr(result, 'tool_calls', None)}")
    
    return {"messages": messages + [result]}

def assistant_tool(state: MessagesState) -> MessagesState:
    """
    Processes the AI response to extract meaningful content.
    """
    messages = state["messages"]
    logger.info(f"assistant_tool examining {len(messages)} messages")
    

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not getattr(message, 'tool_calls', None):
            logger.info(f"Found valid AI message: {message.content}")
            state["ai_answer"] = message.content
            return state
    

    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, 'tool_calls', None):
            tool_info = message.tool_calls[0] if message.tool_calls else None
            if tool_info:
                tool_name = tool_info.get('name', 'unknown')
                tool_args = tool_info.get('args', {})
                state["ai_answer"] = f"Used {tool_name} with {tool_args}"
                logger.info(f"Using tool call info: {state['ai_answer']}")
                return state
 
    logger.warning("No valid AI response or tool calls found")
    state["ai_answer"] = "No valid response found"
    return state

def play_with_ai_answer(state: MessagesState) -> MessagesState:
    """
    Formats the final answer for output.
    """
    if not state.get("ai_answer"):
        logger.warning("No AI answer found")
        state["ai_answer"] = "No answer available"
    
    logger.info(f"Final answer: {state['ai_answer']}")
    
    result = AIMessage(content=f"Final result: {state['ai_answer']}", name="Anantha")
    
    state["messages"].append(result)
    return state


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("assistant_tool", assistant_tool)
builder.add_node("play_with_ai_answer", play_with_ai_answer)
builder.add_node("tools", ToolNode(tools))

# Set up the graph flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_continue)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant_tool", "play_with_ai_answer")
builder.add_edge("play_with_ai_answer", END)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())
val = graph.invoke({"messages": [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]})

logger.debug("Final Message State:")
for i, message in enumerate(val['messages']):
    prefix = f"[{i}] "
    
    if isinstance(message, HumanMessage):
        logger.debug(f"{prefix}Human: {message.content}")
        print(f"{prefix}Human: {message.content}")
    
    elif isinstance(message, AIMessage):
        name_str = f" ({message.name})" if hasattr(message, 'name') and message.name else ""
        logger.debug(f"{prefix}AI{name_str}: {message.content}")
        print(f"{prefix}AI{name_str}: {message.content}")
        
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_str = ", ".join([f"{t['name']}({t['args']})" for t in message.tool_calls])
            logger.debug(f"{prefix}Tool calls: {tool_str}")
            print(f"{prefix}Tool calls: {tool_str}")
    
    else:
        logger.debug(f"{prefix}Other: {type(message).__name__}")
        print(f"{prefix}Other: {type(message).__name__}")

# Print final answer for clarity
print("\nFINAL RESULT:")
for message in reversed(val['messages']):
    if isinstance(message, AIMessage) and getattr(message, 'name', '') == 'Anantha':
        print(message.content)
        break