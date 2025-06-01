One type of LLM application you can build is an agent. There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible.

Agent use LLM to select their own control flow of the system 


it is incredibly difficult to build systems that reliably execute on these tasks. As we’ve worked with our users to put agents into production, 
we’ve learned that more control is often necessary.

the control flow forms a chain (some steps before and after LLM Call)

Chains are realiable and same control flow every time but we want the LLM system that can pick their own control flow!

Agent ~= Control flow defined by LLM 

their are different kind of Agents 
1. Router (Less control)
2. Fully Autonomus (More control)


## tools
1. Tools are useful whenever you want a model to interact with external systems.
2. External systems (e.g., APIs) often require a particular input schema or payload, rather than natural language.
3. When we bind an API, for example, as a tool we given the model awareness of the required input schema.
4. The model will choose to call a tool based upon the natural language input from the user.

## Agent
1. Our chat model will decide to make a tool call or not based upon the user input
2. We use a conditional edge to route to a node that will call our tool or simply end
3. # act - let the model call specific tools
4. # observe - pass the tool output back to the model
5. # reason - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
