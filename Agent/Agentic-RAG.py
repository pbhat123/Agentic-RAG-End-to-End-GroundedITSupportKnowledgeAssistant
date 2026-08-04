import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from tools import SupportTicketTools

# Load Environment Variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY')
)

# Create Agent Tools
tool_manager = SupportTicketTools()
tools = tool_manager.get_tools()

# Convert tools to OpenAI function format
tool_definitions = []
for tool in tools:
    tool_definitions.append({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input to the tool"
                    }
                },
                "required": ["input"]
            }
        }
    })

# Bind tools to LLM
llm_with_tools = llm.bind(tools=tool_definitions)

print(f"✓ Created {len(tools)} tools:")
for tool in tools:
    print(f"  • {tool.name}: {tool.description.split('.')[0]}")

def run_agent(query: str, max_iterations: int = 5) -> str:
    """
    Runs a ReAct-style tool-calling loop until the model returns a final answer.
    Loop behavior:
    1) Model sees conversation + tool schema.
    2) Model either answers directly OR emits one/more tool calls.
    3) Execute each tool call, append ToolMessage results.
    4) Repeat until no tool calls remain or iteration cap is reached.
    """
    messages = [
        SystemMessage(content="""You are an expert support desk assistant that helps troubleshoot technical issues.

You have access to a database of previous support tickets with their resolutions.
Use your tools to find relevant information and provide helpful, accurate answers.

Guidelines:
- ALWAYS search for similar tickets when asked about troubleshooting or "how to fix" questions
- Be specific and reference ticket IDs when providing solutions
- If multiple similar issues exist, mention the most relevant ones
- Admit when you don't have enough information
- Be concise but thorough in your responses
- When appropriate, use multiple tools to gather complete information
- Before each tool call, provide a short public rationale in content using this exact prefix:
    "Decision: <one sentence explaining why this tool is needed>"

Remember: Your primary value is retrieving and applying solutions from past tickets!"""),
        HumanMessage(content=query)
    ]
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        # If the model produced no tool calls, treat content as final answer.
        if not response.tool_calls:
            # No more tool calls, return the response
            return response.content
        # Print model-provided rationale
        decision_trace = (response.content or "").strip()
        if decision_trace:
            print(f"\n🧭 {decision_trace}")
        # Execute each requested tool exactly as the model specified.
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            print(f"\n🔧 Calling tool: {tool_name}")
            print(f"   Input: {tool_input}")
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            print(f"   Output: {tool_output[:200]}...")
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    return "Maximum iterations reached. Could not complete the task."

# Selecting Tool 1 - search_similar_tickets
query1 = "How do I fix authentication problems after password reset?"
response1 = run_agent(query1)
print("\n\n" + "Agent's Final Response:")
print(response1)

# Selecting Tool 2 - get_ticket_by_id
query2 = "Show me details of ticket TICK-005"
response2 = run_agent(query2)
print("\n\n" + "Agent's Final Response:")
print(response2)

# Selecting Tool 3 - search_by_category
query3 = "What payment-related issues have we seen?"
response3 = run_agent(query3)
print("\n\n" + "Agent's Final Response:")
print(response3)

# Selecting Tool 4 - get_ticket_statistics
query4 = "Give me an overview of the ticket database"
response4 = run_agent(query4)
print("\n\n" + "Agent's Final Response:")
print(response4)

# Agent conducting Multi-Step Reasoning
query5 = "Find database-related critical issues and tell me how they were resolved"
response5 = run_agent(query5)
print("\n\n" + "Agent's Final Response:")
print(response5)

#Running Multi-Turn Agent with Memory that Remembers History
def run_conversational_agent(conversation_history, query: str, max_iterations: int = 5) -> tuple:
    messages = [SystemMessage(content="""You are an expert support desk assistant that helps troubleshoot technical issues.
Use your tools to find relevant information and maintain context across our conversation.
Before each tool call, provide a short public rationale in content using this exact prefix:
"Decision: <one sentence explaining why this tool is needed>".""")]
    # Replay prior turns and appending new user query.
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=query))
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return messages, response.content
        decision_trace = (response.content or "").strip()
        if decision_trace:
            print(f"\n🧭 {decision_trace}")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            print(f"\n🔧 Calling tool: {tool_name}")
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    return messages, "Maximum iterations reached."

# Start conversation
conversation = []
# Turn 1
conv_query1 = "What issues have we had with iOS?"
print(f"User: {conv_query1}\n")
conversation, conv_response1 = run_conversational_agent(conversation, conv_query1)
print(f"\nAssistant: {conv_response1}")
# Turn 2
conv_query2 = "What was the ticket ID for that?"
print(f"User: {conv_query2}")
conversation, conv_response2 = run_conversational_agent(conversation, conv_query2)
print(f"\nAssistant: {conv_response2}")
# Turn 3
conv_query3 = "How was it resolved?"
print(f"User: {conv_query3}\n")
conversation, conv_response3 = run_conversational_agent(conversation, conv_query3)
print(f"\nAssistant: {conv_response3}")

# Agent Evaluation
test_cases = [
    {
        "query": "How do I fix login issues?",
        "expected_tool": "SearchSimilarTickets",
        "should_contain": ["authentication", "TICK"]
    },
    {
        "query": "Show ticket TICK-001",
        "expected_tool": "GetTicketByID",
        "should_contain": ["TICK-001"]
    },
    {
        "query": "How many tickets are there?",
        "expected_tool": "GetTicketStatistics",
        "should_contain": ["total", "category"]
    }
]

for test in test_cases:
    response = run_agent(test["query"])
    print(test["query"])
    print(response[:150])