import os
import json
import re
import time
import argparse
from pathlib import Path
from statistics import mean
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
REPO_ROOT = Path(__file__).resolve().parent.parent
tool_manager = SupportTicketTools(
    tickets_path=str(REPO_ROOT / "data" / "synthetic_tickets.json")
)
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

def _token_usage(message) -> dict:
    """Normalize token usage across LangChain/OpenAI response formats."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        usage = getattr(message, "response_metadata", {}).get("token_usage", {})
    return {
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "usage_available": bool(usage),
    }


def _add_usage(total: dict, message) -> None:
    usage = _token_usage(message)
    total["input_tokens"] += usage["input_tokens"]
    total["output_tokens"] += usage["output_tokens"]
    total["total_tokens"] += usage["total_tokens"] or (
        usage["input_tokens"] + usage["output_tokens"]
    )
    total["usage_available"] = total["usage_available"] and usage["usage_available"]


def run_agent_with_trace(query: str, max_iterations: int = 5) -> dict:
    """Run the agent and return its answer plus evidence, timing, and token usage."""
    started_at = time.perf_counter()
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_available": True,
    }
    tool_trace = []
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
        _add_usage(usage, response)
        messages.append(response)
        # If the model produced no tool calls, treat content as final answer.
        if not response.tool_calls:
            # No more tool calls, return the response
            return {
                "answer": response.content,
                "tool_trace": tool_trace,
                "latency_seconds": time.perf_counter() - started_at,
                "token_usage": usage,
                "completed": True,
            }
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
            tool_trace.append({
                "tool": tool_name,
                "input": tool_input,
                "output": tool_output,
            })
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    return {
        "answer": "Maximum iterations reached. Could not complete the task.",
        "tool_trace": tool_trace,
        "latency_seconds": time.perf_counter() - started_at,
        "token_usage": usage,
        "completed": False,
    }


def run_agent(query: str, max_iterations: int = 5) -> str:
    """Backward-compatible answer-only wrapper around the traced agent run."""
    return run_agent_with_trace(query, max_iterations=max_iterations)["answer"]

def run_conversational_agent(conversation_history, query: str, max_iterations: int = 5) -> tuple:
    """Run one conversational turn while retaining prior messages."""
    messages = [SystemMessage(content="""You are an expert support desk assistant that helps troubleshoot technical issues.
Use your tools to find relevant information and maintain context across our conversation.
Before each tool call, provide a short public rationale in content using this exact prefix:
"Decision: <one sentence explaining why this tool is needed>".""")]
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=query))

    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return messages, response.content
        for tool_call in response.tool_calls:
            tool_output = None
            for tool in tools:
                if tool.name == tool_call["name"]:
                    tool_output = tool.func(tool_call["args"].get("input", ""))
                    break
            messages.append(ToolMessage(
                content=tool_output or f"Error: Tool {tool_call['name']} not found",
                tool_call_id=tool_call["id"],
            ))
    return messages, "Maximum iterations reached."


def _retrieval_metrics(retrieved_ids: list, relevant_ids: list) -> dict:
    """Calculate label-based Precision@k, Recall@k, F1@k, MRR, and Hit@k."""
    relevant = set(relevant_ids)
    retrieved = list(dict.fromkeys(retrieved_ids))
    true_positives = len(relevant.intersection(retrieved))
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(relevant) if relevant else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0
        else 0.0 if recall is not None else None
    )
    reciprocal_rank = 0.0
    for rank, ticket_id in enumerate(retrieved, start=1):
        if ticket_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "reciprocal_rank": reciprocal_rank,
        "hit_at_k": float(true_positives > 0),
    }


def _parse_json_object(text: str) -> dict:
    """Parse a judge response without silently manufacturing scores."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response was not a JSON object")
    return parsed


def _judge_answer(question: str, answer: str, reference_answer: str, evidence: str) -> dict:
    """Use an LLM judge for semantic metrics that cannot be derived from ID labels."""
    judge = ChatOpenAI(
        model=os.getenv("OPENAI_JUDGE_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = f"""Evaluate an IT-support answer using only the supplied inputs.

Definitions:
- groundedness: fraction from 0 to 1 of substantive answer claims supported by EVIDENCE. Penalize unsupported claims.
- completeness: fraction from 0 to 1 of material points in REFERENCE ANSWER covered by ANSWER.
- task_success: true only if ANSWER is relevant, correct relative to the reference, sufficiently complete, and grounded.

Return strict JSON only with this schema:
{{"groundedness": 0.0, "completeness": 0.0, "task_success": false,
  "groundedness_reason": "...", "completeness_reason": "...", "task_success_reason": "..."}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

EVIDENCE:
{evidence}

ANSWER:
{answer}"""
    response = judge.invoke([HumanMessage(content=prompt)])
    result = _parse_json_object(response.content)
    for metric in ("groundedness", "completeness"):
        value = result.get(metric)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Judge returned invalid {metric}: {value}")
    if not isinstance(result.get("task_success"), bool):
        raise ValueError("Judge returned invalid task_success")
    result["token_usage"] = _token_usage(response)
    return result


def _cost_usd(token_usage: dict) -> float | None:
    """Calculate cost only when the caller supplies explicit model prices."""
    input_rate = os.getenv("OPENAI_INPUT_COST_PER_1M_TOKENS")
    output_rate = os.getenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS")
    if input_rate is None or output_rate is None or not token_usage.get("usage_available"):
        return None
    return (
        token_usage["input_tokens"] * float(input_rate)
        + token_usage["output_tokens"] * float(output_rate)
    ) / 1_000_000


def evaluate_dataset(dataset_path: str, output_path: str, k: int = 3) -> dict:
    """Evaluate retrieval, answer quality, latency, token usage, and configured cost."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        cases = json.load(file)

    required = {"query_id", "question", "relevant_ticket_ids", "reference_answer"}
    results = []
    for case in cases:
        missing = required.difference(case)
        if missing:
            raise ValueError(f"{case.get('query_id', 'Unknown case')} missing fields: {sorted(missing)}")

        retrieved_docs = tool_manager.vectorstore.similarity_search(case["question"], k=k)
        retrieved_ids = [doc.metadata["ticket_id"] for doc in retrieved_docs]
        retrieval = _retrieval_metrics(retrieved_ids, case["relevant_ticket_ids"])
        agent_run = run_agent_with_trace(case["question"])
        evidence = "\n\n".join(step["output"] for step in agent_run["tool_trace"])

        judge_error = None
        try:
            judge = _judge_answer(
                case["question"],
                agent_run["answer"],
                case["reference_answer"],
                evidence,
            )
        except Exception as error:
            judge = {
                "groundedness": None,
                "completeness": None,
                "task_success": None,
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "usage_available": False,
                },
            }
            judge_error = str(error)

        combined_usage = {
            key: agent_run["token_usage"].get(key, 0) + judge["token_usage"].get(key, 0)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        combined_usage["usage_available"] = (
            agent_run["token_usage"].get("usage_available", False)
            and judge["token_usage"].get("usage_available", False)
        )
        results.append({
            "query_id": case["query_id"],
            "question": case["question"],
            "relevant_ticket_ids": case["relevant_ticket_ids"],
            "retrieved_ticket_ids": retrieved_ids,
            **retrieval,
            "groundedness": judge.get("groundedness"),
            "completeness": judge.get("completeness"),
            "task_success": judge.get("task_success"),
            "judge_reasons": {
                "groundedness": judge.get("groundedness_reason"),
                "completeness": judge.get("completeness_reason"),
                "task_success": judge.get("task_success_reason"),
            },
            "judge_error": judge_error,
            "latency_seconds": agent_run["latency_seconds"],
            "token_usage_including_judge": combined_usage,
            "cost_usd_including_judge": _cost_usd(combined_usage),
            "answer": agent_run["answer"],
        })

    def average(field: str):
        values = [row[field] for row in results if row[field] is not None]
        return mean(values) if values else None

    successful = [row["task_success"] for row in results if row["task_success"] is not None]
    summary = {
        "number_of_queries": len(results),
        "k": k,
        "retrieval_quality": {
            "mean_reciprocal_rank": average("reciprocal_rank"),
            "hit_rate_at_k": average("hit_at_k"),
        },
        "mean_precision_at_k": average("precision_at_k"),
        "mean_recall_at_k": average("recall_at_k"),
        "mean_f1_at_k": average("f1_at_k"),
        "mean_groundedness": average("groundedness"),
        "mean_completeness": average("completeness"),
        "task_success_rate": mean(successful) if successful else None,
        "mean_latency_seconds": average("latency_seconds"),
        "mean_cost_usd_including_judge": average("cost_usd_including_judge"),
        "cost_note": (
            "Calculated from configured per-million-token rates."
            if all(row["cost_usd_including_judge"] is not None for row in results)
            else "Unavailable: token usage and both OPENAI_INPUT_COST_PER_1M_TOKENS "
                 "and OPENAI_OUTPUT_COST_PER_1M_TOKENS are required; prices are not guessed."
        ),
        "judge_failures": sum(row["judge_error"] is not None for row in results),
    }
    report = {"summary": summary, "results": results}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return report


def run_demo() -> None:
    examples = [
        "How do I fix authentication problems after password reset?",
        "Show me details of ticket TICK-005",
        "What payment-related issues have we seen?",
        "Give me an overview of the ticket database",
        "Find database-related critical issues and tell me how they were resolved",
    ]
    for query in examples:
        print(f"\nUser: {query}\n")
        print(f"Assistant: {run_agent(query)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the support agent or its evaluation suite.")
    parser.add_argument("--evaluate", help="Path to evaluation_queries.json")
    parser.add_argument("--output", default="evaluation_results.json", help="Evaluation report path")
    parser.add_argument("--k", type=int, default=3, help="Number of semantic results to evaluate")
    args = parser.parse_args()
    if args.evaluate:
        evaluation_report = evaluate_dataset(args.evaluate, args.output, k=args.k)
        print(json.dumps(evaluation_report["summary"], indent=2))
    else:
        run_demo()

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
