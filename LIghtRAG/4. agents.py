# agents.py
from pydantic_ai import Agent, RunContext
from lightrag import QueryParam
from metrics import update_retrieval_metrics, update_reasoning_metrics, detect_pii, metrics

rag = None  # Global rag

retrieval_agent = Agent(
    "openai:gpt-4o",
    deps_type=str,
    output_type=list,
    system_prompt="Retrieve legal/HR document summaries."
)

@retrieval_agent.tool
async def light_rag_retrieve(ctx: RunContext[str], query_text: str) -> list:
    results = await rag.query(query_text, param=QueryParam(mode="hybrid"))
    update_retrieval_metrics(results["docs"], query_text)
    return [doc["text"] for doc in results["docs"]]

reasoner_agent = Agent(
    "openai:gpt-4o",
    deps_type=list,
    output_type=str,
    system_prompt="Answer accurately only using provided context."
)

@reasoner_agent.tool
async def generate_answer(ctx: RunContext[list], summaries: list) -> str:
    joined = "\n".join(summaries)
    return await ctx.llm.apredict(f"Context:\n{joined}\n\nAnswer without assumptions.")

async def guardrails_reflection(state):
    if detect_pii(state.draft_answer):
        metrics["pii_leakage_detected"] += 1
        return {"final_answer": "⚠️ PII detected, suppressed.", "hallucination_flag": True}

    metrics["successful_tasks"] += 1
    return {"final_answer": state.draft_answer, "hallucination_flag": False}

# Retry if hallucination detected
async def retry_reasoner(state):
    if state.hallucination_flag:
        metrics["retries"] += 1
        summaries = state.retrieved_contexts
        prompt = f"STRICT MODE: Only answer if verifiable from context.\nContext:\n{summaries}"
        from embedder import clip_processor, clip_model
        return {"draft_answer": await ctx.llm.apredict(prompt)}
    return {"draft_answer": state.final_answer}
