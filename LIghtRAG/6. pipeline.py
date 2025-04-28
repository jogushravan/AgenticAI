# pipeline.py
from langgraph.graph import StateGraph
from pydantic import BaseModel
from agents import retrieval_agent, reasoner_agent, guardrails_reflection, retry_reasoner

class LightRAGState(BaseModel):
    query_text: str
    retrieved_contexts: list = []
    draft_answer: str = ""
    final_answer: str = ""
    hallucination_flag: bool = False

async def run_pipeline(user_query):
    graph = StateGraph()

    graph.add_node("Retriever", retrieval_agent)
    graph.add_node("Reasoner", reasoner_agent)
    graph.add_node("Guardrails", guardrails_reflection)
    graph.add_node("Retry", retry_reasoner)

    graph.set_entry_point("Retriever")
    graph.add_edge("Retriever", "Reasoner")
    graph.add_edge("Reasoner", "Guardrails")
    graph.add_edge("Guardrails", "Retry")

    initial_state = LightRAGState(query_text=user_query)

    output = await graph.run(initial_state.dict())
    return output["final_answer"]
