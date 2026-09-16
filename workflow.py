"""
workflow.py - LangGraph Async Parallel Multi-Agent Workflow

Orchestrates brain dump processing via an async LangGraph StateGraph pipeline.
Runs QuestionAgent, PerspectiveAgent, SearchAgent/FeedAgent concurrently
to minimize overall latency and enforce clean state management.
"""

import asyncio
from typing import TypedDict, List, Dict, Optional, Any
import os
import concurrent.futures
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END

# Import agent components & DB layer
from agents import QuestionAgent, PerspectiveAgent, SearchAgent, FeedAgent, GenerationAgent
from braindump_core import BrainDumpDB


# ============== 1. STATE REDUCER & DEFINITION ==============

def overwrite_reducer(a: Any, b: Any) -> Any:
    """Reducer that accepts the latest non-None value."""
    return b if b is not None else a

def merge_dict_reducer(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that merges dictionary updates from parallel nodes."""
    res = a.copy() if a else {}
    if b:
        res.update(b)
    return res


class BrainDumpState(TypedDict, total=False):
    # Core Identifiers
    dump_id: Annotated[Optional[str], overwrite_reducer]
    text: Annotated[Optional[str], overwrite_reducer]
    cluster_id: Annotated[Optional[int], overwrite_reducer]
    cluster_label: Annotated[Optional[str], overwrite_reducer]
    cluster_dumps: Annotated[List[str], overwrite_reducer]
    
    # Parallel Node Outputs
    socratic_questions: Annotated[List[str], overwrite_reducer]
    perspectives: Annotated[Dict[str, Any], merge_dict_reducer]
    summary: Annotated[Optional[str], overwrite_reducer]
    sources: Annotated[List[Dict[str, str]], overwrite_reducer]
    image_urls: Annotated[List[str], overwrite_reducer]
    
    # Final Synthetic Output
    synthetic_dump: Annotated[Optional[str], overwrite_reducer]
    
    # Execution Metadata & Error Tracking
    status: Annotated[Optional[str], overwrite_reducer]
    error: Annotated[Optional[str], overwrite_reducer]


# ============== 2. SHARED AGENT INSTANCES ==============
question_agent = QuestionAgent()
perspective_agent = PerspectiveAgent()
search_agent = SearchAgent()
feed_agent = FeedAgent(search_agent=search_agent)
generation_agent = GenerationAgent()


# ============== 3. ASYNC NODE FUNCTIONS ==============

async def entry_node(state: BrainDumpState) -> Dict[str, Any]:
    """Entry Node: Validates input state and initiates parallel processing branches."""
    text = state.get("text", "") or ""
    print(f"🚀 [LangGraph Pipeline] Starting processing for: '{text[:50]}...'")
    return {"status": "processing"}


async def question_node(state: BrainDumpState) -> Dict[str, Any]:
    """Node: Generates Socratic questions in parallel."""
    text = state.get("text", "") or ""
    print(f"❓ [LangGraph] Starting question_node...")
    try:
        loop = asyncio.get_running_loop()
        questions = await loop.run_in_executor(None, question_agent.generate_questions, text)
        print(f"✓ [LangGraph] question_node complete ({len(questions)} questions)")
        return {"socratic_questions": questions}
    except Exception as e:
        print(f"❌ [LangGraph] Error in question_node: {e}")
        return {"socratic_questions": [f"Error generating questions: {e}"]}


async def perspective_node(state: BrainDumpState) -> Dict[str, Any]:
    """Node: Generates Skeptical, Optimistic, and Nuanced viewpoints in parallel."""
    text = state.get("text", "") or ""
    print(f"👁️ [LangGraph] Starting perspective_node...")
    try:
        loop = asyncio.get_running_loop()
        perspectives = await loop.run_in_executor(None, perspective_agent.analyze, text)
        print("✓ [LangGraph] perspective_node complete")
        return {"perspectives": perspectives}
    except Exception as e:
        print(f"❌ [LangGraph] Error in perspective_node: {e}")
        return {"perspectives": {"skeptical": f"Error: {e}", "optimistic": "", "nuanced": ""}}


async def search_node(state: BrainDumpState) -> Dict[str, Any]:
    """Node: Searches web and generates summary synthesis in parallel."""
    text = state.get("text", "") or ""
    print(f"🔍 [LangGraph] Starting search_node...")
    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(None, feed_agent.generate_summary, text)
        summary = res.get("summary", "No summary generated")
        sources = res.get("sources", [])
        print("✓ [LangGraph] search_node complete")
        return {"summary": summary, "sources": sources}
    except Exception as e:
        print(f"❌ [LangGraph] Error in search_node: {e}")
        return {"summary": f"Search error: {e}", "sources": []}


async def image_node(state: BrainDumpState) -> Dict[str, Any]:
    """Node: Searches for related visual images after search synthesis."""
    text = state.get("text", "") or ""
    print(f"📸 [LangGraph] Starting image_node...")
    try:
        loop = asyncio.get_running_loop()
        image_urls = await loop.run_in_executor(None, feed_agent.search_images, text, 3)
        print(f"✓ [LangGraph] image_node complete ({len(image_urls)} images)")
        return {"image_urls": image_urls}
    except Exception as e:
        print(f"❌ [LangGraph] Error in image_node: {e}")
        return {"image_urls": []}


async def consolidator_node(state: BrainDumpState) -> Dict[str, Any]:
    """
    Consolidation Node (Fan-In):
    Collects outputs from parallel branches, updates Neo4j cache if dump_id is present.
    """
    dump_id = state.get("dump_id")
    summary = state.get("summary", "")
    questions = state.get("socratic_questions", [])
    image_urls = state.get("image_urls", [])
    
    print(f"💾 [LangGraph] Consolidator node running for dump_id={dump_id}")
    
    if dump_id:
        try:
            db = BrainDumpDB()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                db.save_feed_cache, 
                dump_id, 
                summary, 
                questions, 
                image_urls
            )
            db.close()
            print(f"✓ [LangGraph] Saved feed cache to Neo4j for dump_id={dump_id}")
        except Exception as e:
            print(f"⚠️ [LangGraph] Consolidator cache save warning: {e}")
            
    return {"status": "success"}


async def generation_node(state: BrainDumpState) -> Dict[str, Any]:
    """Node: Generates a creative synthetic thought for the cluster if applicable."""
    cluster_label = state.get("cluster_label")
    cluster_dumps = state.get("cluster_dumps", [])
    
    if not cluster_label or not cluster_dumps:
        return {"synthetic_dump": None}
    
    print(f"🧬 [LangGraph] Starting generation_node for cluster: '{cluster_label}'")
    try:
        loop = asyncio.get_running_loop()
        synthetic_dump = await loop.run_in_executor(
            None, 
            generation_agent.generate_braindump, 
            cluster_label, 
            cluster_dumps
        )
        print(f"✓ [LangGraph] generation_node complete")
        return {"synthetic_dump": synthetic_dump}
    except Exception as e:
        print(f"❌ [LangGraph] Error in generation_node: {e}")
        return {"synthetic_dump": None}


# ============== 4. BUILD LANGGRAPH STATEGRAPH ==============

def build_braindump_graph():
    """Builds and compiles the async parallel LangGraph StateGraph workflow."""
    builder = StateGraph(BrainDumpState)
    
    # Add Nodes
    builder.add_node("entry_node", entry_node)
    builder.add_node("question_node", question_node)
    builder.add_node("perspective_node", perspective_node)
    builder.add_node("search_node", search_node)
    builder.add_node("image_node", image_node)
    builder.add_node("consolidator_node", consolidator_node)
    builder.add_node("generation_node", generation_node)
    
    # Pipeline Entry Point
    builder.add_edge(START, "entry_node")
    
    # Parallel Fan-Out from Entry Node
    builder.add_edge("entry_node", "question_node")
    builder.add_edge("entry_node", "perspective_node")
    builder.add_edge("entry_node", "search_node")
    
    # Sequential Sub-branch: search_node -> image_node
    builder.add_edge("search_node", "image_node")
    
    # Fan-In Join: Parallel branches converge into consolidator_node
    builder.add_edge("question_node", "consolidator_node")
    builder.add_edge("perspective_node", "consolidator_node")
    builder.add_edge("image_node", "consolidator_node")
    
    # Final step: Synthetic thought generation for cluster
    builder.add_edge("consolidator_node", "generation_node")
    builder.add_edge("generation_node", END)
    
    return builder.compile()


# Compiled Graph Singleton
braindump_graph = build_braindump_graph()


# ============== 5. CONVENIENCE RUNNER FUNCTION ==============

def run_braindump_workflow_sync(
    dump_id: str,
    text: str,
    cluster_id: Optional[int] = None,
    cluster_label: Optional[str] = None,
    cluster_dumps: Optional[List[str]] = None
) -> BrainDumpState:
    """
    Synchronous entrypoint wrapper to execute the async LangGraph workflow.
    Can be called directly from Streamlit (app.py) or CLI.
    """
    initial_state: BrainDumpState = {
        "dump_id": dump_id,
        "text": text,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "cluster_dumps": cluster_dumps or [],
        "socratic_questions": [],
        "perspectives": {},
        "summary": "",
        "sources": [],
        "image_urls": [],
        "synthetic_dump": None,
        "status": "pending",
        "error": None
    }
    
    async def _execute():
        return await braindump_graph.ainvoke(initial_state)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(_execute()))
            return future.result()
    except Exception as e:
        print(f"❌ [LangGraph Workflow Failure]: {e}")
        initial_state["error"] = str(e)
        initial_state["status"] = "failed"
        return initial_state


if __name__ == "__main__":
    print("=== Testing LangGraph Async Parallel Workflow Execution ===")
    test_res = run_braindump_workflow_sync(
        dump_id="test-uuid-123",
        text="Why do dreams feel so real but fade so quickly?",
        cluster_id=1,
        cluster_label="Dreams & Consciousness",
        cluster_dumps=["Why do we forget dreams?", "Is consciousness emergent?"]
    )
    print("\n=== Final LangGraph State Output ===")
    print("Questions:", test_res.get("socratic_questions"))
    print("Perspectives:", test_res.get("perspectives"))
    print("Summary:", test_res.get("summary")[:100], "...")
    print("Images:", test_res.get("image_urls"))
    print("Status:", test_res.get("status"))
