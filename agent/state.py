from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Represents the state of our drug discovery agent.
    """
    user_query: str
    canonical_smiles: str
    qsar_target: str  # ADD THIS FIELD
    task_plan: List[str]
    qsar_results: Dict[str, Any]
    admet_results: Dict[str, Any]
    docking_results: Dict[str, Any]
    md_results: Dict[str, Any]
    rag_chunks: List[Dict[str, Any]]
    final_report: str
    tool_output: str
    strategist_scratchpad: List[str]
    llm_call_history: List[Dict[str, Any]]