




import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from state import AgentState
from tool_agent import canonicalize_smiles, run_qsar_prediction, retrieve_rag_chunks, run_admet_prediction, sanitize_dict, qsar_agents
from llm import strategist_llm, synthesizer_llm, synthesizer_fallback_llm

# --- Strategist Node ---
def strategist_node(state: AgentState) -> AgentState:
    print("---STRATEGIST NODE---")
    user_query = state["user_query"]

    # 1. Extract SMILES using regex
    match = re.search(r'SMILES:\s*([^\s]+)', user_query)
    current_smiles = match.group(1) if match else ""
    canonical_smiles = ""
    if current_smiles:
        canonical_smiles = canonicalize_smiles(current_smiles)
        print(f"Canonicalized SMILES: {canonical_smiles}")
        if not canonical_smiles:
            state["strategist_scratchpad"].append("Error: Invalid SMILES provided.")
            state["final_report"] = "I received an invalid SMILES string. Please provide a valid one to proceed."
            return state  # Early exit

    state["canonical_smiles"] = canonical_smiles

    # 2. Extract and validate TARGET - THIS IS THE KEY FIX
    target_match = re.search(r'TARGET:\s*([^\s,]+)', user_query, re.IGNORECASE)
    target = target_match.group(1) if target_match else "tubulin"
    
    # Normalize target name (handle case variations)
    target = target.strip()
    
    # Check if target exists in qsar_agents
    if target not in qsar_agents:
        available_targets = ", ".join(qsar_agents.keys())
        state["strategist_scratchpad"].append(
            f"Warning: Unknown target '{target}', available targets: {available_targets}. Defaulting to tubulin."
        )
        target = "tubulin"
    
    state["qsar_target"] = target
    print(f"QSAR Target selected: {target}")

    # 3. Plan tasks
    strategist_prompt = PromptTemplate(
        template="""
You are an expert drug discovery agent. Return your plan strictly in JSON.

Example JSON:
{{
 "plan": ["run_qsar_prediction","retrieve_rag_chunks", "run_admet_prediction"],
 "rag_query": "general {qsar_target} inhibitor science"
}}

User Query: {user_query}
Canonical SMILES: {canonical_smiles}
QSAR Target: {qsar_target}
""",
        input_variables=["user_query", "canonical_smiles", "qsar_target"]
    )

    strategist_chain = strategist_prompt | strategist_llm | JsonOutputParser()

    try:
        planning_output = strategist_chain.invoke({
            "user_query": user_query, 
            "canonical_smiles": canonical_smiles,
            "qsar_target": target
        })
        
        state["task_plan"] = planning_output.get("plan", [])
        state["strategist_scratchpad"].append(f"Strategist Plan: {state['task_plan']}")

        # Generate RAG query with proper target
        final_rag_query = planning_output.get("rag_query") or f"{target} inhibitor activity and structure-activity relationship of {canonical_smiles}"
        state["strategist_scratchpad"].append(f"Final RAG Query: {final_rag_query}")

        # Run RAG
        rag_results = retrieve_rag_chunks(final_rag_query)
        state["rag_chunks"].extend(rag_results)
        state["strategist_scratchpad"].append(f"RAG retrieved {len(rag_results)} chunks.")

        state["llm_call_history"].append({
            "node": "strategist",
            "input": {"user_query": user_query, "canonical_smiles": canonical_smiles},
            "output": planning_output
        })

    except Exception as e:
        print(f"Strategist failed: {e}")
        state["strategist_scratchpad"].append(f"Strategist planning failed: {e}")
        state["final_report"] = f"An error occurred during planning: {e}"

    return state

# --- Tool Executor Node ---
def tool_executor_node(state: AgentState) -> AgentState:
    print("---TOOL EXECUTOR NODE---")
    tool_output = []

    for tool_name in state.get("task_plan", []):
        if tool_name == "run_qsar_prediction":
            # FIX: Use the target from state, with proper fallback
            target = state.get("qsar_target", "tubulin")
            print(f"[Tool Executor] Running QSAR for target: {target}")  # Debug print
            
            qsar_result = run_qsar_prediction(state, target=target)
            state["qsar_results"] = qsar_result
            state["strategist_scratchpad"].append(f"Executed QSAR ({target}), result: {qsar_result}")

        elif tool_name == "run_admet_prediction":
            admet_result = run_admet_prediction(state)
            state["admet_results"] = admet_result
            tool_output.append(f"ADMET Prediction: {admet_result}")
            state["strategist_scratchpad"].append(
                f"Executed ADMET, result keys: {list(admet_result.keys()) if isinstance(admet_result, dict) else 'err'}"
            )
            
        elif tool_name == "retrieve_rag_chunks":
            pass  # Already done in Strategist
            
        else:
            tool_output.append(f"Unknown tool: {tool_name}")
            state["strategist_scratchpad"].append(f"Unknown tool requested: {tool_name}")

    state["tool_output"] = "\n".join(tool_output)
    state["task_plan"] = []  # Clear plan
    return state

# --- Synthesizer Node ---
def synthesizer_node(state: AgentState) -> AgentState:
    print("---SYNTHESIZER NODE---")
    qsar_results = state.get("qsar_results", {})
    admet_results = state.get("admet_results", {})
    rag_chunks = state.get("rag_chunks", [])
    user_query = state.get("user_query")
    canonical_smiles = state.get("canonical_smiles")
    qsar_target = state.get("qsar_target", "unknown")
    
    rag_content = "\n\n".join([chunk["page_content"] for chunk in rag_chunks])

    print("QSAR results:", qsar_results)
    print("ADMET results:", admet_results)
    print("RAG chunks count:", len(rag_chunks))

    state["structured_summary"] = {
        "smiles": canonical_smiles,
        "target": qsar_target,
        "qsar": sanitize_dict(qsar_results),
        "admet": sanitize_dict(admet_results),
        "rag_snippets_count": len(rag_chunks),
    }

    synthesizer_prompt_template = """
You are an expert drug discovery consultant. Summarize findings in short sections and give actionable next steps.

User Query: {user_query}
Target: {qsar_target}
Molecule (SMILES): {canonical_smiles}
QSAR Prediction Results: {qsar_results}
ADMET Prediction Results: {admet_results}
Relevant Scientific Literature (RAG Chunks): {rag_content}

Provide a concise analysis focusing on:
1. Activity prediction for the specified target
2. Drug-likeness and ADMET properties
3. Key insights from literature
4. Actionable next steps for optimization
"""

    synthesizer_prompt = PromptTemplate(
        template=synthesizer_prompt_template,
        input_variables=["user_query", "qsar_target", "canonical_smiles", "qsar_results", "admet_results", "rag_content"]
    )

    try:
        report_chain = synthesizer_prompt | synthesizer_llm
        report_text = report_chain.invoke({
            "user_query": user_query,
            "qsar_target": qsar_target,
            "canonical_smiles": canonical_smiles,
            "qsar_results": str(qsar_results),
            "admet_results": str(admet_results),
            "rag_content": rag_content or "No relevant scientific literature found."
        })

    except Exception as e:
        print(f"Synthesizer primary LLM failed: {e}, using fallback.")
        state["strategist_scratchpad"].append(f"Synthesizer primary LLM failed: {e}")
        
        report_chain = synthesizer_prompt | synthesizer_fallback_llm
        report_text = report_chain.invoke({
            "user_query": user_query,
            "qsar_target": qsar_target,
            "canonical_smiles": canonical_smiles,
            "qsar_results": str(qsar_results),
            "admet_results": str(admet_results),
            "rag_content": rag_content or "No relevant scientific literature found."
        })

    state["final_report"] = report_text
    state["llm_call_history"].append({
        "node": "synthesizer",
        "input": {
            "qsar_results": str(qsar_results),
            "admet_results": str(admet_results),
            "rag_content": rag_content
        },
        "output": report_text
    })
    return state