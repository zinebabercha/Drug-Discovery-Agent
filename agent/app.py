import streamlit as st
import json
from state import AgentState
from graph import app
from tool_agent import sanitize_dict

st.set_page_config(page_title="Drug Discovery Agent", page_icon="🧬", layout="wide")

st.title("🧬 Drug Discovery QSAR Agent")
st.markdown("Analyze molecules for target inhibition with QSAR, ADMET, and literature insights")

# Sidebar for input
with st.sidebar:
    st.header("Input Parameters")
    
    smiles = st.text_input(
        "SMILES String",
        value="COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O",
        help="Enter the molecular SMILES notation"
    )
    
    target = st.selectbox(
        "Target Protein",
        ["MAOB", "tubulin", "AChE", "FAK", "GlyT1", "Ecoli", "Breast_cancer_ANN", "Scaffold_based_discovery"],
        help="Select the biological target"
    )
    
    analyze_btn = st.button("🔬 Analyze Molecule", type="primary", use_container_width=True)

# Main content area
if analyze_btn:
    if not smiles:
        st.error("⚠️ Please enter a SMILES string")
    else:
        with st.spinner("🔄 Running analysis... This may take 30-60 seconds"):
            # Create query
            user_query = f"Analyze this molecule for {target} inhibition. SMILES: {smiles} TARGET: {target}"
            
            # Initialize state
            state = AgentState(
                user_query=user_query,
                canonical_smiles="",
                qsar_target="",
                task_plan=[],
                qsar_results={},
                admet_results={},
                docking_results={},
                md_results={},
                rag_chunks=[],
                final_report="",
                tool_output="",
                strategist_scratchpad=[],
                llm_call_history=[]
            )
            
            try:
                # Run workflow
                final_state = app.invoke(state)
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Report", "🧪 QSAR Results", "💊 ADMET Profile", "🔍 Debug Info"])
                
                with tab1:
                    st.markdown("### Analysis Report")
                    report = final_state["final_report"]
                    if hasattr(report, 'content'):
                        st.markdown(report.content)
                    else:
                        st.markdown(report)
                
                with tab2:
                    st.markdown("### QSAR Predictions")
                    qsar = final_state.get("qsar_results", {})
                    if qsar:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("pIC50", f"{qsar.get('pIC50', 'N/A'):.2f}")
                        with col2:
                            st.metric("IC50 (nM)", f"{qsar.get('IC50_nM', 'N/A'):.0f}")
                        with col3:
                            activity = qsar.get('activity_level', 'N/A')
                            color = "🟢" if activity == "High" else "🟡" if activity == "Medium" else "🔴"
                            st.metric("Activity", f"{color} {activity}")
                        
                        st.json(sanitize_dict(qsar))
                    else:
                        st.warning("No QSAR results available")
                
                with tab3:
                    st.markdown("### ADMET Properties")
                    admet = final_state.get("admet_results", {})
                    if admet:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MW", admet.get('molecular_weight', 'N/A'))
                        with col2:
                            st.metric("LogP", admet.get('logP', 'N/A'))
                        with col3:
                            st.metric("TPSA", admet.get('tpsa', 'N/A'))
                        with col4:
                            lipinski = admet.get('lipinski_compliant', False)
                            st.metric("Lipinski", "✅ Pass" if lipinski else "❌ Fail")
                        
                        st.json(sanitize_dict(admet))
                    else:
                        st.warning("No ADMET results available")
                
                with tab4:
                    st.markdown("### Debug Information")
                    with st.expander("Strategist Scratchpad"):
                        for item in final_state["strategist_scratchpad"]:
                            st.text(item)
                    
                    with st.expander("RAG Chunks"):
                        st.text(f"Retrieved {len(final_state['rag_chunks'])} chunks")
                    
                    with st.expander("Structured Summary (JSON)"):
                        structured = sanitize_dict(final_state.get("structured_summary", {}))
                        st.json(structured)
                
                # Success message
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)

else:
    # Show example
    st.info("👈 Enter a SMILES string and select a target, then click 'Analyze Molecule'")
    
    with st.expander("ℹ️ Example SMILES"):
        st.code("COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O")
        st.caption("This is a complex organic molecule with potential MAOB inhibitory activity")



# pip install streamlit

# streamlit run app.py
