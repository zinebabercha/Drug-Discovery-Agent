


from state import AgentState
from graph import app  # compiled workflow
from tool_agent import sanitize_dict
import json

def main():
    # user_query = "Analyze this molecule for tubulin inhibition. SMILES: COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O TARGET: tubulin"

    user_query = "Analyze this molecule for MAOB inhibition. SMILES: COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O TARGET: MAOB"

# or TARGET: AChE, TARGET: Ecoli, etc.

    
    state = AgentState(
        user_query=user_query,
        canonical_smiles="",
        qsar_target="",  # ADD THIS FIELD
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
        final_state = app.invoke(state) 

        print("\n=== FINAL REPORT ===")
        print(final_state["final_report"])

        print("\n=== STRUCTURED SUMMARY (JSON) ===")
        structured = final_state.get("structured_summary", {})

        # make sure everything is serialisable
        structured = sanitize_dict(structured)
        print(json.dumps(structured, indent=2))



        # optional: write to disk for later analysis
        # with open("agent_results.json", "w", encoding="utf-8") as fh:
        #     json.dump(structured, fh, indent=2)

        print("\n=== DEBUG INFO ===")
        print("Strategist Scratchpad:", final_state["strategist_scratchpad"])
        print("Tool Output:", final_state["tool_output"])
        print("RAG Chunks Retrieved:", len(final_state["rag_chunks"]))
        print("LLM Call History:", final_state["llm_call_history"])
        print(f"Target Used: {final_state.get('qsar_target', 'Not set')}")


    except Exception as e:
        print("Error during workflow execution:", e)

if __name__ == "__main__":
    main()









# (drug) C:\Users\DELL\Downloads\drug\agent>python main.py
# Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
# It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe
# Model loaded from: ..\qsar_models\tubulin_qsar_model.pkl (feature_names restored: No)
# QSAR model 'tubulin' loaded successfully from ..\qsar_models\tubulin_qsar_model.pkl
# Model loaded from: ..\qsar_models\AChE_qsar_model.pkl (feature_names restored: No)
# QSAR model 'AChE' loaded successfully from ..\qsar_models\AChE_qsar_model.pkl
# Model loaded from: ..\qsar_models\Breast_cancer_ANN_qsar_model.pkl (feature_names restored: No)
# QSAR model 'Breast_cancer_ANN' loaded successfully from ..\qsar_models\Breast_cancer_ANN_qsar_model.pkl
# Model loaded from: ..\qsar_models\Ecoli_qsar_model.pkl (feature_names restored: No)
# QSAR model 'Ecoli' loaded successfully from ..\qsar_models\Ecoli_qsar_model.pkl
# Model loaded from: ..\qsar_models\FAK_qsar_model.pkl (feature_names restored: No)
# QSAR model 'FAK' loaded successfully from ..\qsar_models\FAK_qsar_model.pkl
# Model loaded from: ..\qsar_models\GlyT1_qsar_model.pkl (feature_names restored: No)
# QSAR model 'GlyT1' loaded successfully from ..\qsar_models\GlyT1_qsar_model.pkl
# Model loaded from: ..\qsar_models\MAOB_qsar_model.pkl (feature_names restored: No)
# QSAR model 'MAOB' loaded successfully from ..\qsar_models\MAOB_qsar_model.pkl
# Model loaded from: ..\qsar_models\Scaffold_based_discovery_qsar_model.pkl (feature_names restored: No)
# QSAR model 'Scaffold_based_discovery' loaded successfully from ..\qsar_models\Scaffold_based_discovery_qsar_model.pkl
# LangGraph workflow compiled!
# ---STRATEGIST NODE---
# Canonicalized SMILES: COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O
# QSAR Target selected: MAOB
# Retrieving 5 chunks for query: 'MAOB inhibitor activity and structure-activity relationship of COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O'
# ---TOOL EXECUTOR NODE---
# [Tool Executor] Running QSAR for target: MAOB
# Running QSAR prediction for target 'MAOB': COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O
# ---SYNTHESIZER NODE---
# QSAR results: {'pIC50': np.float64(-4.138062866206541), 'IC50_nM': np.float64(13742.408885666036), 'activity_level': 'Low', 'uncertainty': np.float64(0.8705036195325653), 'model_path': '..\\qsar_models\\MAOB_qsar_model.pkl', 'model_version': 'v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'target': 'MAOB'}
# ADMET results: {'tool': 'admet_rdkit_v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'molecular_weight': '384.38', 'logP': '2.62', 'h_bond_donors': '1', 'h_bond_acceptors': '7', 'tpsa': '83.45', 'rotatable_bonds': '3', 'heavy_atom_count': '28', 'aromatic_proportion': '0.429', 'lipinski_compliant': 'True', 'drug_likeness': 'Good', 'bioavailability_score': 'Moderate'}
# RAG chunks count: 5

# === FINAL REPORT ===
# content='### 1. **Activity Prediction for MAOB Inhibition**  \n- **QSAR Results**: Predicted **pIC50 = -4.14 (IC50 = 13,742 nM)**, indicating **low activity** against MAOB.  \n- **Uncertainty**: High model uncertainty (**±0.87**), suggesting limited confidence in the prediction.  \n- **Comparison to Literature**: Compound 31 (from the cited study) achieved superior activity (**lower IC50**), highlighting the need for structural improvements.  \n\n---\n\n### 2. **Drug-Likeness and ADMET Properties**  \n- **Drug-Like Metrics**:  \n  - **Molecular Weight**: 384.38 g/mol (within acceptable range).  \n  - **LogP**: 2.62 (moderate lipophilicity, favorable for CNS penetration).  \n  - **H-Bond Donors/Acceptors**: 1/7 (moderate hydrogen bonding potential).  \n  - **TPSA**: 83.45 Å² (high polar surface area may reduce membrane permeability).  \n- **Compliance**: Passes **Lipinski’s Rule of Five**, with **"Good" drug-likeness**.  \n- **Bioavailability**: **Moderate**—may require optimization for higher CNS penetration.  \n\n---\n\n### 3. **Key Insights from Literature**  \n- **Reference Compounds**:  \n  - **Compound 31/31.j3** demonstrated strong MAOB inhibition via **3D-QSAR and molecular docking**, with stable binding to key residues (e.g., non-covalent interactions).  \n  - **6-Hydroxybenzothiazole-2-carboxamide derivatives** are highlighted as promising scaffolds due to **selectivity**, **low side effects**, and **favorable pharmacokinetics**.  \n- **Design Guidance**: Structural features like **amide substituents**, **aromatic rings**, and **hydroxyl groups** enhance activity and binding stability.  \n\n---\n\n### 4. **Actionable Next Steps for Optimization**  \n1. **Structural Modifications**:  \n   - **Incorporate amide/benzothiazole moieties** (as in 6-hydroxybenzothiazole derivatives) to improve binding affinity.  \n   - **Add hydrogen bond donors** (e.g., -NH2, -OH) to mimic interactions seen in compound 31.j3.  \n   - **Simplify/adjust ester groups** (e.g., replace with amides or ketones) to reduce polar surface area and enhance stability.  \n\n2. **Computational Validation**:  \n   - Perform **molecular docking** (using PDB ID: 3PO7) to identify critical binding interactions (e.g., with residues F356, Y407).  \n   - Use **3D-QSAR/COMSIA** to guide structure-activity relationship (SAR) analysis.  \n\n3. **Experimental Testing**:  \n   - Validate **predicted IC50** via in vitro MAOB inhibition assays.  \n   - Assess **selectivity** against MAO-A to minimize off-target effects.  \n\n4. **ADMET Optimization**:  \n   - Reduce **TPSA** by introducing hydrophobic moieties (e.g., aryl groups) to improve CNS penetration.  \n   - Evaluate **metabolic stability** and **blood-brain barrier (BBB) permeability**.  \n\n5. **Iterative Design**:  \n   - Leverage insights from **compound 31.j3’s success** (e.g., stable non-covalent interactions) to refine the scaffold for higher potency.  \n\n--- \n\n**Summary**: The molecule shows **low predicted MAOB inhibition** but has a **drug-like profile**. Optimization via structural modifications inspired by literature (e.g., benzothiazole scaffolds, amide linkers) and computational/experimental validation is critical to enhance activity and selectivity.' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 1636, 'prompt_tokens': 1671, 'total_tokens': 3307, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'qwen/qwen3-32b', 'system_fingerprint': None, 'id': 'gen-1760835168-8eIGo1OMtWgJluSjgOPr', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--591657a5-e731-40f4-be7e-424f0a2def64-0' usage_metadata={'input_tokens': 1671, 'output_tokens': 1636, 'total_tokens': 3307, 'input_token_details': {}, 'output_token_details': {}}

# === STRUCTURED SUMMARY (JSON) ===
# {}

# === DEBUG INFO ===
# Strategist Scratchpad: ["Strategist Plan: ['run_qsar_prediction', 'retrieve_rag_chunks', 'run_admet_prediction']", 'Final RAG Query: MAOB inhibitor activity and structure-activity relationship of COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'RAG retrieved 5 chunks.', "Executed QSAR (MAOB), result: {'pIC50': np.float64(-4.138062866206541), 'IC50_nM': np.float64(13742.408885666036), 'activity_level': 'Low', 'uncertainty': np.float64(0.8705036195325653), 'model_path': '..\\\\qsar_models\\\\MAOB_qsar_model.pkl', 'model_version': 'v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'target': 'MAOB'}", "Executed ADMET, result keys: ['tool', 'input_smiles', 'molecular_weight', 'logP', 'h_bond_donors', 'h_bond_acceptors', 'tpsa', 'rotatable_bonds', 'heavy_atom_count', 'aromatic_proportion', 'lipinski_compliant', 'drug_likeness', 'bioavailability_score']"]
# Tool Output: ADMET Prediction: {'tool': 'admet_rdkit_v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'molecular_weight': '384.38', 'logP': '2.62', 'h_bond_donors': '1', 'h_bond_acceptors': '7', 'tpsa': '83.45', 'rotatable_bonds': '3', 'heavy_atom_count': '28', 'aromatic_proportion': '0.429', 'lipinski_compliant': 'True', 'drug_likeness': 'Good', 'bioavailability_score': 'Moderate'}
# RAG Chunks Retrieved: 5
# LLM Call History: [{'node': 'strategist', 'input': {'user_query': 'Analyze this molecule for MAOB inhibition. SMILES: COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O TARGET: MAOB', 'canonical_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O'}, 'output': {'plan': ['run_qsar_prediction', 'retrieve_rag_chunks', 'run_admet_prediction'], 'rag游戏副本': 'mechanism of MAOB inhibition by flavonoid-like compounds and structure-activity relationships'}}, {'node': 'synthesizer', 'input': {'qsar_results': "{'pIC50': np.float64(-4.138062866206541), 'IC50_nM': np.float64(13742.408885666036), 'activity_level': 'Low', 'uncertainty': np.float64(0.8705036195325653), 'model_path': '..\\\\qsar_models\\\\MAOB_qsar_model.pkl', 'model_version': 'v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'target': 'MAOB'}", 'admet_results': "{'tool': 'admet_rdkit_v1', 'input_smiles': 'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O', 'molecular_weight': '384.38', 'logP': '2.62', 'h_bond_donors': '1', 'h_bond_acceptors': '7', 'tpsa': '83.45', 'rotatable_bonds': '3', 'heavy_atom_count': '28', 'aromatic_proportion': '0.429', 'lipinski_compliant': 'True', 'drug_likeness': 'Good', 'bioavailability_score': 'Moderate'}", 'rag_content': 'showed good potential in inhibiting MAO-B. In particular,\ncompound 31, as a reference molecule, showed the highest\npharmacological activity in the experiments. Therefore, we used\nit as a basis to derive more compounds by structural modiﬁcation in\norder to further explore the space for activity optimization.\nWe made theoretical predictions of the activities of these\ncompounds using quantitative structure-effect relationship\n(QSAR) models. By COMSIA method, we constructed 3D-QSAR\nand successfully predicted the IC50 values of these compounds. To\nverify the accuracy of the theoretical predictions, we compared the\npredicted results with the actual experimental data and found good\nagreement between them. This further conﬁrmed the structural\nrationality and predictability of our selected compound set in\nterms of activity.\nThe superior performance of compound 31.j3, as evidenced by\nits highest predicted IC50 value, molecular docking score, and\nstable binding in molecular dynamics simulations,\n\nemphasize the importance of these non-covalent forces in\nstabilizing the complex.\nBy gaining a deeper understanding of the underlying\nmechanisms that govern the binding capabilities and stability of\nour compounds, we can more effectively guide the rational design of\nnovel therapeutic agents targeting MAO-B. These insights provide\nvaluable guidance for future drug discovery efforts aimed at\ndeveloping potent and stable MAO-B inhibitors for the treatment\nof neurodegenerative diseases.\n5 Conclusion\nThrough a broader comparison encompassing IC50 values,\nmolecular docking scores, molecular dynamics simulations,\nbinding free energies, and key amino acid residue contributions,\nwe have demonstrated that the newly designed compound\n31.j3 exhibits superior performance as a potential MAO-B\ninhibitor. Its efﬁcient inhibitory activity, stable binding to the\nMAO-B receptor, and favorable interactions with key amino acid\nresidues collectively support its development as a promising\n\nto tau phosphorylation (Behl et al., 2021). Alpha Synuclein (α-Syn)\nand tau are signiﬁcant neuropathogenic proteins that have a crucial\nfunction in neurodegenerative disorders ( Wilson et al., 2023 ;\nZbinden et al., 2020 ). Compounds containing MAO-B have\nshown encouraging outcomes in the treatment of\nneurodegenerative disorders such as Parkinson ’s disease and\nAlzheimer’s disease (Lu et al., 2013).\nAt present, a limited number of MAO inhibitors have been\napproved for commercial use, such as selegiline (R-(−)-deprenyl)\nand rasagiline, which act in an irreversible manner (Liu et al., 2015),\nand saﬁnamide, a reversible MAO inhibitor (deSouza and Schapira,\n2017). However, these inhibitors are associated with various side\neffects and limited efﬁcacy. Recent studies have explored novel 6-\nhydroxybenzothiazole-2-carboxamide derivatives as potential\nMAO-B inhibitors, showing promising results in terms of\nIC50 values (Al-Saad et al., 2024). Our study aims to conduct a\n\ndocking, a small fragment of the MAO-B receptor (crystal structure\nof MAO-B obtained from the RCSB PDB Protein Data Bank, PDB\nID: 3PO7) (Al-Saad et al., 2024) was removed from the crystalline\nwater molecules and hydrogenated atoms. The original ligand in the\nMAO-B fragment was then extracted and its binding site was\ndetermined as shown inFigure 5.\nThe Sybyl-X program was used to perform ﬂexible docking\nbetween small molecule ligands and receptors. Activity pockets\nwere produced by identifying and using the binding sites of the\ntarget ligands. The threshold parameter was conﬁgured to a value\nof 0.5, the expansion factor was set to 1, and the docking process\nwas executed using the Sybyl-Dock standard mode. The\nmolecular conformational changes were preserved for a\nduration of 20 units of time. The evaluation of the interaction\nbetween the small molecule and the target was conducted using\nt h et o t a ls c o r ef u n c t i o no ft h eS y b y l - D o c km o d u l e .T h eT o t a l -\n\nrasagiline, 6-hydroxybenzothiazole-2-carboxamide derivatives\nfeature unique amide substituent modi ﬁcations in their\nstructures. These modiﬁcations not only enhance their selective\ninhibitory effect on MAO-B but also reduce the risk of side effects.\nParticularly, by introducing different side chains, 6-\nhydroxybenzothiazole-2-carboxamide derivatives achieve precise\nregulation of MAO-B activity, which is uncommon among\ncurrent medications. Furthermore, 6-hydroxybenzothiazole-2-\ncarboxamide derivatives demonstrate favorable pharmacokinetic\nproperties and bioavailability in both in vitro and in vivo\nFrontiers inPharmacology frontiersin.org02\nXie et al. 10.3389/fphar.2025.1545791'}, 'output': AIMessage(content='### 1. **Activity Prediction for MAOB Inhibition**  \n- **QSAR Results**: Predicted **pIC50 = -4.14 (IC50 = 13,742 nM)**, indicating **low activity** against MAOB.  \n- **Uncertainty**: High model uncertainty (**±0.87**), suggesting limited confidence in the prediction.  \n- **Comparison to Literature**: Compound 31 (from the cited study) achieved superior activity (**lower IC50**), highlighting the need for structural improvements.  \n\n---\n\n### 2. **Drug-Likeness and ADMET Properties**  \n- **Drug-Like Metrics**:  \n  - **Molecular Weight**: 384.38 g/mol (within acceptable range).  \n  - **LogP**: 2.62 (moderate lipophilicity, favorable for CNS penetration).  \n  - **H-Bond Donors/Acceptors**: 1/7 (moderate hydrogen bonding potential).  \n  - **TPSA**: 83.45 Å² (high polar surface area may reduce membrane permeability).  \n- **Compliance**: Passes **Lipinski’s Rule of Five**, with **"Good" drug-likeness**.  \n- **Bioavailability**: **Moderate**—may require optimization for higher CNS penetration.  \n\n---\n\n### 3. **Key Insights from Literature**  \n- **Reference Compounds**:  \n  - **Compound 31/31.j3** demonstrated strong MAOB inhibition via **3D-QSAR and molecular docking**, with stable binding to key residues (e.g., non-covalent interactions).  \n  - **6-Hydroxybenzothiazole-2-carboxamide derivatives** are highlighted as promising scaffolds due to **selectivity**, **low side effects**, and **favorable pharmacokinetics**.  \n- **Design Guidance**: Structural features like **amide substituents**, **aromatic rings**, and **hydroxyl groups** enhance activity and binding stability.  \n\n---\n\n### 4. **Actionable Next Steps for Optimization**  \n1. **Structural Modifications**:  \n   - **Incorporate amide/benzothiazole moieties** (as in 6-hydroxybenzothiazole derivatives) to improve binding affinity.  \n   - **Add hydrogen bond donors** (e.g., -NH2, -OH) to mimic interactions seen in compound 31.j3.  \n   - **Simplify/adjust ester groups** (e.g., replace with amides or ketones) to reduce polar surface area and enhance stability.  \n\n2. **Computational Validation**:  \n   - Perform **molecular docking** (using PDB ID: 3PO7) to identify critical binding interactions (e.g., with residues F356, Y407).  \n   - Use **3D-QSAR/COMSIA** to guide structure-activity relationship (SAR) analysis.  \n\n3. **Experimental Testing**:  \n   - Validate **predicted IC50** via in vitro MAOB inhibition assays.  \n   - Assess **selectivity** against MAO-A to minimize off-target effects.  \n\n4. **ADMET Optimization**:  \n   - Reduce **TPSA** by introducing hydrophobic moieties (e.g., aryl groups) to improve CNS penetration.  \n   - Evaluate **metabolic stability** and **blood-brain barrier (BBB) permeability**.  \n\n5. **Iterative Design**:  \n   - Leverage insights from **compound 31.j3’s success** (e.g., stable non-covalent interactions) to refine the scaffold for higher potency.  \n\n--- \n\n**Summary**: The molecule shows **low predicted MAOB inhibition** but has a **drug-like profile**. Optimization via structural modifications inspired by literature (e.g., benzothiazole scaffolds, amide linkers) and computational/experimental validation is critical to enhance activity and selectivity.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 1636, 'prompt_tokens': 1671, 'total_tokens': 3307, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_provider': 'openai', 'model_name': 'qwen/qwen3-32b', 'system_fingerprint': None, 'id': 'gen-1760835168-8eIGo1OMtWgJluSjgOPr', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--591657a5-e731-40f4-be7e-424f0a2def64-0', usage_metadata={'input_tokens': 1671, 'output_tokens': 1636, 'total_tokens': 3307, 'input_token_details': {}, 'output_token_details': {}})}]
# Target Used: MAOB

