
import sys
import os
import pandas as pd
from typing import List, Dict, Any
import numpy as np


# Add parent directory to path for importing qsar_training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qsar_training import GeneralQSARModel 

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
import joblib
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from state import AgentState

# --- QSAR Tool ---


qsar_model_registry = {
    "tubulin": os.path.join("..", "qsar_models", "tubulin_qsar_model.pkl"),
    "AChE": os.path.join("..", "qsar_models", "AChE_qsar_model.pkl"),
    "Breast_cancer_ANN": os.path.join("..", "qsar_models", "Breast_cancer_ANN_qsar_model.pkl"),
    "Ecoli": os.path.join("..", "qsar_models", "Ecoli_qsar_model.pkl"),
    "FAK": os.path.join("..", "qsar_models", "FAK_qsar_model.pkl"),
    "GlyT1": os.path.join("..", "qsar_models", "GlyT1_qsar_model.pkl"),
    "MAOB": os.path.join("..", "qsar_models", "MAOB_qsar_model.pkl"),
    "Scaffold_based_discovery": os.path.join("..", "qsar_models", "Scaffold_based_discovery_qsar_model.pkl")
}

qsar_agents = {}
for target, path in qsar_model_registry.items():
    agent = GeneralQSARModel()  
    try:
        agent.load(path)
        qsar_agents[target] = agent
        print(f"QSAR model '{target}' loaded successfully from {path}")
    except Exception as e:
        print(f"Warning: Could not load QSAR model '{target}': {e}")



def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True) if mol else ""



def run_qsar_prediction(state: AgentState, target: str = "tubulin") -> dict:
    """Runs QSAR prediction for a given canonical SMILES and target."""
    smiles = state.get("canonical_smiles")
    if not smiles:
        return {"error": "No canonical SMILES found for QSAR prediction."}

    qsar_agent = qsar_agents.get(target)
    if not qsar_agent:
        return {"error": f"No QSAR model loaded for target '{target}'."}

    print(f"Running QSAR prediction for target '{target}': {smiles}")
    preds, uncs, interps = qsar_agent.predict([smiles])

    if preds is None or not interps:
        return {"error": f"QSAR prediction failed for SMILES: {smiles}"}

    result = interps[0]
    result.update({
        'model_path': qsar_model_registry[target],
        'model_version': "v1",
        'input_smiles': smiles,
        'target': target
    })
    return result




# --- RAG Tool ---
CHROMA_DIR = os.path.join("..", "db", "paper_index")
COLLECTION_NAME = "paper_index"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def retrieve_rag_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve relevant scientific literature chunks from ChromaDB."""
    try:
        db = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print(f"Retrieving {k} chunks for query: '{query}'")
        docs = db.similarity_search(query, k=k)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    except Exception as e:
        print(f"RAG retrieval failed: {e}")
        return [{"error": f"RAG retrieval failed: {e}"}]
    


def _sanitize_value(v):
    """Convert numpy scalars and RDKit types to python builtins."""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    # RDKit types are usually plain numerics; fallback:
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)

def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_dict(v)
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(sanitize_dict(item))
                else:
                    new_list.append(_sanitize_value(item))
            out[k] = new_list
        else:
            out[k] = _sanitize_value(v)
    return out

# ------------------------------
# ADMET Tool (finalized)
# ------------------------------
def run_admet_prediction(state: AgentState) -> Dict[str, Any]:
    """
    Predict ADMET properties using RDKit descriptors (Lipinski, LogP, TPSA, etc.)
    Returns a structured dict (JSON-safe).
    """
    smiles = state.get("canonical_smiles")
    if not smiles:
        return {"error": "No canonical SMILES found for ADMET prediction."}

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {"error": "Invalid SMILES for ADMET calculation"}

    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        aromatic_proportion = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / mol.GetNumAtoms()

        lipinski_pass = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        drug_likeness = "Good" if lipinski_pass and tpsa <= 140 and rotatable_bonds <= 10 else "Poor"
        bioavailability_score = "Moderate" if (0 < logp < 3 and tpsa < 140) else "Low"

        result = {
            "tool": "admet_rdkit_v1",
            "input_smiles": smiles,
            "molecular_weight": round(float(mw), 2),
            "logP": round(float(logp), 2),
            "h_bond_donors": int(hbd),
            "h_bond_acceptors": int(hba),
            "tpsa": round(float(tpsa), 2),
            "rotatable_bonds": int(rotatable_bonds),
            "heavy_atom_count": int(heavy_atom_count),
            "aromatic_proportion": round(float(aromatic_proportion), 3),
            "lipinski_compliant": bool(lipinski_pass),
            "drug_likeness": drug_likeness,
            "bioavailability_score": bioavailability_score
        }
        return sanitize_dict(result)
    except Exception as e:
        return {"error": f"ADMET calculation failed: {str(e)}"}




# pip install langchain-openai
# pip install langchain-groq