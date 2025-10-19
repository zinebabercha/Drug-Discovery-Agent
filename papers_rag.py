import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "data"
CHROMA_DIR = "db/paper_index"
COLLECTION_NAME = "paper_index"   # explicit collection name

# -------- check pdfs ----------
pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
print("PDF files found:", pdf_files)
if not pdf_files:
    raise SystemExit("No PDFs in data/")

# splitter + embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded")

def parse_metadata(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    methods_list = parts[1:-2] if len(parts) > 3 else []
    methods_str = ", ".join(methods_list) if methods_list else ""
    return {
        "year": parts[0] if len(parts) > 0 else "",
        "methods": methods_str,
        "target": parts[-2] if len(parts) > 2 else "",
        "short_title": parts[-1] if len(parts) > 1 else "",
        "source_file": filename
    }

all_documents = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_file)
    print(f"Processing {pdf_file}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunks = text_splitter.split_documents(pages)
    metadata = parse_metadata(pdf_file)
    for chunk in chunks:
        chunk.metadata.update(metadata)
    all_documents.extend(chunks)

print("Total chunks:", len(all_documents))
if not all_documents:
    raise SystemExit("No chunks produced")

# create (or overwrite) Chroma DB with explicit collection name and persist directory
db = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
)
# some langchain-chroma wrappers require an explicit persist; call if available
try:
    db.persist()
    print("DB persisted via db.persist()")
except Exception:
    # newer versions persist automatically; ignore if not available
    print("db.persist() not available or not needed")

print("✅ Indexed and persisted to", CHROMA_DIR)








# pip install pypdf pdfminer.six langchain_text_splitters sentence-transformers chromadb
#pip install langchain --upgrade
#pip install unstructured
#pip install langchain-community langchain-text-splitters sentence-transformers chromadb unstructured
#pip install -U langchain-huggingface








# (drug) C:\Users\DELL\Downloads\drug>python papers_rag.py
# Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
# It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe
# PDF files found: ['2022_QSAR_ADMET_DOCK_MD_GlyT1_inhibitors.pdf', '2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf', '2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf', '2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf', '2024_QSAR_DOCK_MD_tuberculosis_nitrofuran.pdf', '2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf', '2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf', '2025_QSAR_ADMET_DOCK_MD_BTK_pyrrolopyrimidine.pdf', '2025_QSAR_ADMET_DOCK_MD_FAK_inhibitors.pdf', '2025_QSAR_ADMET_DOCK_MD_tuberculosis_multitarget.pdf', '2025_QSAR_ANN_DOCK_ADMET_MD_breast_cancer_agents.pdf', '2025_QSAR_DOCK_MD_AChE_inhibitors.pdf', '2025_QSAR_DOCK_MD_AuroraKinase_imidazopyridine.pdf', '2025_QSAR_DOCK_MD_DFT_antitubercular_agents.pdf', '2025_QSAR_DOCK_MD_Ecoli_flavonoids.pdf', '2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf', '2025_QSAR_DOCK_MD_scaffold_drug_discovery.pdf']
# Embedding model loaded
# Processing 2022_QSAR_ADMET_DOCK_MD_GlyT1_inhibitors.pdf
# Processing 2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf
# Processing 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Processing 2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf
# Processing 2024_QSAR_DOCK_MD_tuberculosis_nitrofuran.pdf
# Processing 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Processing 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Processing 2025_QSAR_ADMET_DOCK_MD_BTK_pyrrolopyrimidine.pdf
# Processing 2025_QSAR_ADMET_DOCK_MD_FAK_inhibitors.pdf
# Processing 2025_QSAR_ADMET_DOCK_MD_tuberculosis_multitarget.pdf
# Processing 2025_QSAR_ANN_DOCK_ADMET_MD_breast_cancer_agents.pdf
# Processing 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Processing 2025_QSAR_DOCK_MD_AuroraKinase_imidazopyridine.pdf
# Processing 2025_QSAR_DOCK_MD_DFT_antitubercular_agents.pdf
# Processing 2025_QSAR_DOCK_MD_Ecoli_flavonoids.pdf
# Processing 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Processing 2025_QSAR_DOCK_MD_scaffold_drug_discovery.pdf
# Total chunks: 1749
# C:\Users\DELL\Downloads\drug\papers_rag.py:60: LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
#   db.persist()
# DB persisted via db.persist()
# ✅ Indexed and persisted to db/paper_index


