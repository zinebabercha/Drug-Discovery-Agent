from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Directory where your RAG database is persisted
CHROMA_DIR = "db/paper_index"
COLLECTION_NAME = "paper_index"

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to your Chroma vector store
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# List of queries you want to test
queries = [
    "tubulin inhibitor binding",
    "VEGFR2 inhibition",
    "antimicrobial mannopyranosides",
    "breast cancer QSAR analysis",
    "AChE inhibitors",
    "antitubercular drug candidates",
    "pyrazole antifungal agents",
    "MAO B inhibitors",
    "Aurora kinase inhibitors",
    "drug scaffold discovery"
]

# Loop over each query and print results
for query in queries:
    print(f"\n🔍 Searching for: '{query}'")
    results = db.similarity_search(query, k=5)
    print("Found:", len(results))
    for r in results:
        print("Source:", r.metadata.get("source_file"))
        print("Title:", r.metadata.get("short_title"))
        print("Snippet:", r.page_content[:300], "...\n")






# (drug) C:\Users\DELL\Downloads\drug>python test_rag.py
# Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
# It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe
# C:\Users\DELL\Downloads\drug\test_rag.py:12: LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. An updated version of the class exists in the `langchain-chroma package and should be used instead. To use it run `pip install -U `langchain-chroma` and import as `from `langchain_chroma import Chroma``.
#   db = Chroma(

# 🔍 Searching for: 'tubulin inhibitor binding'
# Found: 5
# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: for the rational design of drug candidates. By identifying and characterizing novel inhibitors that can potentially
# overcome resistance mechanisms or reduce adverse effects associated with current treatments, this work paves
# the way for more targeted and effective cancer treatment strategies. Look ...

# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: ics simulations. These findings not only enhance our understanding of the chemical interactions necessary for
# Tubulin inhibition but also highlight the robust potential of these compounds to serve as templates for devel-
# oping new anticancer drugs. The integration of multiple computational techniqu ...

# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: 11
# Vol.:(0123456789)Scientific Reports |        (2024) 14:16418  | https://doi.org/10.1038/s41598-024-66877-2
# www.nature.com/scientificreports/
# Microtubule protein A and B chains: structural and functional insights
# Microtubules, essential components of the cytoskeleton, are composed of α-Tubulin (A  ...

# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: statistics highlight the persistent challenge breast cancer poses and underscore the critical need for developing
# more effective therapeutic strategies.
# The 1,2,4-triazine-3(2H)-one derivatives have emerged as a promising class of anticancer agents, with sev-
# eral studies reporting their potent ant ...

# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: 2
# Vol:.(1234567890)Scientific Reports |        (2024) 14:16418  | https://doi.org/10.1038/s41598-024-66877-2
# www.nature.com/scientificreports/
# cytoskeleton and are crucial for cell division, making them an attractive target for cancer therapy. Colchicine-
# binding to Tubulin disrupts its polymerizati ...


# 🔍 Searching for: 'VEGFR2 inhibition'
# Found: 5
# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: [2]. Some cancers particularly colorectal and renal cancers are characterized by high levels of VEGF. Hence, VEGFR-2 is
# Supplementary Information The online version contains supplementary material available at https:// doi. org/ 10. 1007/ s44371- 025-
# 00173-4.
#  * Yogita Bansal, yogita_pharma@pbi.a ...

# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: also indicated stable complex with binding free energy of 0.28 kcal/mol. All the designed molecules were also found to
# be possess good ADMET profile. These findings indicate that the quinazoline nucleus can be investigated and refined to
# create novel VEGFR-2 inhibitors as potential anti-angiogenic ...

# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: Vol.:(0123456789)
# Discover Chemistry           (2025) 2:111  | https://doi.org/10.1007/s44371-025-00173-4

#  Research
#  44. Abdallah AE, Mabrouk RR, Al Ward MMS, Eissa SI, Elkaeed EB, Mehany AB, Abo-Saif MA, El-Feky OA, Alesawy MS, El-Zahabi MA. Synthesis,
# biological evaluation, and molecular docki ...

# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: compounds. In terms of excretion, all compounds showed moderate clearance rate (3.70–4.59 ml/min/kg). Moreover,
# compounds DDSL6a-c showed half-life of more than 3 h while the rest showed less than 3 h. Organic cation transporter 2
# was not found to be inhibited by the designed compounds that sugges ...

# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: Vol:.(1234567890)
# Research
# Discover Chemistry           (2025) 2:111  | https://doi.org/10.1007/s44371-025-00173-4
# exhibits majority of the necessary interactions with key amino acids which are essential for the affinity towards VEGFR-
# 2. These residues include Asp1046, Phe1047, Leu889, Cys919, Cy ...


# 🔍 Searching for: 'antimicrobial mannopyranosides'
# Found: 5
# Source: 2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf
# Title: antimicrobial
# Snippet: JOURNALOFTAIBAHUNIVERSITYFORSCIENCE
# 2024,VOL.18,NO.1,2327101
# https://doi.org/10.1080/16583655.2024.2327101
# Synthesis, antimicrobial activity, molecular docking, molecular dynamics
# simulation, and ADMET properties of the mannopyranoside derivatives as
# antimicrobial agents
# Anis Ul Islama, Hanine Hadni ...

# Source: 2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf
# Title: antimicrobial
# Snippet: Journal of Taibah University for Science
# ISSN: 1658-3655 (Online) Journal homepage: www.tandfonline.com/journals/tusc20
# Synthesis, antimicrobial activity, molecular
# docking, molecular dynamics simulation, and
# ADMET properties of the mannopyranoside
# derivatives as antimicrobial agents
# Anis Ul Islam,  ...

# Source: 2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf
# Title: antimicrobial
# Snippet: human and plant pathogenic bacteria and fungi.
# Figure 2 shows the study process’s essential com-
# ponents, techniques, and sequential procedures in a
# flow diagram. For testing, we needed a wide range
# of substituted mannopyranosides in a single-molecule
# framework. Numerous acylated monosaccharide com- ...

# Source: 2024_DOCK_MD_ADMET_mannopyranoside_antimicrobial.pdf
# Title: antimicrobial
# Snippet: regioselective acylation of methylα-D-mannopyranoside (MDMP) derivatives2-6 afforded from
# the 6-O-butyryl derivative. This isolated 6-O-derivative was converted to 2,3,4-tri-O-acyl deriva-
# tives, and the resulting compounds were analyzed using FTIR,
# 1H-NMR, 13C-NMR, and elemental
# analysis. The acyla ...

# Source: 2024_QSAR_DOCK_MD_tuberculosis_nitrofuran.pdf
# Title: nitrofuran
# Snippet: strains of TB [14]
# Compounds with some antibacterial activity may be considered as a good source of new leads for
# TB drug development. Nitrofuranylamide, metronidazole, nitrofurantoin and nitroimidazole pyran, are
# some antibacterial agents (Fig ure 1) used in different microbial infections  [15,1 ...


# 🔍 Searching for: 'breast cancer QSAR analysis'
# Found: 5
# Source: 2025_QSAR_ANN_DOCK_ADMET_MD_breast_cancer_agents.pdf
# Title: agents
# Snippet: activity as well as the types of interactions with the active site of aro -
# matase. In addition, ADMET predictions have been used to assess spe -
# cific pharmacokinetic properties as well as to virtually screen designed
# molecules against breast cancer. Finally, MD parameters including
# RMSD, RMSF an ...

# Source: 2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf
# Title: derivatives
# Snippet: the following parameters, and the prediction ability of QSAR model can be further verified
# by calculating the biological activity of compounds in the test set [37,38].
# Prediction correlation coefficient q2
# pred based on test set. ∑n
# i=1(yi − ˆyi)2 is the sum of
# squares of the actual molecular activi ...

# Source: 2025_QSAR_DOCK_MD_AuroraKinase_imidazopyridine.pdf
# Title: imidazopyridine
# Snippet: the following parameters, and the prediction ability of QSAR model can be further verified
# by calculating the biological activity of compounds in the test set [37,38].
# Prediction correlation coefficient q2
# pred based on test set. ∑n
# i=1(yi − ˆyi)2 is the sum of
# squares of the actual molecular activi ...

# Source: 2025_QSAR_ADMET_DOCK_MD_FAK_inhibitors.pdf
# Title: inhibitors
# Snippet: continue to be screened by the built QSAR model.
# Fig. 5  Comparison of the redocked binding pose (yellow) with the
# experimental pose of the co-ligand ...

# Source: 2025_QSAR_ADMET_DOCK_MD_BTK_pyrrolopyrimidine.pdf
# Title: pyrrolopyrimidine
# Snippet: the constructed QSAR model is not significantly influenced by the cross-
# validation method ( Fig. 7 ) . These clear results demonstrate the stability
# and robustness of the proposed QSAR model. However, it should be
# noted that cross-validation is not an adequate measure to fully evaluate
# the capa ...


# 🔍 Searching for: 'AChE inhibitors'
# Found: 5
# Source: 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Title: inhibitors
# Snippet: which is involved in memory, learning, and other behavior; moreover, deficiency of this
# neurotransmitter ACh leads to cognitive disorders [16]. Based on these facts, the therapeutic
# approach currently considered for AD consists of increasing cholinergic neurotransmission
# in the brain by reducing unw ...

# Source: 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Title: inhibitors
# Snippet: Ligands M2 M1 M6 Donepezil
# Score (kcal/mol) −13 −12.6 −12.4 −10.8
# The molecular docking study identified the key residues involved in ligand binding to
# AChE. Among these residues, TRP86 and TRP286 were identified as crucial for inhibitor
# binding to AChE.
# The study of mutagenesis at AChE makes it pos ...

# Source: 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Title: inhibitors
# Snippet: conducted to explore their AChE inhibition mechanisms and binding affinities in the active site.
# The binding scores of compounds M1, M2, and M6 were ( −12.6 kcal/mol), (−13 kcal/mol), and
# (−12.4 kcal/mol), respectively, which are higher than the standard inhibitor Donepezil with a binding
# score of ( ...

# Source: 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Title: inhibitors
# Snippet: Pharmaceuticals 2024, 17, 830 9 of 23
# Pharmaceuticals 2024, 17, x FOR PEER REVIEW  9 of 23 
#  
#  
#    
# (a)  (b) 
#    
# (c)  (d) 
#  
# Figure 5. Two-dimensional interactions between AChE and the ligands M2 (a), M6 (b), M1 (c), and 
# Donepezil (d). 
# All of the many non-covalent connections that make up the com ...

# Source: 2025_QSAR_DOCK_MD_AChE_inhibitors.pdf
# Title: inhibitors
# Snippet: Citation: Khedraoui, M.; Abchir, O.;
# Nour, H.; Yamari, I.; Errougui, A.;
# Samadi, A.; Chtita, S. An In Silico
# Study Based on QSAR and Molecular
# Docking and Molecular Dynamics
# Simulation for the Discovery of Novel
# Potent Inhibitor against AChE.
# Pharmaceuticals 2024, 17, 830. https://
# doi.org/10.3390/p ...


# 🔍 Searching for: 'antitubercular drug candidates'
# Found: 5
# Source: 2025_QSAR_ADMET_DOCK_MD_tuberculosis_multitarget.pdf
# Title: multitarget
# Snippet: candidates, and their targets. Med Res Rev. 2020;40:263–92. https:// doi.
# org/ 10. 1002/ med. 21602.
#  23. Capel R, Félix R, Clariano M, Nunes D, Perry MDJ, Lopes F. Target identifica-
# tion in anti-tuberculosis drug discovery. Int J Mol Sci. 2023;24:10482.
# https:// doi. org/ 10. 3390/ ijms2 41310 4 ...

# Source: 2024_QSAR_DOCK_MD_tuberculosis_nitrofuran.pdf
# Title: nitrofuran
# Snippet: AIMS Molecular Science, 11(1): 1–20.
# DOI: 10.3934/molsci.2024001
# Received: 28 September 2023
# Revised: 05 December 2023
# Accepted: 06 December 2023
# Published: 04 January 2024
# http://www.aimspress.com/journal/Molecular

# Research article
# 2D-QSAR and molecular docking study on nitrofuran analogu ...

# Source: 2024_QSAR_DOCK_MD_tuberculosis_nitrofuran.pdf
# Title: nitrofuran
# Snippet: 20
# AIMS Molecular Science  V olume 11, Issue 1, 1–20.
# 25. Tawari NR, Degani MS (2010) Pharmacophore mapping and electronic feature analysis for a
# series of nitroaromatic compounds with antitubercular activity. J Comput Chem  31: 739–751.
# https://doi.org/10.1002/jcc.21371
# 26. Talete srl (2013)   ...

# Source: 2024_QSAR_ADMET_DOCK_MD_tubulin_triazine_inhibitors.pdf
# Title: inhibitors
# Snippet: for the rational design of drug candidates. By identifying and characterizing novel inhibitors that can potentially
# overcome resistance mechanisms or reduce adverse effects associated with current treatments, this work paves
# the way for more targeted and effective cancer treatment strategies. Look ...

# Source: 2025_QSAR_ADMET_DOCK_MD_tuberculosis_multitarget.pdf
# Title: multitarget
# Snippet: Page 26 of 28Panigrahi and Sahu  BMC Chemistry           (2025) 19:39
# Received: 14 March 2024   Accepted: 28 November 2024
# References
#  1. Sachan RSK, Mistry V, Dholaria M, Rana A, Devgon I, Ali I, Iqbal J, Eldin
# SM, Said ARM, Tawaha A, Bawazeer S, Dutta J, Karnwal A. Overcoming
# mycobacterium tube ...


# 🔍 Searching for: 'pyrazole antifungal agents'
# Found: 5
# Source: 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Title: antifungal
# Snippet: value < 0.01, ***: p-value < 0.001, and ****: p-value < 0.0001.

# Figure 10. Antifungal activity of pyrazole derivatives against the fungal strains tested.
# Table 5. MICs and MBCs of new pyrazole derivatives with standard Streptomycine and Fluconazole
# against bacterial and fungal strains expressed ...

# Source: 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Title: antifungal
# Snippet: the lowest MIC at 0.0156 mg/mL, whereas 9c exhibited the highest MIC at 0.25 mg/mL.
# Our results are in contradiction to those of [36], which indicated that various synthesized
# pyrazole–thiazole derivatives showed important antibacterial activity against Pseudomonas
# aeruginosa MTCC 2642. Compound 6c  ...

# Source: 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Title: antifungal
# Snippet: contradiction to those of [36], which indicated that various synthesized pyrazole–thiazole
# derivatives showed important antibacterial activity against Pseudomonas aeruginosa MTCC
# 2642. Compound 6c showed the strongest activity, with a minimum inhibitory concentra-
# tion (MIC) of 0.005 mg/mL, outper ...

# Source: 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Title: antifungal
# Snippet: Pharmaceuticals 2025, 18, 167 12 of 35
# outperforming fluconazole’s 8.16 ± 2.04 mm. Other O compounds showed inhibition
# diameters ranging from 10 to 18 mm, further indicating greater antifungal potential than
# fluconazole across both fungal strains. Overall O2 and O5 showed moderate activity, while
# O1 ...

# Source: 2025_DFT_DOCK_ADMET_pyrazole_antimicrobial_antifungal.pdf
# Title: antifungal
# Snippet: Pharmaceuticals 2025, 18, 167 34 of 35
# 33. Sharma, P .K.; Chandak, N.; Kumar, P .; Sharma, C.; Aneja, K.R. Synthesis and biological evaluation of some 4-functionalized-
# pyrazoles as antimicrobial agents. Eur. J. Med. Chem. 2011, 46, 1425–1432. [CrossRef] [PubMed]
# 34. Nandurkar, Y.; Bhoye, M.R.; Mali ...


# 🔍 Searching for: 'MAO B inhibitors'
# Found: 5
# Source: 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Title: 6hydroxybenzothiazole
# Snippet: to tau phosphorylation (Behl et al., 2021). Alpha Synuclein (α-Syn)
# and tau are signiﬁcant neuropathogenic proteins that have a crucial
# function in neurodegenerative disorders ( Wilson et al., 2023 ;
# Zbinden et al., 2020 ). Compounds containing MAO-B have
# shown encouraging outcomes in the treatment  ...

# Source: 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Title: 6hydroxybenzothiazole
# Snippet: rasagiline, 6-hydroxybenzothiazole-2-carboxamide derivatives
# feature unique amide substituent modi ﬁcations in their
# structures. These modiﬁcations not only enhance their selective
# inhibitory effect on MAO-B but also reduce the risk of side effects.
# Particularly, by introducing different side chains ...

# Source: 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Title: 6hydroxybenzothiazole
# Snippet: emphasize the importance of these non-covalent forces in
# stabilizing the complex.
# By gaining a deeper understanding of the underlying
# mechanisms that govern the binding capabilities and stability of
# our compounds, we can more effectively guide the rational design of
# novel therapeutic agents targetin ...

# Source: 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Title: 6hydroxybenzothiazole
# Snippet: individuals, it is essential to advance the development of novel
# pharmaceuticals and implement innovative treatment approaches.
# Monoamine oxidase (MAO) is situated on the external
# mitochondrial membrane and consists of two isoforms, MAO-A
# and MAO-B. MAO-B is mostly located in the brain, platelets, a ...

# Source: 2025_QSAR_DOCK_MD_MAO_B_6hydroxybenzothiazole.pdf
# Title: 6hydroxybenzothiazole
# Snippet: Conclusion: In this study, the potential of 6-hydroxybenzothiazole-2-
# carboxamide derivatives as MAO-B inhibitors was systematically investigated by
# 3D-QSAR, molecular docking and MD simulations. The successfully designed
# compound 31.j3 not only demonstrated ef ﬁcient inhibitory activity, but also
# v ...


# 🔍 Searching for: 'Aurora kinase inhibitors'
# Found: 5
# Source: 2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf
# Title: derivatives
# Snippet: Molecules 2024, 29, 1772 2 of 31
# can lead to a disorder of genetic material distribution, which can affect the stability of the
# genome [3]. In recent years, relevant studies have confirmed that Aurora kinase is highly
# expressed in a variety of tumor cells and participates in the formation of tumors  ...

# Source: 2025_QSAR_DOCK_MD_AuroraKinase_imidazopyridine.pdf
# Title: imidazopyridine
# Snippet: Molecules 2024, 29, 1772 2 of 31
# can lead to a disorder of genetic material distribution, which can affect the stability of the
# genome [3]. In recent years, relevant studies have confirmed that Aurora kinase is highly
# expressed in a variety of tumor cells and participates in the formation of tumors  ...

# Source: 2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf
# Title: derivatives
# Snippet: highly expressed in a variety of tumor cells and participates in the formation of tumors
# [4,5]. Aurora A is a subtype of Aurora kinase. The Aurora A gene is located in the chromo-
# somal region with active amplification. At present, many Aurora A compounds with dif-
# ferent inhibitory activities have ...

# Source: 2025_QSAR_DOCK_MD_AuroraKinase_imidazopyridine.pdf
# Title: imidazopyridine
# Snippet: highly expressed in a variety of tumor cells and participates in the formation of tumors
# [4,5]. Aurora A is a subtype of Aurora kinase. The Aurora A gene is located in the chromo-
# somal region with active amplification. At present, many Aurora A compounds with dif-
# ferent inhibitory activities have ...

# Source: 2024_QSAR_DOCK_MD_imidazopyridine_derivatives.pdf
# Title: derivatives
# Snippet: tumor invasiveness. (Figure 1 shows several kinds of Aurora kinase inhibitors entering
# clinical trials, including the first Aurora kinase inhibitor ZM-447439 (Figure 1(1)), the first
# Aurora B inhibitor, hesperidin (Figure 1(2)), and a new type of Aurora inhibitor, MLN-8237
# (Figure 1(3)).
# Molecules 2 ...


# 🔍 Searching for: 'drug scaffold discovery'
# Found: 5
# Source: 2024_QSAR_DOCK_MD_VEGFR2_quinazoline.pdf
# Title: quinazoline
# Snippet: Vol:.(1234567890)
# Research
# Discover Chemistry           (2025) 2:111  | https://doi.org/10.1007/s44371-025-00173-4
#  16. Hughes JP , Rees S, Kalindjian SB, Philpott KL. Principles of early drug discovery. Br J Pharmacol. 2011;162:1239–49. https:// doi. org/ 10.
# 1111/j. 1476- 5381. 2010. 01127.x.
#   ...

# Source: 2025_QSAR_ADMET_DOCK_MD_tuberculosis_multitarget.pdf
# Title: multitarget
# Snippet: learning (ML) [27]. Identifying hit molecules through
# computational drug discovery has proved to be a mean -
# ingful methodology in recent years [28]. Structure-
# based similarity search and screening, one of the various
# approaches to drug design and discovery, has become
# a routine concept in the d ...

# Source: 2025_QSAR_DOCK_MD_scaffold_drug_discovery.pdf
# Title: discovery
# Snippet: the first time, warrant synthesis and further evaluation of their biological activity through in vitro and in vivo
# studies.
# 1. Introduction
# Drug discovery is crucial for developing new treatments for a wide
# range of diseases [ 1 ]. The emergence of viruses like SARS-CoV-2 has
# underscored the critica ...

# Source: 2025_QSAR_DOCK_MD_scaffold_drug_discovery.pdf
# Title: discovery
# Snippet: effective therapeutic agents.
# While these predictions provide valuable information, it ’ s important
# to acknowledge the limitations of in silico ADME predictions [ 71 ].
# Experimental validation is crucial to confirm these properties and assess
# their overall suitability for drug development.
# 4. Concl ...

# Source: 2025_QSAR_DOCK_MD_scaffold_drug_discovery.pdf
# Title: discovery
# Snippet: optimize their properties, and ultimately accelerate the discovery pro-
# cess [ 5 ]. These computational methods go beyond simple screening,
# offering profound insights into drug action mechanisms. For example,
# recent studies have employed molecular docking and MD simulations to
# elucidate the dual-tar ...
