from chembl_webresource_client.new_client import new_client
import pandas as pd

def fetch_target_data(target_name, file_name):
    # Search target
    target = new_client.target
    results = target.search(target_name)
    if not results:
        print(f"No target found for {target_name}")
        return
    target_id = results[0]['target_chembl_id']
    print(f"Target {target_name}: {target_id}")

    # Fetch bioactivity data
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id, standard_type="IC50").only(
        ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']
    )

    # Convert to DataFrame
    res_list = list(res)
    if not res_list:
        print(f"No bioactivity data found for {target_name}")
        return
    df = pd.DataFrame(res_list)
    if 'canonical_smiles' in df.columns:
        df = df.dropna(subset=['canonical_smiles'])
    df.to_csv(file_name, index=False)
    print(f"Saved {len(df)} molecules to {file_name}")

# List of targets and filenames
targets = {
    # "GlyT1": "GlyT1_dataset.csv",
    # "Mannopyranoside": "Mannopyranoside_dataset.csv",
    # "Tubulin": "Tubulin_dataset.csv",
    "Imidazopyridine": "Imidazopyridine_dataset.csv",
    "VEGFR2": "VEGFR2_dataset.csv",
    "Pyrazole, antifungal": "Pyrazole_antifungal_dataset.csv",
    "BTK": "BTK_dataset.csv",
    "FAK": "FAK_dataset.csv",
    "AChE": "AChE_dataset.csv",
    "E. coli": "Ecoli_dataset.csv",
    "MAO-B": "MAOB_dataset.csv",
    "Scaffold-based discovery": "Scaffold_based_discovery_dataset.csv",
    "Breast cancer (ANN)": "Breast_cancer_ANN_dataset.csv",
    "Antitubercular": "Antitubercular_dataset.csv"
}


# Fetch datasets for all targets
for name, file in targets.items():
    fetch_target_data(name, file)





# pip install rdkit(-pypi) chembl_webresource_client pandas




# (drug) C:\Users\DELL\Downloads\drug>python tubulin.py
# C:\Users\DELL\Downloads\drug\drug\Lib\site-packages\chembl_webresource_client\__init__.py:4: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
#   __version__ = __import__('pkg_resources').get_distribution('chembl_webresource_client').version
# No target found for Imidazopyridine
# Target VEGFR2: CHEMBL5482997
# No bioactivity data found for VEGFR2
# No target found for Pyrazole, antifungal
# Target BTK: CHEMBL4879468
# Saved 1 molecules to BTK_dataset.csv
# Target FAK: CHEMBL5773
# Saved 3 molecules to FAK_dataset.csv
# Target AChE: CHEMBL4780
# Saved 436 molecules to AChE_dataset.csv
# Target E. coli: CHEMBL3233
# Saved 33 molecules to Ecoli_dataset.csv
# Target MAO-B: CHEMBL2039
# Saved 8649 molecules to MAOB_dataset.csv
# Target Scaffold-based discovery: CHEMBL5247
# Saved 324 molecules to Scaffold_based_discovery_dataset.csv
# Target Breast cancer (ANN): CHEMBL5990
# Saved 20 molecules to Breast_cancer_ANN_dataset.csv
# No target found for Antitubercular


