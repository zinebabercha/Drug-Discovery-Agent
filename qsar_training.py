import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors # Still used for GetMorganFingerprintAsBitVect
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator # New import for MorganGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import os

# Suppress all RDKit warnings for cleaner output, including deprecation warnings
# If you want to see other RDKit warnings, you can make this more specific
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings('ignore') # Keep general warnings ignored

class GeneralQSARModel:
    """General QSAR model for activity prediction across various targets"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        # Initialize Morgan fingerprint generator once
        self.fpg = GetMorganGenerator(radius=2, fpSize=2048) # Use MorganGenerator
        
    def smiles_to_features(self, smiles):
        """Convert SMILES to molecular descriptors + Morgan fingerprints"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Use the new MorganGenerator for fingerprints
        fp = self.fpg.GetFingerprint(mol)
        fp_array = np.array(fp)
        
        # Key molecular descriptors
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.FractionCSP3(mol),
        ])
        
        # Combine fingerprint + descriptors
        features = np.concatenate([fp_array, descriptors])
        return features
    

    def prepare_dataset(self, csv_path):
        """Load and featurize the dataset, handling invalid data safely"""
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Remove missing or invalid standard values
        df = df.dropna(subset=['standard_value'])
        df = df[df['standard_value'] > 0].copy() # Ensure positive values for log transformation
        
        # Convert to pIC50
        df['pIC50'] = -np.log10(df['standard_value'])
        
        print(f"Dataset: {len(df)} molecules")
        if not df.empty:
            print(f"Activity range (nM): {df['standard_value'].min():.1f} - {df['standard_value'].max():.1f}")
            print(f"pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
        else:
            print("No valid molecules found in dataset after filtering.")
            return np.array([]), np.array([]), pd.DataFrame()
        
        # Featurize
        print("\nFeaturizing molecules...")
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df['canonical_smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # print(f"Warning: Invalid SMILES at index {idx} in {csv_path}")
                continue # Skip invalid SMILES
            
            # Use the new MorganGenerator for fingerprints
            fp = self.fpg.GetFingerprint(mol)
            fp_array = np.array(fp)
            
            # Key molecular descriptors
            descriptors = np.array([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
            ])
            
            # Combine fingerprint + descriptors
            features = np.concatenate([fp_array, descriptors])
            features_list.append(features)
            valid_indices.append(idx)
        
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        X = np.array(features_list)
        y = df_valid['pIC50'].values
        
        # Final safety check for NaNs in y (activity values)
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Valid molecules: {len(X)} / {len(df)}")
        
        return X, y, df_valid

    
    def train(self, csv_path, test_size=0.2, random_state=42):
        """Train the QSAR model"""
        # Prepare data
        X, y, df_valid = self.prepare_dataset(csv_path)
        
        min_samples_for_split = 2 # Minimum samples required to get 1 train, 1 test sample
        if X.shape[0] < min_samples_for_split:
            print(f"Warning: Not enough valid molecules ({X.shape[0]}) in {csv_path} for train-test split. Skipping training.")
            return None
            
        # Split
        # Adjust test_size dynamically if needed for very small datasets, 
        # or simply ensure min_samples_for_split is respected.
        # For example, if X.shape[0] is 2, test_size=0.5 would give 1 train, 1 test.
        # The current min_samples_for_split check already handles cases where test_size=0.2 would fail.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # After splitting, double-check that both sets are non-empty
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: Train-test split resulted in an empty set for {csv_path}. Adjusting test_size if possible or skipping training.")
            # This case should ideally be caught by the min_samples_for_split,
            # but is a robust fallback for edge cases with specific test_size values.
            if X.shape[0] >= 2 and len(X_train) == 0: # Try with 1 sample for test if possible
                 X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1, random_state=random_state, shuffle=True
                 )
                 # This would put 1 sample in test, the rest in train
                 if len(X_train) == 0:
                     print(f"Still no valid train set even after adjusting test_size. Skipping training for {csv_path}.")
                     return None
            elif X.shape[0] >= 2 and len(X_test) == 0:
                 X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.5, random_state=random_state, shuffle=True
                 )
                 if len(X_test) == 0:
                     print(f"Still no valid test set even after adjusting test_size. Skipping training for {csv_path}.")
                     return None
            else:
                print(f"Cannot form valid train/test sets for {csv_path}. Skipping training.")
                return None


        print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}")
        
        # Train Random Forest
        print("\nTraining Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²:  {test_r2:.3f}")
        print(f"Test RMSE: {test_rmse:.3f} pIC50 units")
        print(f"Test MAE:  {test_mae:.3f} pIC50 units")
        print("="*50)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    def predict(self, smiles_list, return_uncertainty=True):
        """
        Predict activity for new molecules
        
        Returns:
            predictions: pIC50 predictions
            uncertainties: std dev from tree ensemble (if return_uncertainty=True)
            interpretations: human-readable activity ranges
        """
        if self.model is None:
            raise ValueError("Model not trained! Call .train() first.")
        
        features_list = []
        valid_smiles = []
        invalid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            feat = self.smiles_to_features(smiles)
            if feat is not None:
                features_list.append(feat)
                valid_smiles.append(smiles)
            else:
                invalid_indices.append(idx)
        
        if len(features_list) == 0:
            print("Warning: No valid SMILES provided for prediction.")
            return None, None, None
        
        X = np.array(features_list)
        predictions = self.model.predict(X)
        
        # Uncertainty from tree variance
        uncertainties = None
        if return_uncertainty:
            tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            uncertainties = np.std(tree_preds, axis=0)
        
        # Interpret activity levels
        interpretations = []
        for pred, unc in zip(predictions, uncertainties if uncertainties is not None else [None]*len(predictions)):
            ic50_nm = 10**(-pred)
            
            if ic50_nm < 10:
                level = "Very High"
            elif ic50_nm < 100:
                level = "High"
            elif ic50_nm < 1000:
                level = "Moderate"
            else:
                level = "Low"
            
            interp = {
                'pIC50': pred,
                'IC50_nM': ic50_nm,
                'activity_level': level,
                'uncertainty': unc
            }
            interpretations.append(interp)
        
        return predictions, uncertainties, interpretations
    
    def save(self, path):
        """Save trained model and feature names"""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, path)
        print(f"\nModel and feature names saved to: {path}")

    def load(self, path):
        """Load trained model and feature names"""
        data = joblib.load(path)
        self.model = data.get('model')
        self.feature_names = data.get('feature_names', None)
        print(f"Model loaded from: {path} (feature_names restored: {'Yes' if self.feature_names is not None else 'No'})")

# ============================================================
# USAGE EXAMPLE FOR ALL TARGETS
# ============================================================

if __name__ == "__main__":
    
    # List of all your target datasets
    target_datasets = [
        'AChE_dataset.csv',
        'Breast_cancer_ANN_dataset.csv',
        'BTK_dataset.csv',
        'Ecoli_dataset.csv',
        'FAK_dataset.csv',
        'GlyT1_dataset.csv',
        'MAOB_dataset.csv',
        'Scaffold_based_discovery_dataset.csv',
        'tubulin_dataset.csv' 
    ]

    # Directory to save the models
    models_dir = 'qsar_models'
    os.makedirs(models_dir, exist_ok=True) # Create directory if it doesn't exist

    for dataset_file in target_datasets:
        print(f"\n{'='*70}")
        target_name = os.path.splitext(dataset_file)[0].replace('_dataset', '') # Extract target name
        print(f"TRAINING QSAR MODEL FOR TARGET: {target_name.upper()}")
        print(f"{'='*70}")

        qsar_model_instance = GeneralQSARModel()
        
        results = qsar_model_instance.train(dataset_file)
        
        if results: # Only save if training was successful
            model_save_path = os.path.join(models_dir, f'{target_name}_qsar_model.pkl')
            qsar_model_instance.save(model_save_path)
        else:
            print(f"Skipped saving model for {target_name} due to training issues (e.g., insufficient data).")
        
        print(f"\nCompleted training for {target_name}.\n")

    print("\n" + "="*70)
    print("ALL QSAR MODELS TRAINED AND SAVED (where possible).")
    print("="*70)

    # Example of loading a specific model and making predictions
    print("\n" + "="*50)
    print("DEMONSTRATING LOADING AND PREDICTION FOR TUBULIN")
    print("="*50)

    # Load the tubulin model
    tubulin_model_path = os.path.join(models_dir, 'tubulin_qsar_model.pkl')
    loaded_tubulin_qsar = GeneralQSARModel()
    try:
        loaded_tubulin_qsar.load(tubulin_model_path)
        test_smiles_tubulin = [
            'COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O',  # A known active for tubulin
            'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin (negative control)
            'CCC(=O)Nc1ccc(cc1)C(C)C(=O)O', # Ibuprofen (another negative control)
        ]
        
        preds, uncs, interps = loaded_tubulin_qsar.predict(test_smiles_tubulin)
        
        if preds is not None:
            for i, (smiles, interp) in enumerate(zip(test_smiles_tubulin, interps)):
                print(f"\nMolecule {i+1} (Tubulin Prediction):")
                print(f"  SMILES: {smiles[:50]}...")
                print(f"  Predicted pIC50: {interp['pIC50']:.2f} ± {interp['uncertainty']:.2f}")
                print(f"  Predicted IC50: {interp['IC50_nM']:.1f} nM")
                print(f"  Activity: {interp['activity_level']}")
        else:
            print("Could not make predictions for tubulin test SMILES (possibly invalid SMILES provided or model not trained).")

    except FileNotFoundError:
        print(f"Error: Tubulin model not found at {tubulin_model_path}. Make sure it was trained and saved.")
    except ValueError as e:
        print(f"Error making predictions with Tubulin model: {e}")


    # Example of loading another model (e.g., AChE) and making predictions
    print("\n" + "="*50)
    print("DEMONSTRATING LOADING AND PREDICTION FOR ACHE")
    print("="*50)

    ache_model_path = os.path.join(models_dir, 'AChE_qsar_model.pkl')
    loaded_ache_qsar = GeneralQSARModel()
    try:
        loaded_ache_qsar.load(ache_model_path)
        test_smiles_ache = [
            'CC(=O)Oc1ccccc1C(=O)N(C)CC(C)CC', # Example of an AChE inhibitor
            'CNC(=O)Oc1ccc(cc1)C(C)(C)C',      # Carbaryl (known AChE inhibitor)
            'CC(=O)Oc1ccccc1C(=O)O'            # Aspirin (negative control)
        ]

        preds_ache, uncs_ache, interps_ache = loaded_ache_qsar.predict(test_smiles_ache)

        if preds_ache is not None:
            for i, (smiles, interp) in enumerate(zip(test_smiles_ache, interps_ache)):
                print(f"\nMolecule {i+1} (AChE Prediction):")
                print(f"  SMILES: {smiles[:50]}...")
                print(f"  Predicted pIC50: {interp['pIC50']:.2f} ± {interp['uncertainty']:.2f}")
                print(f"  Predicted IC50: {interp['IC50_nM']:.1f} nM")
                print(f"  Activity: {interp['activity_level']}")
        else:
            print("Could not make predictions for AChE test SMILES (possibly invalid SMILES provided or model not trained).")
            
    except FileNotFoundError:
        print(f"Error: AChE model not found at {ache_model_path}. Make sure it was trained and saved.")
    except ValueError as e:
        print(f"Error making predictions with AChE model: {e}")









# (drug) C:\Users\DELL\Downloads\drug>python qsar_training.py

# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: ACHE
# ======================================================================
# Loading dataset from AChE_dataset.csv...
# Dataset: 426 molecules
# Activity range (nM): 0.0 - 11600000.0
# pIC50 range: -7.06 - 2.05

# Featurizing molecules...
# Valid molecules: 426 / 426

# Training set: 340 | Test set: 86

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.912
# Test R²:  0.612
# Test RMSE: 1.158 pIC50 units
# Test MAE:  0.824 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\AChE_qsar_model.pkl

# Completed training for AChE.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: BREAST_CANCER_ANN
# ======================================================================
# Loading dataset from Breast_cancer_ANN_dataset.csv...
# Dataset: 20 molecules
# Activity range (nM): 310.0 - 250000.0
# pIC50 range: -5.40 - -2.49

# Featurizing molecules...
# Valid molecules: 20 / 20

# Training set: 16 | Test set: 4

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.732
# Test R²:  -0.050
# Test RMSE: 1.086 pIC50 units
# Test MAE:  1.062 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\Breast_cancer_ANN_qsar_model.pkl

# Completed training for Breast_cancer_ANN.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: BTK
# ======================================================================
# Loading dataset from BTK_dataset.csv...
# Dataset: 1 molecules
# Activity range (nM): 11.0 - 11.0
# pIC50 range: -1.04 - -1.04

# Featurizing molecules...
# Valid molecules: 1 / 1
# Warning: Not enough valid molecules (1) in BTK_dataset.csv for train-test split. Skipping training.
# Skipped saving model for BTK due to training issues (e.g., insufficient data).

# Completed training for BTK.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: ECOLI
# ======================================================================
# Loading dataset from Ecoli_dataset.csv...
# Dataset: 33 molecules
# Activity range (nM): 180.0 - 500000.0
# pIC50 range: -5.70 - -2.26

# Featurizing molecules...
# Valid molecules: 33 / 33

# Training set: 26 | Test set: 7

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.768
# Test R²:  0.574
# Test RMSE: 0.392 pIC50 units
# Test MAE:  0.328 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\Ecoli_qsar_model.pkl

# Completed training for Ecoli.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: FAK
# ======================================================================
# Loading dataset from FAK_dataset.csv...
# Dataset: 3 molecules
# Activity range (nM): 19500.0 - 117500.0
# pIC50 range: -5.07 - -4.29

# Featurizing molecules...
# Valid molecules: 3 / 3

# Training set: 2 | Test set: 1

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: -0.002
# Test R²:  nan
# Test RMSE: 0.431 pIC50 units
# Test MAE:  0.431 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\FAK_qsar_model.pkl

# Completed training for FAK.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: GLYT1
# ======================================================================
# Loading dataset from GlyT1_dataset.csv...
# Dataset: 6 molecules
# Activity range (nM): 7.6 - 95.3
# pIC50 range: -1.98 - -0.88

# Featurizing molecules...
# Valid molecules: 6 / 6

# Training set: 4 | Test set: 2

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: -0.001
# Test R²:  -3.101
# Test RMSE: 0.581 pIC50 units
# Test MAE:  0.505 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\GlyT1_qsar_model.pkl

# Completed training for GlyT1.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: MAOB
# ======================================================================
# Loading dataset from MAOB_dataset.csv...
# Dataset: 8419 molecules
# Activity range (nM): 0.0 - 58884365.5
# pIC50 range: -7.77 - 1.85

# Featurizing molecules...
# Valid molecules: 8419 / 8419

# Training set: 6735 | Test set: 1684

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.869
# Test R²:  0.722
# Test RMSE: 0.744 pIC50 units
# Test MAE:  0.476 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\MAOB_qsar_model.pkl

# Completed training for MAOB.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: SCAFFOLD_BASED_DISCOVERY
# ======================================================================
# Loading dataset from Scaffold_based_discovery_dataset.csv...
# Dataset: 322 molecules
# Activity range (nM): 7.0 - 50000.0
# pIC50 range: -4.70 - -0.85

# Featurizing molecules...
# Valid molecules: 322 / 322

# Training set: 257 | Test set: 65

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.929
# Test R²:  0.799
# Test RMSE: 0.325 pIC50 units
# Test MAE:  0.255 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\Scaffold_based_discovery_qsar_model.pkl

# Completed training for Scaffold_based_discovery.


# ======================================================================
# TRAINING QSAR MODEL FOR TARGET: TUBULIN
# ======================================================================
# Loading dataset from tubulin_dataset.csv...
# Dataset: 1398 molecules
# Activity range (nM): 0.1 - 40000000.0
# pIC50 range: -7.60 - 1.26

# Featurizing molecules...
# Valid molecules: 1398 / 1398

# Training set: 1118 | Test set: 280

# Training Random Forest model...

# ==================================================
# MODEL PERFORMANCE
# ==================================================
# Train R²: 0.733
# Test R²:  0.396
# Test RMSE: 0.681 pIC50 units
# Test MAE:  0.454 pIC50 units
# ==================================================

# Model and feature names saved to: qsar_models\tubulin_qsar_model.pkl

# Completed training for tubulin.


# ======================================================================
# ALL QSAR MODELS TRAINED AND SAVED (where possible).
# ======================================================================

# ==================================================
# DEMONSTRATING LOADING AND PREDICTION FOR TUBULIN
# ==================================================
# Model loaded from: qsar_models\tubulin_qsar_model.pkl (feature_names restored: No)

# Molecule 1 (Tubulin Prediction):
#   SMILES: COc1cc(C2c3cc4c(cc3CC3COC(=O)C32)OCO4)cc(OC)c1O...
#   Predicted pIC50: -3.87 ± 0.48
#   Predicted IC50: 7379.5 nM
#   Activity: Low

# Molecule 2 (Tubulin Prediction):
#   SMILES: CC(=O)Oc1ccccc1C(=O)O...
#   Predicted pIC50: -4.45 ± 0.54
#   Predicted IC50: 28011.3 nM
#   Activity: Low

# Molecule 3 (Tubulin Prediction):
#   SMILES: CCC(=O)Nc1ccc(cc1)C(C)C(=O)O...
#   Predicted pIC50: -4.26 ± 0.67
#   Predicted IC50: 18284.4 nM
#   Activity: Low

# ==================================================
# DEMONSTRATING LOADING AND PREDICTION FOR ACHE
# ==================================================
# Model loaded from: qsar_models\AChE_qsar_model.pkl (feature_names restored: No)

# Molecule 1 (AChE Prediction):
#   SMILES: CC(=O)Oc1ccccc1C(=O)N(C)CC(C)CC...
#   Predicted pIC50: -4.85 ± 0.88
#   Predicted IC50: 70841.6 nM
#   Activity: Low

# Molecule 2 (AChE Prediction):
#   SMILES: CNC(=O)Oc1ccc(cc1)C(C)(C)C...
#   Predicted pIC50: -4.59 ± 1.57
#   Predicted IC50: 38780.1 nM
#   Activity: Low

# Molecule 3 (AChE Prediction):
#   SMILES: CC(=O)Oc1ccccc1C(=O)O...
#   Predicted pIC50: -4.89 ± 1.47
#   Predicted IC50: 77535.9 nM
#   Activity: Low

