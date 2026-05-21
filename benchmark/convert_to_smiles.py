import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import time
from tqdm.auto import tqdm
from pathlib import Path


def get_smiles_from_pubchem(molecule_name):
    try:
        # Search for compounds by name. The 'compounds' method returns a list of matching compounds.
        # Take the first result as the most relevant one.
        compounds = pcp.get_compounds(molecule_name, 'name')
        if compounds:
            # If compounds are found, return the canonical SMILES of the first one.
            return compounds[0].connectivity_smiles
        else:
            return None
    except Exception as e:
        print(f"An error occurred while querying PubChem for '{molecule_name}': {e}")
        return None


def get_heavy_atom_count(mol):
    """Returns the number of heavy (non-hydrogen) atoms in a molecule."""
    if mol:
        return mol.GetNumHeavyAtoms()
    return 0


# Configuration
datapath = Path.cwd() / 'data'
input_file = datapath / 'molecules_by_id.csv'
output_file = datapath / 'molecules_filtered.csv'
MAX_HEAVY_ATOMS = 50  # Adjust this threshold to filter polymers/large molecules
MAX_MW = 900          # Molecular Weight threshold

try:
    df_input = pd.read_csv(input_file)
    results = []

    for index, row in tqdm(df_input.iterrows(), total=df_input.shape[0], desc="Filtering molecules"):
        name = row['latin_name']
        pubchem_smiles = get_smiles_from_pubchem(name)

        if pubchem_smiles:
            mol = Chem.MolFromSmiles(pubchem_smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                heavy_atoms = get_heavy_atom_count(mol)

                # Filter logic: Must be under MW limit AND Heavy Atom limit
                if mw < MAX_MW and heavy_atoms < MAX_HEAVY_ATOMS:
                    results.append({
                        'latin_name': name,
                        'smiles': Chem.MolToSmiles(mol, canonical=True),
                        'non_canonical_smiles': Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                    })

        time.sleep(0.2)  # Rate limiting

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"\nProcessed {len(df_input)} molecules. Kept {len(result_df)} after filtering.")
    print(result_df.head())

except Exception as e:
    print(f"Error: {e}")