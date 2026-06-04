import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import sascorer

# Configuration
DATAPATH = Path.cwd() / 'data'
INPUT_FILE = 'smiles.csv'
OUTPUT_FILE = 'mol_features.csv'
GLAXO_PATH = DATAPATH / 'glaxo_filters.csv' # default divider is ';'

# Initialize BRENK and PAINS Filter Catalogs
brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_catalog = FilterCatalog.FilterCatalog(brenk_params)

pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_catalog = FilterCatalog.FilterCatalog(pains_params)


def filter_mols(smiles, catalog):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    matches = catalog.GetMatches(mol)
    if matches:
        return True, [match.GetDescription() for match in matches]
    return False, []


# Glaxo
if GLAXO_PATH.exists():
    try:
        glaxo_filters = pd.read_csv(GLAXO_PATH, sep=';')
        glaxo_catalog = [(row['smarts'], Chem.MolFromSmarts(row['smarts'])) for _, row in glaxo_filters.iterrows()]
        print(f"Glaxo filters loaded from {GLAXO_PATH}")
    except Exception as e:
        print(f"Error loading Glaxo filters: {e}")
        glaxo_catalog = None
else:
    glaxo_catalog = None
    print(f"Glaxo filters file not found: {GLAXO_PATH}. Skipping Glaxo filter.")


def calc_glaxo(smiles, catalog):
    if catalog is None:
        return None, [] # Skip Glaxo filter
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    found_structs = []
    for smarts_str, pattern in catalog:
        if pattern and mol.HasSubstructMatch(pattern):
            found_structs.append(smarts_str)
    if found_structs:
        return True, found_structs
    return False, found_structs


def process_smiles(brenk_catalog, pains_catalog, glaxo_catalog):
    df = pd.read_csv(DATAPATH / INPUT_FILE)
    results = []

    use_glaxo = glaxo_catalog is not None

    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)

            # Lipinski
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1

            # Filters
            brenk_alert, brenk_matches = filter_mols(smiles, brenk_catalog)
            pains_alert, pains_matches = filter_mols(smiles, pains_catalog)
            if use_glaxo:
                glaxo_alert, glaxo_matches = calc_glaxo(smiles, glaxo_catalog)

            res = {
                'smiles': smiles,
                'QED': round(QED.qed(mol), 2),
                'SA': round(sascorer.calculateScore(mol), 2),
                'MW': round(mw, 2),
                'LogP': round(logp, 2),
                'Lipinski': violations <= 1,
                'Lipinski_violations_0': violations == 0,
                'BRENK': brenk_alert,
                'BRENK_matches': ", ".join(brenk_matches) if brenk_matches != [] else "",
                'PAINS': pains_alert,
                'PAINS_matches': ", ".join(pains_matches) if pains_matches != [] else "",
            }

            if use_glaxo:
                res['Glaxo'] = glaxo_alert
                res['Glaxo_matches'] = ", ".join(glaxo_matches) if glaxo_matches != [] else ""
            
            results.append(res)
        else:
            results.append({'smiles': smiles, 'error': 'Invalid SMILES'})

    output_df = pd.DataFrame(results)
    output_df.to_csv(DATAPATH / OUTPUT_FILE, index=False)

    print(output_df.head())
    print(f'Results saved to {DATAPATH / OUTPUT_FILE}')


# To run this, ensure input file is uploaded
process_smiles(BRENK_catalog, PAINS_catalog, glaxo_catalog)
