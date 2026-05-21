import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import sascorer

INPUT_FILE = 'molecules_filtered.csv'
OUTPUT_FILE = 'mol_features.csv'
DATAPATH = Path.cwd() / 'data'

# Initialize Filter Catalogs
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
glaxo_filters = pd.read_csv(DATAPATH / 'glaxo_filters.csv', sep=';')
glaxo_catalog = [(row['smarts'], Chem.MolFromSmarts(row['smarts'])) for _, row in glaxo_filters.iterrows()]


def calc_glaxo(smiles, catalog):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, []
    found_structs = []
    for smarts_str, pattern in catalog:
        if pattern and mol.HasSubstructMatch(pattern):
            found_structs.append(smarts_str)
    if found_structs:
        return True, found_structs
    return False, found_structs


def process_smiles(input_file, output_file, brenk_catalog, pains_catalog, glaxo_catalog):
    df = pd.read_csv(DATAPATH / input_file)
    results = []

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
            glaxo_alert, glaxo_matches = calc_glaxo(smiles, glaxo_catalog)

            res = {
                'smiles': smiles,
                'QED': round(QED.qed(mol), 2),
                'SA': round(sascorer.calculateScore(mol), 2),
                'MW': round(mw, 2),
                'LogP': round(logp, 2),
                'Lipinski': violations <= 1,
                'BRENK': brenk_alert,
                'PAINS': pains_alert,
                'Glaxo': glaxo_alert,
                'Lipinski_violations_0': violations == 0,
                'BRENK_matches': ", ".join(brenk_matches) if brenk_matches != [] else "",
                'PAINS_matches': ", ".join(pains_matches) if pains_matches != [] else "",
                'Glaxo_matches': ", ".join(glaxo_matches) if glaxo_matches != [] else "",
            }
            results.append(res)
        else:
            results.append({'smiles': smiles, 'error': 'Invalid SMILES'})

    output_df = pd.DataFrame(results)
    output_df.to_csv(DATAPATH / output_file, index=False)

    print(output_df.head())
    print(f'Results saved to {output_file}')


# To run this, ensure input file is uploaded
process_smiles(INPUT_FILE, OUTPUT_FILE, BRENK_catalog, PAINS_catalog, glaxo_catalog)
