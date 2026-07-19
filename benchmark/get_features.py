import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
import sascorer
from RAscore import RAscore_XGB
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams


DEFAULT_INPUT = Path("data") / "smiles.csv"
DEFAULT_OUTPUT = Path("data") / "mol_features.csv"
DEFAULT_GLAXO = Path("data") / "glaxo_filters.csv"


def make_filter_catalog(catalog_name) -> FilterCatalog.FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(catalog_name)
    return FilterCatalog.FilterCatalog(params)


def filter_mols(smiles: str, catalog: FilterCatalog.FilterCatalog) -> Tuple[Optional[bool], List[str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    matches = catalog.GetMatches(mol)
    if matches:
        return True, [match.GetDescription() for match in matches]
    return False, []


def load_glaxo_catalog(glaxo_path: Path) -> Optional[List[Tuple[str, Any]]]:
    if not glaxo_path.exists():
        print(f"Glaxo filters file not found: {glaxo_path}. Skipping Glaxo filter.")
        return None

    glaxo_filters = pd.read_csv(glaxo_path, sep=";")
    if "smarts" not in glaxo_filters.columns:
        raise ValueError(f"Glaxo filters file must contain a 'smarts' column: {glaxo_path}")

    catalog = []
    for _, row in glaxo_filters.iterrows():
        pattern = Chem.MolFromSmarts(row["smarts"])
        if pattern is not None:
            catalog.append((row["smarts"], pattern))
    print(f"Loaded {len(catalog)} Glaxo filters from {glaxo_path}")
    return catalog


def calc_glaxo(smiles: str, catalog: Optional[List[Tuple[str, Any]]]) -> Tuple[Optional[bool], List[str]]:
    if catalog is None:
        return None, []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []

    found_structs = [smarts for smarts, pattern in catalog if mol.HasSubstructMatch(pattern)]
    return bool(found_structs), found_structs


def lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def process_smiles(input_path: Path, output_path: Path, glaxo_path: Path) -> None:
    df = pd.read_csv(input_path)
    required_columns = {"latin_name", "smiles", "non_canonical_smiles", "iupac"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input file is missing columns {sorted(missing_columns)}: {input_path}")

    brenk_catalog = make_filter_catalog(FilterCatalogParams.FilterCatalogs.BRENK)
    pains_catalog = make_filter_catalog(FilterCatalogParams.FilterCatalogs.PAINS)
    glaxo_catalog = load_glaxo_catalog(glaxo_path)
    rascore_scorer = RAscore_XGB.RAScorerXGB()

    results = []
    for _, row in df.iterrows():
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({"smiles": smiles, "error": "Invalid SMILES"})
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        violations = lipinski_violations(mw, logp, hbd, hba)

        brenk_alert, brenk_matches = filter_mols(smiles, brenk_catalog)
        pains_alert, pains_matches = filter_mols(smiles, pains_catalog)
        glaxo_alert, glaxo_matches = calc_glaxo(smiles, glaxo_catalog)

        results.append(
            {
                "latin_name": row["latin_name"],
                "smiles": smiles,
                "non_canonical_smiles": row["non_canonical_smiles"],
                "iupac": row["iupac"],
                "qed": round(QED.qed(mol), 2),
                "sa": round(sascorer.calculateScore(mol), 2),
                "mw": round(mw, 2),
                "logp": round(logp, 2),
                "rascore": rascore_scorer.predict(smiles),
                "lipinski": violations <= 1,
                "lipinski_violations_0": violations == 0,
                "brenk": brenk_alert,
                "brenk_matches": ", ".join(brenk_matches),
                "pains": pains_alert,
                "pains_matches": ", ".join(pains_matches),
                "glaxo": glaxo_alert,
                "glaxo_matches": ", ".join(glaxo_matches),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(output_df.head())
    print(f"Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate molecular features from SMILES.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--glaxo", type=Path, default=DEFAULT_GLAXO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_smiles(input_path=args.input, output_path=args.output, glaxo_path=args.glaxo)


if __name__ == "__main__":
    main()
