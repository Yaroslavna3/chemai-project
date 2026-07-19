import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm.auto import tqdm


DEFAULT_INPUT = Path("data") / "molecules_by_latin.csv"
DEFAULT_OUTPUT = Path("data") / "smiles.csv"


def get_smiles_from_pubchem(molecule_name: str) -> Optional[str]:
    try:
        compounds = pcp.get_compounds(molecule_name, "name")
        if compounds:
            return compounds[0].connectivity_smiles
        print(f"No compounds found for '{molecule_name}'")
        return None
    except Exception as exc:
        print(f"An error occurred while querying PubChem for '{molecule_name}': {exc}")
        return None


def get_iupac(smiles: str) -> str:
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds:
            return compounds[0].iupac_name
        return "N/F"
    except Exception as exc:
        return f"Error: {exc}"


def get_heavy_atom_count(mol) -> int:
    if mol:
        return mol.GetNumHeavyAtoms()
    return 0


def convert_names_to_smiles(
    input_path: Path,
    output_path: Path,
    adjust_filters: bool,
    max_heavy_atoms: int,
    max_mw: float,
    delay_seconds: float,
) -> None:
    df_input = pd.read_csv(input_path)
    if "latin_name" not in df_input.columns:
        raise ValueError(f"Input file must contain a 'latin_name' column: {input_path}")

    results = []
    for _, row in tqdm(df_input.iterrows(), total=df_input.shape[0], desc="Filtering molecules"):
        name = row["latin_name"]
        pubchem_smiles = get_smiles_from_pubchem(name)
        if not pubchem_smiles:
            time.sleep(delay_seconds)
            continue

        mol = Chem.MolFromSmiles(pubchem_smiles)
        if not mol:
            time.sleep(delay_seconds)
            continue

        mw = Descriptors.MolWt(mol)
        heavy_atoms = get_heavy_atom_count(mol)
        if adjust_filters and (mw >= max_mw or heavy_atoms >= max_heavy_atoms):
            time.sleep(delay_seconds)
            continue

        results.append(
            {
                "latin_name": name,
                "smiles": Chem.MolToSmiles(mol, canonical=True),
                "non_canonical_smiles": Chem.MolToSmiles(mol, canonical=False, doRandom=True),
                "iupac": get_iupac(pubchem_smiles),
            }
        )
        time.sleep(delay_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"\nProcessed {len(df_input)} molecules. Kept {len(result_df)} after filtering.")
    print(f"Saved at {output_path}")
    print(result_df.head())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Latin molecule names to SMILES via PubChem.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adjust-filters", action="store_true")
    parser.add_argument("--max-heavy-atoms", type=int, default=50)
    parser.add_argument("--max-mw", type=float, default=900)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_names_to_smiles(
        input_path=args.input,
        output_path=args.output,
        adjust_filters=args.adjust_filters,
        max_heavy_atoms=args.max_heavy_atoms,
        max_mw=args.max_mw,
        delay_seconds=args.delay_seconds,
    )


if __name__ == "__main__":
    main()
