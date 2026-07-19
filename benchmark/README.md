## Installation

If you do not have a SMILES dataset yet, start from steps 1-2. Otherwise, skip
directly to feature calculation.

1. Install the conda environment for scraping and PubChem conversion:
```shell
conda env create -f lj_scratch_convert_env.yml

conda activate lj_scratch_convert
```

Then you can use `scratching.py` and `convert_to_smiles.py` (see
[Usage](#usage)).

2. Install the conda environment for feature calculation:
```shell
conda env create -f lj_get_features_env.yml

conda activate lj_get_features
```

3. Download `sascorer.py` and `fpscores.pkl.gz` from the
[RDKit SA_Score repository](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score)
directly into this folder (`chemai-project/benchmark`). Or do it via wget:
```shell
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
```
```shell
wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
```

## Usage

Run commands from the repository root unless you pass explicit paths.

1. Scrape Latin compound names with `scratching.py`. Use the
`lj_scratch_convert` environment.
```shell
python benchmark/scratching.py --base-url "https://example.org/molecule/"
```
The results will appear at `data/molecules_by_latin.csv` by default. The
`--base-url` value must be the URL prefix before the numeric molecule ID.

2. Convert Latin names to SMILES with `convert_to_smiles.py`. Use the
`lj_scratch_convert` environment.
```shell
python benchmark/convert_to_smiles.py
```
The results will appear at `data/smiles.csv`

3. Calculate molecule features from SMILES with `get_features.py`. Use the
`lj_get_features` environment.

Before running this script, put `glaxo_filters.csv` with `;` as the separator
into the `data` directory if you want to use Glaxo filters. The script can also
run without this file.

```shell
python benchmark/get_features.py
```
The results will appear at `data/mol_features.csv`
