## Installation

1. Install **conda environment** from the `benchmark_environment.yml`
```shell
conda env create -f environment.yml
```
2. Activate conda environment
```shell
conda activate llm_judge
```

## Usage

1. To scratch latin compounds name from `www.vidal.ru` use `scratching.py`
```shell
python scratching.py
```
The results will appear at `data/molecules_by_id.csv`

2. To convert latin names to SMILES use `convert_to_smiles.py`
```shell
python convert_to_smiles.py
```
The results will appear at `data/smiles.csv`

3. To get molecule features from SMILES use `get_features.py`

Before running this script you should put `glaxo_filters.csv` file with divider `;` into the `data` directory if you want to use Glaxo filters. Or you can run this script without glaxo filters file.

```shell
python get_smiles.py
```
The results will appear at `data/mol_features.csv`