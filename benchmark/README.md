## Installation

If you don't have SMILES dataset firstly **(else skip this step)**:
1. Install **conda environment** from the `lj_scratch_convert_env.yml`
```shell
conda env create -f lj_scratch_convert_env.yml

conda activate lj_scratch_convert
```

Then you can use the sample of scratcher `scratching.py` and latin name to SMILES converter `convert_to_smiles.py` (check [usage](#usage)).

2. Install **conda environment** from the `lj_get_features_env.yml`
```shell
conda env create -f lj_get_features_env.yml
```
3. Activate conda environment
```shell
conda activate lj_get_features
```
4. Download `sascorer.py` and `fpscores.pkl.gz` from [rdkit github repository](https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score) directly in this folder (chemai-project/benchmark). Or you can do it via wget
```shell
!wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py
```
```shell
!wget https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz
```

## Usage

1. Example of scratching latin compounds name is `scratching.py`. Use the `lj_scratch_convert` envitonment.
```shell
python scratching.py
```
The results will appear at `data/molecules_by_latin.csv`

2. To convert latin names to SMILES use `convert_to_smiles.py`. Use the `lj_scratch_convert` envitonment.
```shell
python convert_to_smiles.py
```
The results will appear at `data/smiles.csv`

3. To get molecule features from SMILES use `get_features.py`. Use the `lj_get_features` envitonment.

Before running this script you should put `glaxo_filters.csv` file with divider `;` into the `data` directory if you want to use Glaxo filters. Or you can run this script without glaxo filters file.

```shell
python get_smiles.py
```
The results will appear at `data/mol_features.csv`
