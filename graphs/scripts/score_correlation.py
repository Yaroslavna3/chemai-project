from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Configure paths
LLM_SCORES = 'results_detailed_abs.csv'
FEATURES = 'mol_features.csv'
DATAPATH = Path.cwd() / 'data'
IMGPATH = Path.cwd() / 'graphs/img'
LLM_PATH = DATAPATH / LLM_SCORES
FEATURES_PATH = DATAPATH / FEATURES


def main():
    if LLM_PATH.exists() and FEATURES_PATH.exists():
        df_llm = pd.read_csv(LLM_PATH)
        df_features = pd.read_csv(FEATURES_PATH)

        # Filter for smiles and iupac representations
        for representation in ['smiles', 'iupac']:
            df_llm_rep = df_llm[df_llm['representation'] == representation]
            
            if 'name' in df_llm.columns and 'latin_name' in df_features.columns:
                df = pd.merge(
                    df_features,
                    df_llm_rep[['name', 'score']],
                    left_on='latin_name',
                    right_on='name',
                    how='left'
                )
                df = df.rename(columns={'score': 'llm_score'})

                # Correlation analysis
                digital_features = df.select_dtypes(include=['float64', 'int64'])
                correlation_matrix = digital_features.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, fmt='.2f')
                plt.title(f'Correlation matrix for {representation} representation', fontsize=14)
                plt.tight_layout()
                plt.savefig(IMGPATH / f'correlation_matrix_{representation}.png', dpi=300, bbox_inches='tight')
                plt.show()

            else:
                print("Columns not found")
    else:
        print(f"Files {LLM_SCORES} and/or {FEATURES} not found at {DATAPATH}")   


if __name__ == "__main__":
    main()