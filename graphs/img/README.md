# Correlation matrices for Qwen3-14B absolute scoring

This folder contains correlation matrices for the full `absolute_score` benchmark evaluated with `qwen/qwen3-14b` under the strict prompt. The matrices compare the LLM score with classical molecular descriptors for the SMILES and IUPAC representations.

Brief conclusion: the LLM score shows only weak correlations with individual scalar descriptors. This suggests that the Qwen3-14B reward is not simply reproducing QED, SA, molecular weight, LogP, or RAscore, but captures a broader medicinal-chemistry signal that combines several molecular features.
