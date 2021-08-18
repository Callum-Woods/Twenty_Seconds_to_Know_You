import numpy as np
import pandas as pd
import scipy.stats

personal_attributes = pd.read_csv(
    "Data/Labels/personal_attributes.csv", index_col=0
)
personality_scores = pd.read_csv("Data/Labels/big5_scores.csv", index_col=0)

joined = personality_scores.join(
    personal_attributes.loc[personality_scores.index, :]
)

coeffmat = np.zeros((joined.shape[1], joined.shape[1]))
pvalmat = coeffmat.copy()

for i, col in enumerate(joined):

    valid_indices = joined[col].dropna().index

    for j, col_ in enumerate(joined):

        x = joined.loc[:, col]
        y = joined.loc[:, col_]

        missing_indices = np.logical_or(np.isnan(x), np.isnan(y))

        corrtest = scipy.stats.pearsonr(
            x.loc[~missing_indices], y.loc[~missing_indices]
        )

        coeffmat[i, j] = corrtest[0]
        pvalmat[i, j] = corrtest[1]

coeff_df = pd.DataFrame(coeffmat, columns=joined.columns, index=joined.columns)
pvalue_df = pd.DataFrame(pvalmat, columns=joined.columns, index=joined.columns)

sig_results = (
    coeff_df[pvalue_df < 0.05]
    .loc[["sex", "age"], [col for col in coeff_df.columns if col not in ["sex", "age"]]]
    .round(3)
)


cols = []
for col in coeff_df.columns:
    cols.append(list(zip(coeff_df.loc[:, col].round(3), pvalue_df.loc[:, col].round(3))))

full_results = pd.DataFrame(cols, index=coeff_df.index, columns=coeff_df.columns)
