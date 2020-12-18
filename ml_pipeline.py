# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import KFold
from sklearn.base import clone
from custom_modules import collect_features,\
    get_clf_results, clf_results_to_df, find_pvalues, correct_pvalues

# SETUP: Adjust according to your system
n_cores = 6

# Load in features and labels
p = Path.cwd()
feature_dir = Path(p, 'Data', 'Features', '*.csv')
label_path = Path(p, 'Data', 'Labels', 'big5_scores.csv')

# features
features = collect_features(feature_dir)

# Labels
raw_labels = pd.read_csv(str(label_path), index_col=[0])

# Form the categories
# 0 = low
# 1 = Medium
# 2 = High
labels = raw_labels.copy()
split = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
for col in labels:
    labels.loc[:, col] = clone(split).fit_transform(labels.loc[:, col].values.reshape(-1, 1))

def get_support(df):

    freqs = []
    for col in df.columns:
        freqs.append(pd.Series(pd.value_counts(df.loc[:, col]), name=col))
    return pd.concat(freqs, axis=1)

support = get_support(labels)

support.to_csv('Output/Support_per_Category.csv')

# Nested cross-validation scheme
outer_cv, inner_cv = KFold(5), KFold(5)

classifier_results = get_clf_results(features=features,
                                     labels=labels,
                                     inner_cv=inner_cv,
                                     outer_cv=outer_cv)

with open('Output/CrossValOutput.pkl', 'wb') as file:
    pickle.dump(classifier_results, file)

all_results_df, all_params = clf_results_to_df(classifier_results, filter_results=False)
filtered_results_df, filtered_params = clf_results_to_df(classifier_results, filter_results=True)

# export the results dataframe
all_results_df.to_csv('Output/all_crossval_results.csv')

# export parameters selected as pickled dictionary
with open('Output/all_params.pkl', 'wb') as file:
    pickle.dump(all_params, file)

filtered_results_df.index = pd.MultiIndex.from_tuples(
    zip(filtered_results_df.loc[:, 'Label'],
        filtered_results_df.loc[:, 'Algorithm']))

def get_significance(filtered_results):
    # Which of these models are significantly better than chance?
    results_copy = filtered_results_df.copy(deep=True)
    results_copy.reset_index(inplace=True, drop=True)

    final_results_df = find_pvalues(results=results_copy,
                                    true_labels=labels.copy(),
                                    n_cores=n_cores)

    final_results_df_corrected = correct_pvalues(final_results_df)

    return final_results_df_corrected

results_with_significance = get_significance(filtered_results=filtered_results_df)

results_with_significance.to_csv('Output/above_chance_models_with_pvalues.csv')

significant_results = results_with_significance[results_with_significance.reject_null == True]

# Get the per-class F1 scores and Accuracy for Significant Models
def get_f1_by_class(result_dict, key, algorithm):
    f1_keys = ['test_low_f1', 'test_med_f1', 'test_high_f1']

    scores = [result_dict[key][algorithm][category] for category in f1_keys]

    return pd.DataFrame((np.mean(scores, axis=1), np.std(scores, axis=1)),
                        columns=['Low', 'Medium', 'High'],
                        index=['Mean', 'Std'])


openness_ems_svc = get_f1_by_class(classifier_results,
                                   'ems_viewing_info_Openness',
                                    'SVC')

extroversion_ems_ridge = get_f1_by_class(classifier_results,
                                         'ems_viewing_info_Extroversion',
                                         'ridge')
extroversion_ems_svc = get_f1_by_class(classifier_results,
                                       'ems_viewing_info_Extroversion',
                                       'SVC')

consc_aoi_ridge = get_f1_by_class(classifier_results,
                            'aoi_Conscientiousness',
                            'ridge')

consc_aoi_prop_ridge = get_f1_by_class(classifier_results,
                                       'aoi_prop_Conscientiousness',
                                       'ridge')

consc_page_info_svc = get_f1_by_class(classifier_results,
                                      'page_content_info_Conscientiousness',
                                      'ridge')