
def collect_features(feature_path: object) -> dict:
    """

    Args:
        feature_path: pathlib.Path object denoting file location of the features

    Returns:
        Dictionary {feature_group_name: pd.DataFrame}
    """
    import glob
    import os
    import pandas as pd

    features = {}

    feature_paths = glob.glob(str(feature_path))
    feature_names = [os.path.split(path)[1].rstrip('.csv') for path in feature_paths]

    for path, name in zip(feature_paths, feature_names):
        features[name] = pd.read_csv(path, index_col=[0])

    return features


def classification_model_performance(X, y, inner_cv, outer_cv):
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.linear_model import RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import make_scorer, f1_score
    from sklearn.metrics import accuracy_score, confusion_matrix

    f1_macro = make_scorer(f1_score, average='macro')

    def cm00(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

    def cm10(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

    def cm20(y_true, y_pred): return confusion_matrix(y_true, y_pred)[2, 0]

    def cm01(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

    def cm11(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

    def cm21(y_true, y_pred): return confusion_matrix(y_true, y_pred)[2, 1]

    def cm02(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 2]

    def cm12(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 2]

    def cm22(y_true, y_pred): return confusion_matrix(y_true, y_pred)[2, 2]

    def lowF1(y_true, y_pred): return f1_score(y_true, y_pred, average=None)[0]

    def medF1(y_true, y_pred): return f1_score(y_true, y_pred, average=None)[1]

    def highF1(y_true, y_pred): return f1_score(y_true, y_pred, average=None)[2]

    scoring = {  # top row of cm matrix
        'cm00': make_scorer(cm00), 'cm10': make_scorer(cm10),
        'cm20': make_scorer(cm20),
        # mid row
        'cm01': make_scorer(cm01), 'cm11': make_scorer(cm11),
        'cm21': make_scorer(cm21),
        # bottom row
        'cm02': make_scorer(cm02), 'cm12': make_scorer(cm12),
        'cm22': make_scorer(cm22),

        'f1_macro': f1_macro, 'accuracy': make_scorer(accuracy_score),

        'low_f1': make_scorer(lowF1), 'med_f1': make_scorer(medF1),
        'high_f1': make_scorer(highF1)}

    models_and_parameters = {
        'knn': (  # classifier object
            Pipeline([
                ('std_scale', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            # parameter grid
            {'knn__n_neighbors': [1, 3, 5, 7, 9, 11],
             'knn__p': [1, 2]}),  # Manhattan or euclidean distance

        # clf
        'SVC': (
            Pipeline([
                ('std_scale', StandardScaler()),
                ('svc', SVC(gamma='auto'))
            ]),
            # param grid
            {'svc__kernel': ['linear', 'rbf'],
             'svc__C': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1]}),

        'ridge': (
            Pipeline([
                ('std_scale', StandardScaler()),
                ('ridge', RidgeClassifier())
            ]),

            {'ridge__alpha': [0.01, 0.1, 0.3, 0.5, 1]}),

        'naive': (
            Pipeline([
                ('std_scale', StandardScaler()),
                ('naive', GaussianNB())
            ]),

            {'naive__var_smoothing': [1e-09]})}

    avg_outer_fold_scores = dict()

    for name, (model, params) in models_and_parameters.items():
        # Inner folds - hyperparameter selection takes place here.
        # When used within another cross validation scheme,
        # this object searches for the best hyperparameters within
        # the training fold provided to it.

        grid_cv = GridSearchCV(
            estimator=model, param_grid=params,
            cv=inner_cv, scoring=f1_macro,
            n_jobs=1)

        # Outer folds - model performance assessed here
        # This object provides the training folds to the grid_cv estimator
        # and estimates performance using the unseen test fold.
        # The purpose here is to select which algorithm we are going to use.
        scores_across_outer_folds = cross_validate(
            grid_cv,
            X, y,
            cv=outer_cv,
            scoring=scoring,  # ['f1_macro', 'accuracy'],
            n_jobs=-1,
            return_train_score=True,
            return_estimator=True)

        # get the F1 macro score across each of outer_cv's folds
        avg_outer_fold_scores[name] = scores_across_outer_folds

    return avg_outer_fold_scores


def get_clf_results(features, labels, inner_cv, outer_cv):
    clf_results = {}

    for feature in features.keys():

        X = features[feature]

        print(f" --- \n --- \n Feature Group: {feature}"
              " --- \t --- \t ---")

        for label in labels:
            print(f' --- \n Metrics for the label of {label} \n ---')

            y = labels.loc[:, label]

            # Quality Assurance: Ensure y rows match the feature matrix idx
            assert len(y) == len(X), 'failed: length different'
            assert all(X.index == y.index), 'failed: non-matching indices'

            results = classification_model_performance(X=X,
                                                       y=y,
                                                       inner_cv=inner_cv,
                                                       outer_cv=outer_cv
                                                       )
            clf_results[feature + '_' + label] = results

    return clf_results


def clf_results_to_df(clf_results, filter_results=False):
    """
    Helper function:
        Input = classifier results in 4-deep nested dictionary.

        Level 0:
            Keys = feature_group, label combinations (60 keys)

            Level 1:
                keys = algorithms
                values = cross_validate output

                Level 2:
                    key = 'estimator'
                    values = Outer Loop GridSearchCV estimators (x5)

                    Level 3:
                        key = 'best_params'
                        value = dict{param: value}

    Returns:
        Output = Pandas DataFrame where:
            columns = ['Mean_Accuracy', # Float
                       'Std_Accuracy', # Float
                       'Mean_F1Macro', # Float
                       'Std_F1Macro', # Float
                       'Best Params']
            rows = (Label, Feature Group) Combinations e.g. ('AOI', 'Agreeableness')
    """
    import numpy as np
    import pandas as pd

    dfs = []
    all_best_params = {}

    for key, value in clf_results.items():

        feature_group = '_'.join(key.split('_')[:-1])
        label = key.split('_')[-1]

        for algorithm, algorithm_results in value.items():

            # accuracy metric
            accuracy_mean = np.mean(algorithm_results['test_accuracy'])
            accuracy_std = np.std(algorithm_results['test_accuracy'])
            # f1 score metric
            f1_mean = np.mean(algorithm_results['test_f1_macro'])
            f1_std = np.std(algorithm_results['test_f1_macro'])

            # best estimators
            best_params = []
            for model in algorithm_results['estimator']:
                best_params.append(model.best_params_)

            all_best_params[label + '_' +
                            feature_group + '_' + algorithm] = best_params

            df = pd.DataFrame({
                'Label': label,
                'Feature_Group': feature_group,
                'Algorithm': algorithm,
                'Accuracy_Mean': accuracy_mean,
                'Accuracy_Std': accuracy_std,
                'F1-Macro_Mean': f1_mean,
                'F1-Macro_Std': f1_std},
                index=[key])

            dfs.append(df)

    clf_results_df = pd.concat([df for df in dfs])

    # set a hierarchical index from the column values
    clf_results_df.index = pd.MultiIndex.from_tuples(
        zip(clf_results_df.loc[:, 'Label'],
            clf_results_df.loc[:, 'Feature_Group']))

    # if required, filter F1 macro scores by criterion
    if filter_results:

        baseline_accuracy = {
            'Openness': .400,
            'Conscientiousness': .361,
            'Extroversion': .406,
            'Agreeableness': .389,
            'Neuroticism': .350
        }

        # Remove results where F1-Macro is less than .33 (chance)
        clf_results_df = clf_results_df[
            clf_results_df.loc[:, 'F1-Macro_Mean'] > .33]

        dfs = []
        # Remove results where accuracy < baseline accuracy (per trait)
        for trait, baseline in baseline_accuracy.items():
            dfs.append(clf_results_df.loc[trait].loc[
                           clf_results_df.loc[trait, 'Accuracy_Mean'] > baseline])

        clf_results_df = pd.concat(dfs)

        for trait, baseline in baseline_accuracy.items():
            clf_results_df.loc[clf_results_df['Label'] == trait, 'Baseline_Accuracy'] = baseline

    return clf_results_df, all_best_params


def label_pvalues(clf_score, null_distribution):
    import numpy as np
    # number scores better than model
    n_perms_above_model = np.sum(null_distribution >= clf_score)

    # p = |N_obvs greater than model + 1 | divided by N_Permutations + 1
    p_value = round((n_perms_above_model + 1) / (float(len(null_distribution) + 1)), 3)

    return p_value


def create_permutations(vector, permutations=200000):
    import numpy as np

    perms = []

    for i in range(permutations):
        perms.append(np.random.permutation(vector))

    return perms


def fetch_null_in_parallel(truth, evaluation_metric, n_cores):
    from joblib import Parallel, delayed
    import numpy as np

    permutations = create_permutations(truth)

    perm_score = Parallel(n_jobs=n_cores)(
        delayed(evaluation_metric)(y_true=truth, y_pred=perm, average='macro') for perm in permutations)

    return np.array(perm_score)


def find_pvalues(results, true_labels, n_cores):
    from sklearn.metrics import f1_score
    import pandas as pd

    dfs = []
    for label, df in results.groupby('Label'):

        truth = true_labels.loc[:, label]

        null_distribution = fetch_null_in_parallel(truth=truth,
                                                   evaluation_metric=f1_score,
                                                   n_cores=n_cores)

        print(label)

        p_values = []

        for i in df.itertuples():
            p_values.append(label_pvalues(i.Accuracy_Mean, null_distribution))

        df_copy = df.copy()
        df_copy.loc[:, 'pvalues_f1'] = p_values
        dfs.append(df_copy)

    return pd.concat(dfs)


def correct_pvalues(results_df):
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    dfs = []
    for label, obs in results_df.groupby('Label'):
        df = obs.copy()

        correction_data = multipletests(pvals=obs.loc[:, 'pvalues_f1'], method='fdr_bh')

        df.loc[:, 'reject_null'] = correction_data[0]
        df.loc[:, 'pvalue_f1_bh'] = correction_data[1]

        dfs.append(df)

    return pd.concat(dfs)
