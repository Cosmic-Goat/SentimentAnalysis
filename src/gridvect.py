from pprint import pprint
from time import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class MultiColVectoriser(TransformerMixin, BaseEstimator):
    def __init__(self, cols, vect, ngram_range=(1, 1), max_df=1.0, min_df=1):
        self.cols = cols
        self.vect = vect
        self.vect = vect
        self.vect.max_df, self.vect.min_df, self.vect.ngram_range = max_df, min_df, ngram_range
        self.mk_ct = make_column_transformer(
            *[(self.vect, col) for col in self.cols]
        )

    @property
    def max_df(self):
        return self.vect.max_df

    @max_df.setter
    def max_df(self, max_df):
        self.vect.max_df = max_df

    @property
    def min_df(self):
        return self.vect.min_df

    @min_df.setter
    def min_df(self, min_df):
        self.vect.min_df = min_df

    @property
    def ngram_range(self):
        return self.vect.ngram_range

    @ngram_range.setter
    def ngram_range(self, ngram_range):
        self.vect.ngram_range = ngram_range

    def transform(self, x, **transform_params):
        return self.mk_ct.transform(x)

    def fit(self, x, y=None, **fit_params):
        return self.mk_ct.fit(x, y)


def grid_vect(clf, parameters_clf, x_train, x_test, y_train, y_test, text_cols, parameters_text=None, vect=None):

    pipeline = Pipeline([
        ('mv', MultiColVectoriser(text_cols, vect)),
        ('smote', SMOTE()),
        ('clf', clf)
    ])

    print(pipeline.get_params().keys)

    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)
    # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(x_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(x_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(x_test)))

    return grid_search
