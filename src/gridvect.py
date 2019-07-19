from pprint import pprint
from time import time
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class MultiColVectoriser(TransformerMixin, BaseEstimator):
    def __init__(self, cols, vect, ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None):
        self.cols = cols
        self.vect = vect
        self.vect = vect
        self.vect.max_df, self.vect.min_df, self.vect.ngram_range, self.max_features = max_df, min_df, ngram_range, max_features
        self.mk_ct = self._make_ct()

    def _make_ct(self):
        return make_column_transformer(
            *[(self.vect, col) for col in self.cols]
        )

    @property
    def max_df(self):
        return self.vect.max_df

    @max_df.setter
    def max_df(self, max_df):
        self.vect.max_df = max_df
        self.mk_ct = self._make_ct()

    @property
    def min_df(self):
        return self.vect.min_df

    @min_df.setter
    def min_df(self, min_df):
        self.vect.min_df = min_df
        self.mk_ct = self._make_ct()

    @property
    def ngram_range(self):
        return self.vect.ngram_range

    @ngram_range.setter
    def ngram_range(self, ngram_range):
        self.vect.ngram_range = ngram_range
        self.mk_ct = self._make_ct()

    @property
    def max_features(self):
        return self.vect.max_features

    @max_features.setter
    def max_features(self, max_features):
        self.vect.max_features = max_features
        self.mk_ct = self._make_ct()

    def transform(self, x, **transform_params):
        return self.mk_ct.transform(x)

    def fit(self, x, y=None, **fit_params):
        return self.mk_ct.fit(x, y)


def grid_vect(clf, parameters_clf, x_train, x_test, y_train, y_test, text_cols, parameters_text=None, vect=None):
    # Convert Y-axis to numpy arrays with a specified type, as SMOTE throws and error saying it is of unknown type when converting to numpy array.
    y_train = np.asarray(y_train, dtype='int64')
    y_test = np.asarray(y_test, dtype='int64')

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
