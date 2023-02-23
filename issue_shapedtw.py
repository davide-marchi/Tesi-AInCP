from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from importlib import import_module as imp
import numpy as np

X_frag = [[0, 1, 2, 3, 5, 6, 8], [0, 1, 2, 3, 5, 6, 9], [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 5, 6, 7]]
y_frag = [0, 0, 1, 1, 1]
X = []
y = []

for i in range(198):
    X.extend(X_frag)
    y.extend(y_frag)
    print(i)


X_arr = np.array(X)
y_arr = np.array(y)


shapedtw_type = 'sktime.classification.distance_based._shape_dtw.ShapeDTW'

module_name, class_name = shapedtw_type.rsplit(".", 1)
model = getattr(imp(module_name), class_name)()

param_grid = {'shape_descriptor_function': ['raw', 'paa'] }
parameter_tuning_method = GridSearchCV(model, param_grid, cv=KFold(n_splits=5), return_train_score=True, verbose=3)
parameter_tuning_method.fit(X_arr, y_arr)

#print(parameter_tuning_method.best_score_)
print(parameter_tuning_method.cv_results_)