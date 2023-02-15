from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
import importlib


model = TimeSeriesKMeans()

print(type(model))
print(type(TimeSeriesKMeans()))
print(type(model) == type(TimeSeriesKMeans()))  # <-------------------
print(model.get_params(), '\n')
#print(model.get_class_tags(), '\n')
#print(model.get_tags(), '\n')

model1 = TimeSeriesKMedoids()
print(model1.get_params(), '\n')


################################################

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test
from sktime.base import BaseEstimator

X_train, y_train = load_unit_test(split="train", return_X_y=True)
X_test, y_test = load_unit_test(split="test", return_X_y=True)

clf = TimeSeriesForestClassifier(n_estimators=5)
clf.fit(X_train, y_train)
print(clf.get_params())
print(clf.is_fitted)
y_pred = clf.predict(X_test)

clf.save('kek_model_clf')

clf_loaded_bad = TimeSeriesForestClassifier()
print(clf_loaded_bad.get_params())
print(clf_loaded_bad.is_fitted)
clf_loaded_bad.load_from_path('kek_model_clf.zip')
print(clf_loaded_bad.get_params())
print(clf_loaded_bad.is_fitted)

clf_loaded_good = TimeSeriesForestClassifier().load_from_path('kek_model_clf.zip')
print(clf_loaded_good.get_params())
print(clf_loaded_good.is_fitted)

# fico e funziona...
estimator_loaded_good = BaseEstimator().load_from_path('kek_model_clf.zip')
print(estimator_loaded_good.get_params())
print(estimator_loaded_good.is_fitted)
print(type(estimator_loaded_good))

baseEst = BaseEstimator()


print(y_pred)
#print(clf_loaded_bad.predict(X_test))
print(clf_loaded_good.predict(X_test))
print(estimator_loaded_good.predict(X_test))


class_string = "sktime.classification.interval_based._tsf.TimeSeriesForestClassifier"

# Split the string into the module and class names
module_name, class_name = class_string.rsplit(".", 1)

# Import the module
module = importlib.import_module(module_name)

# Get the class object from the module
class_obj = getattr(module, class_name)

# Use the instance as needed

print('\n--------------\n')
estimator_from_txt = class_obj()
print(estimator_from_txt.get_params())
print("\t",type(type(clf)))
#estimator_from_txt = TimeSeriesForestClassifier()
print(clf.get_params())
#print(clf.get_fitted_params())
print(clf.get_tags())
print(estimator_from_txt.get_params())
print(estimator_from_txt.get_tags())
estimator_from_txt.set_params(**(clf.get_params()))  # LITERALLY INSANE
print(estimator_from_txt.get_params())
print(estimator_from_txt.get_tags())
print(type(estimator_from_txt))