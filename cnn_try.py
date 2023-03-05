from sktime.classification.deep_learning.cnn import CNNClassifier
print('imported cnn')
from sktime.datasets import load_unit_test
print('imported dataset')
X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")
cnn = CNNClassifier(n_epochs=20,batch_size=4)
print('fit')
cnn.fit(X_train, y_train)
print('predict')
print(cnn.predict(X_test))