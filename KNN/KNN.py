import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model

data = pd.read_csv('car.data')
# print(data)

le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.1)
# print(X, Y)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

right = 0
total = 0
for i in range(len(x_test)):
    print("Prediction: ", predicted[i], "Actual: ", y_test[i])
    total += 1
    right += (predicted[i] == y_test[i])

print("Acc: ", (right/total)*100)


