import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

acc = 0
mae = 10000

for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.2)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    cur_acc = linear.score(x_test, y_test)
    predictions = linear.predict(x_test)
    cur_mae = sklearn.metrics.mean_absolute_error(y_test, predictions)
    print("Acc: ", cur_acc, " ||  MAE :", cur_mae)
    if abs(cur_mae) < abs(mae):
        acc = cur_acc
        mae = cur_mae
        with open('StudentModel.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('StudentModel.pickle', 'rb')
linear = pickle.load(pickle_in)

# print("Coeff: \n", linear.coef_)
# print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(x_test[i], y_test[i], predictions[i])

mae = sklearn.metrics.mean_absolute_error(y_test, predictions)
print("Mean Absolute Error: ", mae)

p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel("P")
plt.ylabel("Final Grade")
plt.show()