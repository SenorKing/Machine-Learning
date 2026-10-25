# Welcome 
print("Hello World")
import sys
import numpy as np

print(sys.executable)
print(np.__version__)

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(a[0])
%pip install matplotlib scikit-learn pandas ipykernel
import sys
import numpy as np

print("Python:", sys.executable)
print("NumPy version:", np.__version__)

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(a[0])
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

#load the diabetes dataset
x, y = datasets.load_diabetes(return_X_y=True)

#print the shape of the data and the first row
print(x.shape)
print(x[0])
# extract the first feature (column) from the dataset
x = x[:, 2]
print(x.shape)

# reshape the data to be a 2D array with one column
x = x.reshape(-1, 1)
print(x.shape)
#split the data into training and testing sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.33)
# create a linear regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
# predict the target values for the test set
y_pred = model.predict(x_test)
#create a scatter plot of the test data and the predicted values
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

# add labels and title to the plot
plt.xlabel('Scaled BMI')
plt.ylabel('Disease Progression')
plt.title('A graph showing Diabetes progression against BMI') 

# draw the plot
plt.show()
