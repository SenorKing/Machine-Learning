# Machine-Learning

## Linear Regression Example

This repository includes a simple linear regression example in `Linear_regression.py`.

The script:
- loads the diabetes dataset from `scikit-learn`
- selects a single feature (BMI)
- reshapes the data for modeling
- splits the data into training and testing sets
- trains a `LinearRegression` model
- plots the actual values against the model predictions

### Run the example

```bash
pip install matplotlib scikit-learn pandas ipykernel numpy
python Linear_regression.py
```

### Example workflow

```python
from sklearn import datasets, linear_model, model_selection

x, y = datasets.load_diabetes(return_X_y=True)
x = x[:, 2].reshape(-1, 1)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.33
)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```
