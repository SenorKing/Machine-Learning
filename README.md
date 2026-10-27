# Machine-Learning

## Linear Regression: BMI vs Diabetes Progression

This project demonstrates a simple linear regression model using the `diabetes` dataset from `scikit-learn`. The goal is to show how a statistical model can learn the relationship between a single input feature, body mass index (`BMI`), and the target variable, disease progression.

### What the code is doing

The script starts by loading the dataset:

```python
x, y = datasets.load_diabetes(return_X_y=True)
```

This dataset contains multiple medical measurements, but for this example we focus only on the third feature, which corresponds to BMI. The code selects that column and reshapes it so it works with a regression model:

```python
x = x[:, 2]
x = x.reshape(-1, 1)
```

The data is then split into training and testing sets:

```python
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.33
)
```

This means the model learns patterns from part of the data (`x_train`, `y_train`) and then evaluates how well it predicts unseen data (`x_test`, `y_test`).

A linear regression model is created and trained:

```python
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
```

The model learns a straight-line relationship of the form:

```python
y = m * x + b
```

where:
- `x` is BMI
- `y` is diabetes progression
- `m` is the slope of the regression line
- `b` is the intercept

In simple terms, the model estimates how much the disease progression changes as BMI changes.

### How the regression trend shows prediction

After training, the model makes predictions for the test data:

```python
y_pred = model.predict(x_test)
```

The plot then compares the real test values against the model's line of best fit:

```python
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
```

The black dots represent the actual BMI and diabetes progression values from the dataset. The blue line is the regression trend, showing the model's predicted relationship between BMI and disease progression.

The line helps explain the model's prediction logic:
- when BMI is lower, the model predicts a lower disease progression
- when BMI is higher, the model predicts a higher disease progression
- the slope captures the average increase in disease progression per increase in BMI

This is why the blue trend line is useful: it visually summarizes how the model is estimating outcomes across the range of BMI values.

### Why this matters

The regression line shows a simplified interpretation of the dataset. It does not mean that BMI alone fully determines diabetes progression, but it helps illustrate that there is a measurable relationship between body mass index and disease progression in this dataset.

The model uses training data to learn this trend and then applies it to new cases to estimate the target value.

### Run the script

```bash
pip install matplotlib scikit-learn pandas ipykernel numpy
python Linear_regression.py
```

This will generate a scatter plot showing the observed data points and the fitted regression line for BMI versus diabetes progression.
