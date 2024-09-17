from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def perform_regression(df, target_column):
    # Splitting the data into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a linear regression model
    model = LinearRegression()

    # Fitting the model
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Regression equation
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.2f}"
    for i, coef in enumerate(coefficients):
        feature_name = X.columns[i]
        equation += f" + {coef:.2f} * {feature_name}"
    print("Regression Equation:", equation)

    # Scatter plot with regression line
    sns.lmplot(x='Warranty Period', y='Price', data=df, aspect=1.5)
    plt.title('Scatter Plot with Regression Line')
    plt.show()

    # Plotting Prediction vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the correlarion heatmap
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Correlation Matrix')
    plt.show()

    # Residual plot
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

    return model