import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_squared_error
from data_processing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = load_data()

# Preprocess the data
df = preprocess_data(df)

# Streamlit app
st.title('Regression Analysis Dashboard')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
test_size = st.sidebar.slider('Test Size', 0.1, 0.9, 0.2)

# Splitting the data into features and target
X = df.drop(columns=['Price'])
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Create the regression equation
equation = f"y = {intercept:.2f}"
for i, coef in enumerate(coefficients):
    feature_name = X.columns[i]
    equation += f" + {coef:.2f} * {feature_name}"

st.subheader('Regression Equation')
st.write(equation)

# Plot 1: Scatter Plot with Regression Line
st.subheader('Scatter Plot with Regression Line')
fig, ax = plt.subplots()
sns.regplot(x='Warranty Period', y='Price', data=df, ax=ax)
st.pyplot(fig)

# Plot 2: Residual Plot
st.subheader('Residual Plot')
residuals = y_test - y_pred
fig, ax = plt.subplots()
ax.scatter(y_pred, residuals)
ax.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
st.pyplot(fig)

# Plot 3: Prediction vs Actual Values Plot
st.subheader('Prediction vs Actual Values')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Prediction vs Actual Values')
ax.legend()
st.pyplot(fig)

# Plot 4: Heatmap of Correlation Matrix
st.subheader('Heatmap of Correlation Matrix')
fig, ax = plt.subplots()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Heatmap of Correlation Matrix')
st.pyplot(fig)