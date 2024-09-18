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

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit app
st.markdown("<h1 style='text-align: center;'>Regression Analysis Dashboard</h1>", unsafe_allow_html=True)

# Add a divider
st.divider()

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

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Create the regression equation
equation = f"y = {intercept:.2f}"
for i, coef in enumerate(coefficients):
    feature_name = X.columns[i]
    equation += f" + {coef:.2f} * {feature_name}"

# User inputs for predicting fridge price
st.sidebar.subheader('Select parameters for predicting fridge price')

# Capacity
capacity = st.sidebar.number_input('Capacity', min_value=0, max_value=8000, value=5000)

# Brand
brands = ['Whirlpool', 'Electrolux', 'Bosch', 'Br@ndX', 'Frigidaire', 'GE', 'Haier', 'LG', 'Panasonic', 'Samsung', 'Siemens']
selected_brand = st.sidebar.selectbox('Select Brand', options=brands)
whirlpool = 1 if selected_brand == 'Whirlpool' else 0
electrolux = 1 if selected_brand == 'Electrolux' else 0
bosch = 1 if selected_brand == 'Bosch' else 0
brandx = 1 if selected_brand == 'Br@ndX' else 0
frigidaire = 1 if selected_brand == 'Frigidaire' else 0
ge = 1 if selected_brand == 'GE' else 0
haier = 1 if selected_brand == 'Haier' else 0
lg = 1 if selected_brand == 'LG' else 0
panasonic = 1 if selected_brand == 'Panasonic' else 0
samsung = 1 if selected_brand == 'Samsung' else 0
siemens = 1 if selected_brand == 'Siemens' else 0

# Energy rating
ratings = ['A', 'A+', 'A++', 'B']
selected_rating = st.sidebar.selectbox('Select Energy Rating', options=ratings)
a = 1 if selected_rating == 'A' else 0
a_plus = 1 if selected_rating == 'A+' else 0
a_2plus = 1 if selected_rating == 'A++' else 0
b = 1 if selected_rating == 'B' else 0

# Type
types = ['Double Door', 'Single Door', 'Side-By-Side']
selected_type = st.sidebar.selectbox('Select Door Type', options=types)
double_door = 1 if selected_type == 'Double Door' else 0
single_door = 1 if selected_type == 'Single Door' else 0
side_by_side = 1 if selected_type == 'Side-By-Side' else 0

# Feature
features = ['Ice Dispenser', 'Water Dispenser', 'Smart Connectivity']
selected_feature = st.sidebar.selectbox('Select Feature', options=features)
ice = 1 if selected_feature == 'Ice Dispenser' else 0
water = 1 if selected_feature == 'Water Dispenser' else 0
smart = 1 if selected_feature == 'Smart Connectivity' else 0

# Color
colors = ['Black', 'Gray', 'Silver', 'White']
selected_color = st.sidebar.selectbox('Select Color', options=colors)
black = 1 if selected_color == 'Black' else 0
gray = 1 if selected_color == 'Gray' else 0
silver = 1 if selected_color == 'Silver' else 0
white = 1 if selected_color == 'White' else 0

# Material
materials = ['Glass', 'Stainless Steel', 'Plastic']
selected_material = st.sidebar.selectbox('Select Material', options=materials)
glass = 1 if selected_material == 'Glass' else 0
plastic = 1 if selected_material == 'Plastic' else 0
steel = 1 if selected_material == 'Stainless Steel' else 0

# Country of manutacturing
countries = ['China', 'India', 'USA', 'Germany']
selected_country = st.sidebar.selectbox('Select Country', options=countries)
china = 1 if selected_country == 'China' else 0
india = 1 if selected_country == 'India' else 0
usa = 1 if selected_country == 'USA' else 0
germany = 1 if selected_country == 'Germany' else 0

# Warranty
warranty = st.sidebar.number_input('Warranty (Years)', min_value=10, max_value=90, value=20, step=10)

# Prediction Calculation
input_features = [[capacity, warranty, a, a_plus, a_2plus, b, bosch, brandx, electrolux, frigidaire, ge, haier, lg, panasonic, samsung, siemens,
                   whirlpool, double_door, side_by_side, single_door, ice, smart, water, black, gray, silver, white, glass, plastic, steel,
                   china, germany, india, usa]]

# Columns for price and mean squared error
col1, col2 = st.columns(2)  # Create two columns

# Predict the price using the regression model
with col1:
    predicted_price = model.predict(input_features)[0]
    formatted_price = f"$ {predicted_price:.0f}"  # No decimal places and no "k"

    # Display predicted price with custom styling: bold, larger, and colored
    st.subheader('Predicted Fridge Price')
    st.markdown(f"<h2 style='color: green; font-weight: bold;'>{formatted_price}</h2>", unsafe_allow_html=True)

# Calculating Mean Squared Error
with col2:
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")

# Add a divider
st.divider()

# Columns for plots
col1, col2 = st.columns(2)  # Create two columns

# Plot 3: Prediction vs Actual Values Plot
with col1:
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue', label='Predicted Values')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction vs Actual Values')
    ax.legend()
    st.pyplot(fig)

# Plot 4: Heatmap of Correlation Matrix
with col2:
    fig, ax = plt.subplots()
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap of Correlation Matrix')
    st.pyplot(fig)

# Add a divider
st.divider()

# Correlation between user selected variables
st.sidebar.header('Select Variables for Correlation Plot')
columns = df.columns.tolist()
variable_1 = st.sidebar.selectbox('Select the first variable', columns)
variable_2 = st.sidebar.selectbox('Select the second variable', columns)
if variable_1 != variable_2:
    # Plotting the scatter plot with regression line
    st.markdown(f"<h2 style='text-align: center;'>Scatter Plot of {variable_1} VS {variable_2}</h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(20, 17))  # Adjust the width and height as needed
    sns.regplot(x=df[variable_1], y=df[variable_2], ax=ax, scatter_kws={'s':50}, line_kws={'color':'red'})
    ax.set_xlabel(variable_1)
    ax.set_ylabel(variable_2)
    st.pyplot(fig)
else:
    st.write("Please select different variables for the scatter plot.")