import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('laptop_price.csv', encoding='ISO-8859-1')

# Data preprocessing
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Inches'] = df['Inches'].astype(float)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Prepare data for training
X = df[['Inches', 'Ram', 'Weight']]
y = df['Price_euros']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the model and scaler
joblib.dump(model, 'model_laptop.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
