import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load dataset
df = pd.read_csv("cleaned_data_Bank.csv")

# Step 2: Display dataset columns to verify names
print("Columns in dataset:", df.columns)

# Step 3: Check for missing values before processing
print("\nMissing values before processing:")
print(df.isnull().sum())

# Step 4: Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns:", categorical_columns)

# Step 5: Apply Label Encoding for categorical variables
label_enc = LabelEncoder()
for col in categorical_columns:
    df[col] = df[col].astype(str)  # Convert to string to avoid errors
    df[col] = label_enc.fit_transform(df[col])

# Step 6: Fill missing values
df.fillna(df.mean(), inplace=True)

# Step 7: Verify missing values after imputation
print("\nMissing values after imputation:")
print(df.isnull().sum())

# Step 8: Check if "churn" column exists and correct any mislabeling
if "Churn" in df.columns:
    df.rename(columns={"Churn": "churn"}, inplace=True)

if "churn" not in df.columns:
    raise ValueError("The dataset does not contain a 'churn' column.")

# Step 9: Convert target variable "churn" to numeric if necessary
if df["churn"].dtype == 'object':
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Step 10: Ensure no missing values in the target variable
if df["churn"].isnull().sum() > 0:
    raise ValueError("Missing values detected in the target column after processing.")

# Step 11: Split data into features (X) and target (y)
X = df.drop(columns=["churn"])
y = df["churn"]

# Step 12: Verify feature and target shapes
print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Step 13: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 14: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 15: Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Step 16: Save model to file
with open("churn_model_Bank.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel training complete. Model saved successfully!")
