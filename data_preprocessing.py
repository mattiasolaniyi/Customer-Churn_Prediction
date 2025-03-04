import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/matthiasolaniyi/Desktop/Computing/customer_churn_prediction/Datasets/Bank Customer Churn Prediction.csv")
#df = pd.read_csv("/Users/matthiasolaniyi/Desktop/Computing/customer_churn_prediction/Datasets/Generated_customer_churn.csv")
# Check missing values
print(df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)

# Convert categorical columns to numeric
#df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["churn"] = df["churn"].map({ 1: "Yes", 0 : "No" })

# Data visualization
#sns.countplot(x="Churn", data=df)
sns.countplot(x="churn", data=df)
plt.show()

# Save cleaned data
df.to_csv("cleaned_data_Bank.csv", index=False)
