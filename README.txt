
# Customer Churn Prediction

This project aims to predict customer churn for a bank using machine learning. The model is trained on a customer dataset from Kaggle and deployed as a web application using Flask. The app allows users to input customer details (age, gender, subscription length, monthly bill) and receive a churn prediction. The project includes data cleaning, feature engineering, and model training using Random Forest, along with a web interface for easy access.

## Features
- **Customer Input**: Users can input data such as age, gender, subscription length, and monthly bill to predict churn.
- **Churn Prediction**: The model predicts whether a customer is likely to churn based on the input data.
- **Data Visualization**: Displays a countplot of churn data to visualize the distribution.
- **Model Training**: A Random Forest Classifier is trained to predict churn, and the model is saved using pickle for deployment.

## Dataset
The dataset used in this project is the **Bank Customer Churn Prediction dataset** sourced from [Kaggle](https://www.kaggle.com/). It contains customer information, including demographic and account-related features, as well as a target variable indicating whether the customer churned.

### Key Features:
- `Age`: The age of the customer.
- `Gender`: The gender of the customer.
- `Subscription_Length`: The duration in months for which the customer has been subscribed to the bank.
- `Monthly_Bill`: The amount billed to the customer each month.
- `Churn`: A binary variable indicating if the customer churned (1) or not (0).

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/customer-churn-prediction.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the dataset `Bank Customer Churn Prediction.csv` is placed in the correct directory (refer to the path in the `app.py` file).

4. Run the application:
   ```
   python app.py
   ```

   This will start a local development server. Visit `http://127.0.0.1:5000/` in your browser to interact with the churn prediction model.

## How It Works

1. **Data Cleaning**: The data is loaded, missing values are handled, and categorical variables are encoded using Label Encoding.
2. **Model Training**: A Random Forest Classifier is used to train the model, which is then saved to a `.pkl` file for later use.
3. **Flask Web Application**: The Flask app serves an API for churn predictions. The data is presented through an HTML table, and the user can make predictions by providing their customer information.

## API Endpoints

- **GET `/`**: Returns a basic homepage message.
- **POST `/predict`**: Accepts JSON data containing customer features and returns a churn prediction.
  Example input:
  ```json
  {
    "features": [30, 12, 100, 1]
  }
  ```
  The output will be:
  ```json
  {
    "Churn Prediction": 1
  }
  ```
  where `1` indicates churn and `0` indicates no churn.

- **GET `/index`**: Displays the customer churn dataset in a table format.

## Requirements
- Python 3.x
- Flask
- Scikit-learn
- Pandas
- Seaborn
- Matplotlib
- Pickle

Install the dependencies using:
```
pip install flask scikit-learn pandas seaborn matplotlib
```

## Future Improvements
- **Model Optimization**: Experiment with different machine learning models for better accuracy.
- **UI Enhancements**: Improve the user interface for better customer experience.
- **Model Retraining**: Allow the model to be retrained with new data from the application.

## License
This project is licensed under the MIT License.
