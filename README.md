# Singapore-Resale-Flat-Prices-Predicting

Domain: Real-Estate

Skills needed: Data Wrangling, EDA, Model Building & Evaluation, Model Deployment

# Objective
  The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat. 

# Requirements
* Python
* Pandas
* Numpy
* Streamlit
* Sklearn
# Workflow
**1.Data Collection and Preprocessing:** 

    Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) for the years 2017 to Till Date. Preprocess the data to clean and structure it for machine learning.
  
**2.Feature Engineering:**

    Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.
  
**3.Model Selection and Training:**

    Choose an appropriate machine learning model for regression(Random Forest Regressor). Train the model on the historical data, using a portion of the dataset for training.
  
**4.Model Evaluation:**

    Evaluate the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (RMSE), or Root Mean Squared Error (RMSE) and R2 Score.
  
**5.Streamlit Web Application:**

    Developed a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.
  
**6.Deployment on Hugging Face:**

    Deploy the Streamlit application on the Hugging Face spaces platform to make it accessible to users over the internet.
  
# App Usage
* Run Singapore.py file to launch the app or click the hugging face link given below to access app over internet
* Click the 'Launch ML Model' to launch the model.
* Fill the required information,
  
    1.town,street name,storey range,flat type are selectbox, so select from the given options.
  
    2.lease commence year,foor area,remaining lease in years are numeric inputbox. You can calculate 'remaining lease' = '(lease commence       year +99 ) - current year(eg:2024)'.
  
* Click 'Predict' button
  
* The app will display the prediction below the button.
  
# Links

Hugging Face: https://huggingface.co/spaces/kamalesh-g/Singapore-RealEstate-Streamlit

Linkedin: https://www.linkedin.com/in/g-kamaleashwar-28a2802ba/
 
