import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_pickles(model_pickle_path, label_encoder_pickle_path):
    with open(model_pickle_path, "rb") as model_pickle_opener:
        model = pickle.load(model_pickle_opener)
    
    with open(label_encoder_pickle_path, "rb") as label_encoder_opener:
        label_encoder_dict = pickle.load(label_encoder_opener)
        
    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    df_out = df.copy()
    df_out.replace(" ", 0, inplace=True)
    df_out.loc[:, 'TotalCharges'] = pd.to_numeric(df_out.loc[:, 'TotalCharges'])
    if 'customerID' in df_out.columns:
        df_out.drop('customerID', axis=1, inplace=True)
    for column, le in label_encoder_dict.items():
        df_out.loc[:, column] = le.transform(df_out.loc[:, column])
        
    return df_out


def make_predictions(test_data):
    model_pickle_path = "./models/churn_prediction_model.pkl"
    label_encoder_pickle_path = "./models/churn_prediction_label_encoder.pkl"
    
    model, label_encoder_dict = load_pickles(model_pickle_path, label_encoder_pickle_path)
    
    data_processed = pre_process_data(test_data, label_encoder_dict)
    if 'Churn' in data_processed.columns:
        data_processed = data_processed.drop(columns=['Churn'])
    prediction = model.predict(data_processed)
    return prediction


if __name__ == "__main__":
    st.title("Customer churn prediction")
    data = pd.read_csv("./data/holdout_data.csv")
    
    # making customer data input
    gender = st.selectbox("Select customer's gender :",
                         ['Female', 'Male'])
    senior_citizen_input = st.selectbox('Is customer a senior citizen? :',
                                     ["No", "Yes"])
    if senior_citizen_input == "Yes":
        senior_citizen = 1
    else:
        senior_citizen = 0
    partner = st.selectbox('Does the customer have a partner? :',
                             ["No", "Yes"])
    dependents = st.selectbox('Does the customer have dependents? :',
                              ["Yes", "No"])
    tenure = st.slider('How many months has the customer been with the company? :',
                       min_value=0, max_value=72, value=24)
    phone_service = st.selectbox('Does the customer have phone service? :',
                             ["No", "Yes"])
    multiple_lines = st.selectbox('Does the customer have multiple lines? :',
                                 ["No", "Yes", "No phone service"])
    internet_service = st.selectbox('What type of internet service does the customer have? :',
                                  ["No", "DSL", "Fiber optic"])
    online_security = st.selectbox('Does the customer have online security? :',
                                  ["No", "Yes", "No internet service"])
    online_backup = st.selectbox('Does the customer have online backup? :',
                                  ["No", "Yes", "No internet service"])
    device_protection = st.selectbox('Does the customer have device protection? :',
                                  ["No", "Yes", "No internet service"])
    tech_support = st.selectbox('Does the customer have tech support? :',
                                  ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox('Does the customer have streaming TV? :',
                                  ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox('Does the customer have streaming movies? :',
                                  ["No", "Yes", "No internet service"])
    contract = st.selectbox('What kind of contract does the customer have? :',
                                  ["Month-to-month", "Two year", "One year"])
    paperless_billing = st.selectbox('Does the customer have paperless billing? :',
                                  ["No", "Yes"])
    payment_method = st.selectbox("What is the customer's payment method? :",
                                     ["Mailed check", "Credit card (automatic)", "Bank transfer (automatic)",
                                      "Electronic check"])
    monthly_charges = st.slider("What is the customer's monthly charge? :", min_value=0, max_value=118, value=50)
    total_charges = st.slider('What is the total charge of the customer? :', min_value=0, max_value=8600, value=2000)
    input_dict = {'gender': gender,
                  'SeniorCitizen': senior_citizen,
                  'Partner': partner,
                  'Dependents': dependents,
                  'tenure': tenure,
                  'PhoneService': phone_service,
                  'MultipleLines': multiple_lines,
                  'InternetService': internet_service,
                  'OnlineSecurity': online_security,
                  'OnlineBackup': online_backup,
                  'DeviceProtection': device_protection,
                  'TechSupport': tech_support,
                  'StreamingTV': streaming_tv,
                  'StreamingMovies': streaming_movies,
                  'Contract': contract,
                  'PaperlessBilling': paperless_billing,
                  'PaymentMethod': payment_method,
                  'MonthlyCharges': monthly_charges,
                  'TotalCharges': total_charges,
                  }
    
    customer_data = pd.DataFrame(
        [input_dict]
    )
    
    if st.button("Predict Churn"):
        prediction = make_predictions(customer_data)[0]
        prediction_string = "will churn" if prediction == 1 else "won't churn"
        st.markdown(f"Customer prediction: {prediction_string}")
    