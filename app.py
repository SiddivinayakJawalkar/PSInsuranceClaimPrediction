### Necessary libraries
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder #Encoding the data
from sklearn.preprocessing import StandardScaler # Scaling the data
import lightgbm as lgb # Light Gradient Boosting
# import joblib # To save trained model in a pickle file
from scipy import sparse
import pickle
from PIL import Image
import streamlit as st
from trained_model import *
import config

# suppress_st_warning=True
# @st.cache()
# Model Prediction
def main():
    """
    This is the main function to design the application using the streamlit and python functions
    :return:
    """
    # display the front end aspect
    image = Image.open("Porto_Seguro_Logo.png")
    st.image(image)

    st.header("Porto Seguro Insurance Claim Prediction")
    with st.expander("About Porto Seguro Insurance Claim Prediction use-case"):
        st.write("""Porto Seguro is one of Brazil’s largest auto, homeowner & life insurance companies.\n\n"
            "The company wants to improve their services for customers raising legit insurance \nclaims and avoid the "
            "customers who raise false insurance claims for their vehicles. \n\nFor this Porto Seguro want a system that "
            "will predict if a particular customer will \nraise an auto insurance claim in the next year or not. "
            "Porto Seguro have provided a \ndataset for this on Kaggle. \n\nIn this project we will use this dataset along "
            "with Machine Learning / Deep \nLearning and data analysis techniques to design a system that will predict \n"
            "whether or not a particular customer will raise an auto insurance.
            \n For more details refer [Kaggle Link](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/overview)""")


    predict_option, retrain_option, analysis= st.tabs(['Prediction', 'Re-Training Model','Analysis'])
    with predict_option:
        st.info("Performing the prediction using the uploaded data")
        # adding a file uploader
        uploaded_file_predict = st.file_uploader("Please choose a file for prediction.")
        try:
            if uploaded_file_predict is not None:
                uploaded_df = pd.read_csv(uploaded_file_predict)
                test_df = uploaded_df.drop(['id'], axis=1)
                if set(test_df.columns) == set(config.training_columns):
                    st.info("Uploaded Data: ")
                    st.dataframe(data=test_df, use_container_width=True)
                    # Load the trained Light Gradient Boosting Model pickle file saved in Trained_model.py script
                    lgb_model = pickle.load(open('lgb_model.pkl', "rb"))

                    # # Steps to predict using the test data
                    processed_data = preprocessing(data=test_df)
                    y_prediction = lgb_model.predict(processed_data,predict_disable_shape_check=True)
                    # st.dataframe(data=y_prediction, use_container_width=True)
                    st.info("Gini-score of the uploaded data is predicted")
                    submission = pd.DataFrame(uploaded_df['id'])
                    submission['target'] = y_prediction
                    submission['Claim_status'] = np.where(y_prediction>0.5, "Will claim","Does not claim")
                    st.dataframe(data = submission, use_container_width=False)
                    # submission.to_csv(config.path + "/" + "Submission_test.csv", index=False) #Commenting for Cloud
                else:
                    st.error("The Uploaded file has {} features instead of {} features".
                             format(len(uploaded_df.columns), len(config.training_columns)))
                    # st.error("The uploaded file features does not match with the expected input features.", icon="🚨")
        except Exception as e:
            # st.info(e)
            st.info("The uploaded file is empty/invalid. Please upload a valid file for Prediction.")

    with retrain_option:
        st.info("Using the new data, model will be generated")
        uploaded_file_retrain = st.file_uploader("Please choose a new training file.")
        try:
            if uploaded_file_retrain is not None:
                retrain_data = pd.read_csv(uploaded_file_retrain)
                # Steps to predict using the test data
                # y_target = retrain_data[['target']]
                submission = pd.DataFrame(retrain_data['id'])
                retrain_data = retrain_data.drop(['id'], axis=1)
                x_training, x_validation, y_target = retraining_data_split(retrain_data)
                processed_data = preprocessing(data=retrain_data)
                x_training, x_validation = split_data(data=processed_data, initial_train_data_length= len(x_training))
                test_preds_lgb, val_preds_lgb, lgb_gini_score = light_gradient_boost_model(folds, x_training, x_validation, y_target)
                submission['target'] = test_preds_lgb
                # submission.to_csv('LGB_submission_1.csv')

        except Exception as e:
            # st.info(e)
            st.info("Validating the Re-training file size")
    with analysis:
        st.header("Analysis of Target Variable")
        st.image(Image.open("Only_Target_Variable_Analysis.png"))

        st.header("Correlation Matrix of Categorical Variable")
        st.image(Image.open("Categorical_correlation_matrix.png"))


        st.header("Correlation Matrix of Binary Variable")
        st.image(Image.open("Binary_correlation matrix.png"))

        st.header("Distribution of a 'ps_calc_14' variable")
        st.image(Image.open("Distribution_feature.png"))


        st.header("Binary 'ps_ind_17' variable analysis")
        st.image(Image.open("Target_Variable_Analysis_2.png"))
if __name__ == '__main__':
    main()
