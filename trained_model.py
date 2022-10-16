# Import Python Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb  # Light Gradient Boosting
# import xgboost as xgb # XGBoost
import joblib  # To save trained model in a pickle file
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import config

folds = StratifiedKFold(n_splits=config.k_fold_n_splits, shuffle=config.k_fold_shuffle, random_state=config.random_state)

def loading_initial_data():
    """
    Function to load the use-case data for first time model execution
    """
    # Loading training dataset from Kaggle use-case
    train_data = pd.read_csv(config.path + 'train.csv', index_col='id')
    # Loading testing dataset from Kaggle use-case
    test_data = pd.read_csv(config.path + 'test.csv', index_col='id')
    # Loading submission dataset from Kaggle use-case
    submission_data = pd.read_csv(config.path + 'sample_submission.csv', index_col='id')
    # Target feature of Training data
    y_target = train_data['target'].values

    # Combining Training and Testing data
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Remove the target column from the combined training and testing data.
    combined_data = combined_data.drop('target', axis=1)
    # combined_features = combined_data.columns
    return train_data, combined_data, submission_data, y_target


def remove_duplicates(data):
    """
    Function to remove duplicate records if any.
    """
    return data.drop_duplicates()


def drop_columns(data, columns):
    """
    Function to remove unnecessary features. During EDA, a set of features are identified with maximum
    number of missing values, which can be discarded from the dataset.
    """
    return data.drop(columns, axis=1)


def handle_missing_values(data):
    """
    Function to replace missing values marked as -1 in the data with NaN.
    """
    return data.replace(to_replace=-1, value=np.nan)


def impute_numeric_missing_data(data, column):
    """
    Function to impute or replace missing values of numeric feature with median value.
    """
    return data[column].fillna(data[column].median())


def impute_categorical_missing_data(data):
    """
    Function to impute or replace missing values of categorical feature with mode value.
    """
    return data.fillna(data.mode().iloc[0])


# def categorial_feature_encoding(data):
#     """
#     Function to encode the categorical features using One-Hot Encoding with get_dummies.
#     """
#     data = pd.get_dummies(data, columns=[x for x in data.columns.tolist() if x.endswith('cat')])
#     config.encoded_train_columns = data.columns
#     return data
#
#
# def feature_scaling(data):
#     """
#     Function to scale the data using Standard Scaler.
#     """
#     scaler = StandardScaler()
#     scaler.fit(data)
#     return pd.DataFrame(scaler.transform(data), columns=data.columns)

def preprocessing(data):
    """
    This function preprocess & cleans the data
    :param data: DataFrame
    :return: Preprocessed and cleaned dataframe is returned
    """
    # Remove duplicates
    combined_data = remove_duplicates(data)

    # As per the EDA below columns can be removed from the given set of features
    combined_data = drop_columns(data=combined_data, columns=config.columns_to_remove)
    combined_data = handle_missing_values(data=combined_data)

    # As per EDA, features 'ps_reg_03', 'ps_car_11' & 'ps_car_12' are imputed with median for missing values
    for col in config.impute_numeric_columns:
        combined_data[col] = impute_numeric_missing_data(data=combined_data, column=col)

    # if not target_available:
    combined_data = impute_categorical_missing_data(data=combined_data)

    # Encoding the categorical variables
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_features = [feature for feature in combined_data.columns.tolist() if feature.endswith('cat')]
    encoded_categorical_matrix = onehot_encoder.fit_transform(combined_data[categorical_features])

    # sparse matrix using csr format
    combined_data_sparse = sparse.hstack([sparse.csr_matrix(combined_data), encoded_categorical_matrix],
                                         format="csr")
    return combined_data_sparse

def split_data(data, initial_train_data_length):
    """
    This function splits the data into the training and test dataset while Initial training the model with Training
    and Test data.
    :param data: DataFrame
    :param initial_train_data_length: length of training data
    :return: DataFrames with Training and Test dataset
    """
    # Splitting the combined data into train and test
    x_training = data[:initial_train_data_length]
    x_test = data[initial_train_data_length:]
    return x_training, x_test

def retraining_data_split(data):
    """
    This function splits the data into the training and test dataset while retraining the model with new data.
    :param data: DataFrame
    :return: DataFrames with Training and Test dataset
    """
    """
    This function splits the data into the training and test dataset while retraining the model with new data.
    :param :
    :return: DataFrames with Training and Test dataset
    """
    # Splitting the New data into train and test
    x_training, x_validation = train_test_split(data, test_size=config.test_size, random_state=config.random_state)
    y_target = x_training[['target']]
    x_training = x_training.drop(['target'], axis = 1)
    x_validation = x_validation.drop(['target'], axis=1)
    return x_training, x_validation, y_target

def eval_gini(y_true, predicted_target):
    """
    This function evaluates the gini score based on the Actual and predicted target feature.
    :param y_true: DataFrame of actual target values
    :param predicted_target: DataFrame of predicted target values
    :return: Gini Score
    """
    # An error occurs when the actual and predicted values are different.
    assert y_true.shape == predicted_target.shape

    n_samples = y_true.shape[0]  # number of samples
    l_mid = np.linspace(1 / n_samples, 1, n_samples)  # diagonal value

    # 1) Gini coefficient for predicted values
    predicted_order = y_true[predicted_target.argsort()]  # y_pred -- Sort y_true values by size
    l_predicted = np.cumsum(predicted_order) / np.sum(predicted_order)  # Lorentz curve
    g_predicted = np.sum(l_mid - l_predicted)  # Gini coefficient for predicted values

    # 2) Gini coefficient when prediction is perfect
    true_order = y_true[y_true.argsort()]  # y_true -- Sort y_true values by size
    l_true = np.cumsum(true_order) / np.sum(true_order)  # Lorentz curve
    g_true = np.sum(l_mid - l_true)  # Gini coefficient when prediction is perfect

    # Normalized Gini coefficient
    return g_predicted / g_true

def gini_lgb(prediction, train_data):
    """
    This function is used to gini for LightGBM.
    :param prediction: DataFrame
    :param train_data: DataFrame
    :return: gini score
    """
    labels = train_data.get_label()
    return 'gini', eval_gini(labels, prediction), True

def gini_xgb(prediction, train_data):
    """
    This function is used to gini for XGBoost.
    :param prediction: DataFrame
    :param train_data: DataFrame
    :return: gini score
    """
    labels = train_data.get_label()
    return 'gini', eval_gini(labels, prediction)

def light_gradient_boost_model(folds, x_training, x_test, y_target):
    """
    # This function performs the Light Gradient Boosting model.
    :param folds: K-Fold data
    :return: Prediction data, Validation data gini_score
    """
    # 1-dimensional array containing the probability of predicting the target value using the validation data.
    global lgb_gini_score
    val_preds_lgb = np.zeros(x_training.shape[0])

    # 1-dimensional array containing the probability of predicting the test data target value with the model trained.
    test_preds_lgb = np.zeros(x_test.shape[0])

    # Train, validate, and predict models in a way
    for id_x, (train_idx, valid_idx) in enumerate(folds.split(x_training, y_target)):
        # Print the text that distinguishes each fold
        print('#' * 50, f'fold {id_x + 1} / fold {folds.n_splits}', '#' * 50)

        # Setting data for training and validation
        x_train, y_train = x_training[train_idx], y_target[train_idx]  # training data
        x_valid, y_valid = x_training[valid_idx], y_target[valid_idx]  # Validation data

        # LightGBM dataset
        train_data = lgb.Dataset(x_train, y_train)  # LightGBM training data
        validation_data = lgb.Dataset(x_valid, y_valid)  # LightGBM Validation data

        # LightGBM model training
        lgb_model = lgb.train(params=config.lgb_max_params,  # Optimal hyper-parameters
                              train_set=train_data,  # Training data
                              num_boost_round=config.lgb_num_boost_round,  # number of boosting iterations
                              valid_sets=validation_data,  # Validation dataset for performance evaluation
                              feval=gini_lgb,  # Evaluation Indicators for Verification
                              early_stopping_rounds=config.lgb_early_stopping_rounds,
                              # Early Termination Conditions
                              verbose_eval=config.lgb_verbose_eval)  # Print score every 100th

        # Prediction using test data
        test_preds_lgb += lgb_model.predict(x_test) / folds.n_splits
        # print('test_preds_lgb', test_preds_lgb)

        # Predicting validation data target values for model performance evaluation
        val_preds_lgb[valid_idx] += lgb_model.predict(x_valid)

        # Normalized Gini coefficient for the validation data prediction probability
        lgb_gini_score = eval_gini(y_valid, val_preds_lgb[valid_idx])
        print(f'fold {id_x + 1} Gini coefficient: {lgb_gini_score}\n')

        joblib.dump(lgb_model, 'lgb_model_v1.pkl')
        print('LightGBM - Verification data Gini-Coefficient:', eval_gini(y_target, val_preds_lgb))
    return test_preds_lgb, val_preds_lgb, lgb_gini_score


# def xg_boost_model(folds, x_training, x_test, y_target):
#     """
#     # This function performs the XG Boosting model.
#     :param folds: k-Fold data
#     :return: Prediction data, Validation data gini_score
#     """
#     # 1-dimensional array containing the probability of predicting the target value of the validation data with a model trained.
#     val_preds_xgb = np.zeros(x_training.shape[0])
#     # 1-dimensional array containing the probabilities of predicting the test data with the model trained.
#     test_preds_xgb = np.zeros(x_test.shape[0])
#
#     # Training and predicting the models
#     for id_x, (train_idx, valid_idx) in enumerate(folds.split(x_training, y_target)):
#         # Print output of each fold
#         print('#' * 50, f'Fold {id_x + 1} / Fold {folds.n_splits}', '#' * 50)
#
#         # Set Train & Validation dataset
#         x_train, y_train = x_training[train_idx], y_target[train_idx]
#         x_valid, y_valid = x_training[valid_idx], y_target[valid_idx]
#
#         # XGBoost dataset
#         train_data = xgb.DMatrix(x_train, y_train)
#         validation_data = xgb.DMatrix(x_valid, y_valid)
#         test_data = xgb.DMatrix(x_test)
#
#         # XGBoost model training
#         xgb_model = xgb.train(params=config.max_params_xgb,
#                               dtrain=train_data,
#                               num_boost_round=2000,
#                               evals=[(validation_data, 'valid')],
#                               maximize=True,
#                               feval=gini_xgb,
#                               early_stopping_rounds=200,
#                               verbose_eval=100)
#
#         # Storing the number of boosting iterations when the model performs best
#         best_iterations = xgb_model.best_iteration
#
#         # Prediction using test data
#         test_preds_xgb += xgb_model.predict(test_data, iteration_range=(0, best_iterations)) / folds.n_splits
#
#         # Predicting validation data target values for model performance evaluation
#         val_preds_xgb[valid_idx] += xgb_model.predict(validation_data, iteration_range=(0, best_iterations))
#
#         # Normalized Gini coefficient for the validation data prediction probability
#         xgb_gini_score = eval_gini(y_valid, val_preds_xgb[valid_idx])
#         print(f'Fold {id_x + 1} Gini-Coefficient : {xgb_gini_score}\n')
#
#         # Save the training model in a pickle file for deployment
#         joblib.dump(xgb_model, config.path + 'xgb_model_v1.pkl')
#
#     return test_preds_xgb, xgb_gini_score

# Ensemble Performance during first time training and pickle file generation
# def model_retraining(folds, data):
#     y_target = data[['target']]
#     x_training, x_validation, submission = retraining_data_split(data)
#     test_preds_lgb, val_preds_lgb, lgb_gini_score = light_gradient_boost_model(folds, x_training, x_validation, y_target)
#     # test_preds_xgb, xgb_gini_score = XG_boost_model(folds, x_training, x_validation, y_target)
#     # test_preds = test_preds_lgb * 0.5 + test_preds_xgb * 0.5
#     submission['target'] = test_preds_lgb
#     submission.to_csv('LGB_submission_1.csv')
#     return submission