# Parameters
path = "D:\\1Applied_UOH_PG\\Project\\data\\"
training_columns = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin',
                    'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin','ps_ind_12_bin',
                    'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
                    'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
                    'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat','ps_car_09_cat',
                    'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15',
                    'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07',
                    'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14',
                    'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
                    'ps_calc_20_bin']

k_fold_n_splits = 5
random_state = 100
test_size = 0.2
k_fold_shuffle = True
columns_to_remove = ['ps_car_03_cat', 'ps_car_05_cat', 'ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

impute_numeric_columns = ['ps_car_11', 'ps_car_12', 'ps_reg_03']

encoded_train_columns = []
# Parameters for LightGBM model
lgb_max_params = {'bagging_fraction': 0.6,
                  'feature_fraction': 0.6, # default = 1.0. LGBM will select 60% of features before training each tree
                  'lambda_l1': 0.7,
                  'lambda_l2': 0.98,
                  'min_child_samples': 9,
                  'min_child_weight': 36,
                  'num_leaves': 40,
                  'objective': 'binary',
                  'learning_rate': 0.005,
                  'bagging_freq': 1,
                  'force_row_wise': True, # Default = False, true for column wise histogram, when the number of column is large or total # of bins is large.
                  'random_state': 1001}

lgb_num_boost_round = 2500
lgb_early_stopping_rounds = 300
lgb_verbose_eval = 100

# Parameters for XGBoost model
xgb_max_params = {'colsample_bytree': 0.8,
                  'gamma': 10.4,
                  'max_depth': 7,
                  'min_child_weight': 6.5,
                  'reg_alpha': 8.5,
                  'reg_lambda': 1.4,
                  'scale_pos_weight': 1.4,
                  'subsample': 0.7,
                  'objective': 'binary:logistic',
                  'learning_rate': 0.02,
                  'random_state': 1001}

xgb_num_boost_round = 2500
xgb_early_stopping_rounds = 300
xgb_verbose_eval = 100
