# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import random
import scipy.stats as st
#############
# Read data #
#############
# set random seed
np.random.seed(52)
# inputs of the training set
X_train = pd.read_csv('X_train.csv', index_col = 0)
# outputs of the training set
y_train = pd.read_csv('y_train.csv', index_col = 0, squeeze = True)
# inputs of the test set
X_test = pd.read_csv('X_test.csv', index_col = 0)
###############################
# Default LGBM implementation #
###############################
default_lgb = lgb.LGBMClassifier()
default_lgb.fit(X_train, y_train)
# define cross validation
kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
scores = cross_val_score(default_lgb, X_train, y_train, cv = kf)
# training and CV accuracy for default model
print('Accuracy of LGBM on training set: ', default_lgb.score(X_train, y_train))
print('Mean 5-fold CV accuracy for default LGBM parameters: %0.3f' % (scores.mean()))
###################################################################
# Finding the best parameters of LGBM using Bayesian Optimisation #
###################################################################
# define the model
model = lgb.LGBMClassifier()
# obtain range for max_bin
max_bin = max(X_train.nunique())
# define search space
params = dict()
params['learning_rate'] = (0.01, 0.3)
params['n_estimators'] = (100, 10000)
params['num_leaves'] = (8, 4096)
params['max_depth'] = (3, 12)
# this must be greater than 2 when path_smooth is > 0
params['min_data_in_leaf'] = (3, 50)
params['max_bin'] = (2, max_bin)
params['lambda_l1'] = np.linspace(0, 100, 100)
params['lambda_l2'] = np.linspace(0, 100, 100)
params['bagging_fraction'] = (0.2, 1)
params['bagging_freq'] = (0, 15)
params['feature_fraction'] = (0.2, 1)
params['min_gain_to_split'] = (0, 15)
params['min_data_in_bin'] = (3, 20)
# define no. of evaluations
iterations = 100
# define the search
search = BayesSearchCV(estimator = model,
search_spaces = params,
n_iter = iterations,
cv = kf)
# Perform the search, note that this may take a few hours, depending on the machine,
# and will give warnings which are expected as LGBM tries different parameter values
# to the default
search.fit(X_train, y_train)
# Report the best result
print('The best parameters are %s with a 5-fold CV accuracy of %f'
% (search.best_params_, search.best_score_))
# Optimal model based on above
clf = lgb.LGBMClassifier(learning_rate = 0.3,
max_depth = 12,
num_leaves = 8,
n_estimators = 3587,
min_data_in_leaf = 17,
feature_fraction = 0.2,
max_bin = 87)

clf.fit(X_train, y_train)
print('Accuracy of LGBM on training set: ', clf.score(X_train, y_train))
######################
# Feature Importance #
######################
importances = clf.booster_.feature_importance(importance_type = 'gain')
importances_df = pd.concat([pd.DataFrame(importances, columns = ['Importance']),
pd.DataFrame(X_train.columns, columns = ['Word'])], axis=1)
plt.figure()
plt.title('Feature importances')
ax = plt.barh(importances_df.sort_values(by = 'Importance',
ascending = False)[:20]['Word'],
importances_df.sort_values(by = 'Importance',
ascending = False)[:20]['Importance'],
align='center');
plt.xlabel('Cumulative information gain over all splits')
######################
# Export Predictions #
######################
# compute predictions on the test inputs
y_pred = clf.predict(X_test)
# export the predictions on the test data in csv format
prediction = pd.DataFrame(y_pred, columns=['Class'])
prediction.index.name='Index'
prediction.to_csv('P470-P187-P775-P453 - Predictions.csv')
####################################
# CV accuracy confidence intervals #
####################################
# returns cv accuracy estimate, and a confidence interval
def naive_cv(X, Y, estimator, n_folds = 5):
# standard normal .975 quantile
    z = st.norm.ppf(.975)
   # define split
    fold_id = np.array(range(X.shape[0])) % n_folds
    fold_id = random.sample(list(fold_id), len(fold_id))
    accs = []
    for k in range(n_folds):
# define validation indices for fold
        ind_val = list(np.where(np.equal(fold_id, k)))[0]
# define training indices for fold
        ind_train = list(np.where(np.not_equal(fold_id, k)))[0]
# fit model on training set
        estimator.fit(X[ind_train,:], Y[ind_train])
# predict on validation set
        predictions = estimator.predict(X[ind_val, :])
# store 1 - loss for each validation point
        acc_k = (predictions == Y[ind_val])
        accs.extend(acc_k)
    # estimate standard error for cv accuracy
    accs_std = np.std(accs) / np.sqrt(X.shape[0])
# define confidence interval based on normality
    conf_int = [np.mean(accs) - z * accs_std, np.mean(accs) + z * accs_std]
    return dict({'mean':np.mean(accs), 'std.dev':accs_std, '95% CI':conf_int})
# train final model on new seed and print confidence interval
clf1 = lgb.LGBMClassifier(learning_rate = 0.3,
max_depth = 12,
num_leaves = 8,
n_estimators = 3587,
min_data_in_leaf = 17,
feature_fraction = 0.2,
max_bin = 87)
random.seed(104)
clf1_cv = naive_cv(X = np.array(X_train), Y = y_train, estimator = clf2, n_folds = 5)
print('LGBM:', clf1_cv)



clf1 = lgb.LGBMClassifier(learning_rate = 0.15198243170955264,
                          max_depth = 12,
                          num_leaves = 2515,
                          n_estimators = 10000,
                          min_data_in_leaf = 11,
                          min_data_in_bin = 6,
                          bagging_fraction = 0.6048265271041056,
                          bagging_freq = 2,
                          feature_fraction = 0.525979437379654,
                          max_bin = 59)


clf2 = lgb.LGBMClassifier(learning_rate = 0.22834358003273927,
                          max_depth = 11,
                          num_leaves = 2586,
                          n_estimators = 6183,
                          min_data_in_leaf = 50,
                          min_data_in_bin = 20,
                          bagging_fraction = 0.34257092586585495,
                          bagging_freq = 0,
                          feature_fraction = 0.2,
                          max_bin = 82)


scores = cross_val_score(clf1, X_train, y_train, cv = kf)
print(scores.mean())

import timeit
start = timeit.default_timer()
clf = lgb.LGBMClassifier(learning_rate = 0.3,
max_depth = 12,
num_leaves = 8,
n_estimators = 3587,
min_data_in_leaf = 17,
feature_fraction = 0.2,
max_bin = 87)
clf.fit(X_train, y_train)
stop = timeit.default_timer()
print('LGBM Time: ', stop - start)  
