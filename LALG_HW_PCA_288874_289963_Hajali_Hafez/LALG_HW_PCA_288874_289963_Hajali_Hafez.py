#!/usr/bin/env python
# coding: utf-8
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time

def show_null(df):
    """
        Simple function to show columns with null values + the count
    """
    
    no_na = True
    null_columns = []

    for i, s in enumerate(df.isna().sum()):
        if s > 0:
            no_na = False
            print(df.columns[i], s)
            null_columns.append(df.columns[i])
            
    if no_na: 
        print('No null values')
        
    return null_columns;


# ## Initialization and loading

#random seed - 288874

s = 288874
np.random.seed(s)

def main(s = s):
    data = pd.read_csv('COMBO17.csv')

    # ## Preprocessing
    data.describe()
    show_null(data)

    data[data.isna().any(axis=1)].loc[:, show_null(data)]

    data_wo_null = data.dropna()  # drop the rows with null values
    show_null(data_wo_null)

    data_wo_null.columns  # All the columns

    # Drop useless columns which are useless in our analysis like IDs or columns related to redshift
    useless_columns = ['Nr', 'Mcz', 'e.Mcz', 'MCzml', 'chi2red']

    y = data_wo_null['Mcz']
    X = data_wo_null.drop(columns=useless_columns)

    # Python parameters for displaying all the columns and rows in the dataset
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    def color_red(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for strings>0.85,
        black otherwise.
        """
        color = 'red' if abs(val) > 0.85 else 'black'
        return 'color: %s' % color

    # Show correlation matrix between values. Correlation higher than 85% is highlighted with red
    X.corr().style.applymap(color_red)

    # Print and count highly correlated variables

    corr_mat = X.corr()
    corr_cols = corr_mat.columns

    # Show which variable is highly correlated with any other variable how many times
    ### Uncomment below if you want to see
    # corr_count = dict([(key, 0) for key in corr_cols])  # initialize empty dict
    #
    # for i in range(len(corr_mat)):
    #     for j in range(i + 1, len(corr_mat)):
    #         if corr_mat.iloc[i, j] > 0.85:
    #             print(f"{corr_cols[i]} , {corr_cols[j]} \n {X.corr().iloc[i, j]} \n")
    #
    #             corr_count[corr_cols[i]] += 1
    #             corr_count[corr_cols[j]] += 1
    # corr_count

    # ## Preparing the dataset

    # Split the unpreprocessed dataset into test and train and save them into .csv files
    train, test = train_test_split(data_wo_null, random_state=s,
                                   train_size=2500 / len(y))  # split dataset into train and test(evaluation)

    train_df = pd.DataFrame(train)
    test_df = pd.DataFrame(test)

    train_df = train_df.drop(columns=['Nr', 'e.Mcz', 'MCzml', 'chi2red'])
    test_df = test_df.drop(columns=['Nr', 'e.Mcz', 'MCzml', 'chi2red'])

    train_df.to_csv(f'COMBO17pca_{s}.csv', index=False)
    test_df.to_csv(f'COMBO17eval_{s}.csv', index=False)

    # Return settings back to normal
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=s, train_size=2500 / len(y))
    X_train

    # ## PCA

    # initializing the objects for normal PCA and PCA with normalized dataset
    pca = PCA()
    pca_z = PCA()

    # Normalize the dataset
    znorm = StandardScaler()
    znorm.fit(X_train)
    X_train_hat = znorm.transform(X_train)

    # Fit train data to pca
    pca.fit(X_train)
    pca_z.fit(X_train_hat)

    # Plot visualization for Cumulative sum of explained variance ratios

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), label='PCA')
    ax.plot(np.cumsum(pca_z.explained_variance_ratio_), label='PCA_Z')
    ax.plot(range(X_train.shape[1]), 0.95 * np.ones(X_train.shape[1]), 'r--')
    ax.plot(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0][0] * np.ones(X_train.shape[1]),
            np.linspace(0, 1, 60), 'b--')
    ax.plot(np.where(np.cumsum(pca_z.explained_variance_ratio_) > 0.95)[0][0] * np.ones(X_train.shape[1]),
            np.linspace(0, 1, 60), 'y--')
    ax.set(title='Cumulative sum of explained varience ratios')
    ax.legend(loc='best')
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained varience ratio")

    fig.savefig('cumsum_pca.eps', format='eps')

    # Plot visualization for explained variance ratios for each component
    # It can be seen that PCA without normalization starts with high explained variance ratio
    # than that with normalization
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(pca.explained_variance_ratio_, label='PCA')
    ax.plot(pca_z.explained_variance_ratio_, label='PCA_Z')
    ax.set(title='Explained varience ratios for each component')
    ax.legend(loc='best')
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained varience ratio")

    fig.savefig('evr_pca.eps', format='eps')

    print("Component vs Cum. sum of explained var. ratio")
    for i, p in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        print(i + 2, p)
    # it reaches 99% explanined variance ratio with 8 components

    print("Component vs Cum. sum of explained var. ratio for normalized data")
    for i, p in enumerate(np.cumsum(pca_z.explained_variance_ratio_)):
        print(i + 2, p)
    # it reaches 99% explanined variance ratio with 26 components

    # ## Projections

    Qm = pca.transform(X_train)
    Qm_z = pca_z.transform(X_train_hat)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Qm[:, 0], Qm[:, 2], c=y_train)
    fig.savefig('qm0_2.eps', format='eps')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Qm_z[:, 0], Qm_z[:, 2], c=y_train)
    fig.savefig('qmz0_2.eps', format='eps')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Qm[:, 0], Qm[:, 1], c=y_train)
    fig.savefig('qm0_1.eps', format='eps')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Qm_z[:, 0], Qm_z[:, 1], c=y_train)
    fig.savefig('qmz0_1.eps', format='eps')

    # basic function to calculate MRE
    def mre_f(y_pred, y_true):
        mre_ = sum(abs(y_pred - y_true) / abs(y_true)) / len(y_test)
        return mre_

    # ## Regression analysis with normalized data w/o PCA

    knn_wo_PCA = KNeighborsRegressor(weights='distance')
    s = time.time()
    knn_wo_PCA.fit(X_train_hat, y_train)
    y_pred_wo_PCA = knn_wo_PCA.predict(znorm.transform(X_test))
    e = time.time()
    print("\n\nRegression analysis with normalized data w/o PCA\n--------------------------------------\n")
    print(
        f"r2: {round(r2_score(y_pred_wo_PCA, y_test), 3)}, MAE: {round(mean_absolute_error(y_pred_wo_PCA, y_test), 3)}, MRE: {round(mre_f(y_pred_wo_PCA, y_test), 3)}, Time: {round(e - s, 3)}")

    # ## Regression analysis w/o normalized data w/o PCA

    knn_wo_PCA = KNeighborsRegressor(weights='distance')
    s = time.time()
    knn_wo_PCA.fit(X_train, y_train)
    y_pred_wo_PCA = knn_wo_PCA.predict(X_test)
    e = time.time()
    print("\n\nRegression analysis w/o normalized data w/o PCA\n--------------------------------------\n")
    print(
        f"r2: {round(r2_score(y_pred_wo_PCA, y_test), 3)}, MAE: {round(mean_absolute_error(y_pred_wo_PCA, y_test), 3)}, MRE: {round(mre_f(y_pred_wo_PCA, y_test), 3)}, Time: {round(e - s, 3)}")

    # ## Regression analysis with normalized data for PCA

    pca_20z = PCA(20)
    pca_20z.fit(X_train_hat)
    pcs = pca_20z.transform(X_train_hat)

    accuracies = []
    mae = []
    mre = []
    print("\n\nRegression analysis with normalized data for PCA\n--------------------------------------\n")
    for i in range(2, 30):
        pca_30z = PCA(i)
        pca_30z.fit(X_train_hat)
        pcs = pca_30z.transform(X_train_hat)

        knn = KNeighborsRegressor(weights='distance')
        s = time.time()
        knn.fit(pcs, y_train)
        y_pred = knn.predict(pca_30z.transform(znorm.transform(X_test)))
        e = time.time()

        accuracies.append(r2_score(y_pred, y_test))
        mae.append(mean_absolute_error(y_pred, y_test))
        mre.append(mre_f(y_pred, y_test))

        print(
            f"#Components: {i}, r2: {round(r2_score(y_pred, y_test), 3)}, MAE: {round(mean_absolute_error(y_pred, y_test), 3)}, MRE: {round(mre_f(y_pred, y_test), 3)}, Time: {round(e - s, 3)}")

    # ### With the PCA and normalization it is about 7 times faster to reach even higher for the near R2 score, with as few as 3 components

    # ## Regression analysis without normalization

    pca_20 = PCA(20)
    pca_20.fit(X_train)
    pcs_wo_norm = pca_20.transform(X_train)

    accuracies = []
    mae = []
    mre = []
    print("\n\nRegression analysis without normalization\n--------------------------------------\n")
    for i in range(2, 30):
        pca_30 = PCA(i)
        pca_30.fit(X_train)
        pcs = pca_30.transform(X_train)

        knn = KNeighborsRegressor(weights='distance')
        s = time.time()
        knn.fit(pcs, y_train)
        y_pred = knn.predict(pca_30.transform(X_test))
        e = time.time()

        accuracies.append(r2_score(y_pred, y_test))
        mae.append(mean_absolute_error(y_pred, y_test))
        mre.append(mre_f(y_pred, y_test))

        print(
            f"#Components: {i}, r2: {round(r2_score(y_pred, y_test), 3)}, MAE: {round(mean_absolute_error(y_pred, y_test), 3)}, MRE: {round(mre_f(y_pred, y_test), 3)}, Time: {round(e - s, 3)}")
        # ### With the PCA it is about 3 times faster to reach the same R2 score, with 10 components


if __name__ == '__main__':
    main()