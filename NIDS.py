from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_halving_search_cv
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import sys

def main(heldout_testset=None, classification_method=None, task=None, load_model_name=None):
    df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)

    # Remove the space in all columns
    df = df.applymap(lambda x: x.strip() if type(x)==str else x)

    # Remove the value other than 0 and 1 in binary column 'is_ftp_login'
    df['is_ftp_login'] = np.where(df['is_ftp_login']>1, 1, df['is_ftp_login'])

    # Remove duplicated values for Backdoors and backdoor in attact_cat
    df['attack_cat'] = df['attack_cat'].replace('Backdoors','Backdoor', regex=True)

    # Handling the data set
    for col in df.columns:
        # If the column is of a 'object' type, factorize it
        if df[col].dtype == 'object':
            df[col], _ = pd.factorize(df[col])

    X = df.drop(['attack_cat', 'Label'], axis=1)
    y = df['Label']

    # If attack_cat is null, fill it with 'None'
    df['attack_cat'] = df['attack_cat'].fillna('None')

    # Impute the missing values
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = iterative_imputer.fit_transform(X)
    # X_scaled = MinMaxScaler().fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    if heldout_testset is not None:
        # Do the same preprocessing for heldout_testset
        heldout_testset = heldout_testset.applymap(lambda x: x.strip() if type(x)==str else x)
        heldout_testset['is_ftp_login'] = np.where(heldout_testset['is_ftp_login']>1, 1, heldout_testset['is_ftp_login'])
        for col in heldout_testset.columns:
            if heldout_testset[col].dtype == 'object':
                heldout_testset[col], _ = pd.factorize(heldout_testset[col])

        X_heldout = heldout_testset.drop(['attack_cat', 'Label'], axis=1)
        y_heldout_test = heldout_testset['Label']

        # If attack_cat is null, fill it with 'None'
        heldout_testset['attack_cat'] = heldout_testset['attack_cat'].fillna('None')

        # Impute the missing values
        iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
        X_heldout_imputed = iterative_imputer.fit_transform(X_heldout)
        # X_scaled = MinMaxScaler().fit_transform(X_imputed)

        X_heldout_test = pd.DataFrame(X_heldout_imputed, columns=X_heldout.columns)

    def feature_selection():
        """ Part 1: Feature Selection """

        # Use RandomForest to select important features

        # Creating the model instance
        model = RandomForestClassifier(random_state=42)

        # Load pre-trained model if exists, otherwise train and save new model
        model_path = 'trained_model.joblib'
        if os.path.exists(model_path):
            model = load(model_path)
            print("Load pre-trained model.")
        else:
            model.fit(X_train, y_train)
            dump(model, model_path)
            print("Train and save new model.")

        # Evaluate the model
        y_valid_pred = model.predict(X_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        print("Validation set accuracy before selecting important features: ", valid_accuracy)
        print(classification_report(y_valid, y_valid_pred))

        # Load important features if exists, otherwise train and save new model with important features
        if os.path.exists('important_features.npy'):
            important_features = np.load('important_features.npy', allow_pickle=True)
            print("Load important features.")
            X_important_train = X_train[important_features]
            X_important_valid = X_valid[important_features]
            X_important_test = X_test[important_features]    
        else:
            # Choosing the important features
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            names = [X_train.columns[i] for i in indices]

            cumulative_importance = np.cumsum(importances[indices])
            threshold = 0.95
            selected_features = np.where(cumulative_importance > threshold)[0][0]

            selected_features_indices = indices[:selected_features]
            
            # Validating the selected features
            X_important_train = X_train.iloc[:, selected_features_indices]
            X_important_valid = X_valid.iloc[:, selected_features_indices]
            X_important_test = X_test.iloc[:, selected_features_indices]

            # Delete features with high correlation
            corr_threshold = 0.90
            corr_matrix = X_important_train.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
            print("Features to drop: ", to_drop)
            X_important_train = X_important_train.drop(to_drop, axis=1)
            X_important_valid = X_important_valid.drop(to_drop, axis=1)
            X_important_test = X_important_test.drop(to_drop, axis=1)

            # Print the names of the selected features
            selected_features_names = [X_important_train.columns[i] for i in range(len(X_important_train.columns))]
            print("Selected features: ", selected_features_names)

            important_features = X_important_train.columns
            np.save('important_features.npy', important_features)
            print("Save important features.")

        # # Plot the feature importances
        # plt.figure(figsize=(10, 6))
        # plt.title("Feature Importance")
        # plt.bar(range(X_train.shape[1]), importances[indices])
        # plt.xticks(range(X_train.shape[1]), names, rotation=90)
        # plt.show()

        # # Plot the cumulative feature importances
        # plt.figure(figsize=(10, 6))
        # plt.title("Cumulative Feature Importance")
        # plt.plot(np.cumsum(importances[indices]), marker='o', linestyle='-')
        # plt.xlabel("Number of features")
        # plt.ylabel("Cumulative importance")
        # plt.show()

        # # Plot the confusion matrix
        # cm = confusion_matrix(y_valid, y_valid_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix before selecting important features")
        # plt.show()

        # # Plot the correlation matrix
        # corr = X_train.corr()
        # plt.figure(figsize=(10, 6))
        # plt.matshow(corr, fignum=1)

        model_important_features = RandomForestClassifier(random_state=42)

        model_important_features_path = 'trained_model_with_important_features.joblib'
        if os.path.exists(model_important_features_path):
            model_important_features = load(model_important_features_path)
            print("Load pre-trained model with important features.")
        else:
            model_important_features.fit(X_important_train, y_train)
            dump(model_important_features, model_important_features_path)
            print("Train and save new model with important features.")

        y_important_valid_pred = model_important_features.predict(X_important_valid)
        valid_accuracy_important_features = accuracy_score(y_valid, y_important_valid_pred)

        print("Validation set accuracy with important features: ", valid_accuracy_important_features)

        print(classification_report(y_valid, y_important_valid_pred))

        # cm = confusion_matrix(y_valid, y_important_valid_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix with Important Features")
        # plt.show()

        # # Plot the correlation matrix of selected features
        # corr = X_important_train.corr()
        # plt.figure(figsize=(10, 6))
        # plt.title("Correlation Matrix of Selected Features")
        # plt.matshow(corr, fignum=1)
        # plt.xticks(range(len(X_important_train.columns)), X_important_train.columns, rotation=90)
        # plt.yticks(range(len(X_important_train.columns)), X_important_train.columns)
        # plt.colorbar()
        # plt.show()

    def label_classification(heldout_testset, classification_method, load_model_name):
        """ Part 2: Label Classification """

        important_features = np.load('important_features.npy', allow_pickle=True)

        # Use RandomForest to predict the label

        # Create the model instance
        model_RF_label = RandomForestClassifier(random_state=42)

        # Load pre-trained model if exists, otherwise train and save new model
        if load_model_name:
            model_RF_label_path = load_model_name
        else:
            model_RF_label_path = 'trained_model_RF_label.joblib'
        if os.path.exists(model_RF_label_path):
            model_RF_label = load(model_RF_label_path)
            print("Load pre-trained model.")
        else:
            # Define the hyperparameters grid
            param_dist = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
            }

            # Define the search
            search = HalvingRandomSearchCV(model_RF_label, param_dist, random_state=42, cv=5, n_jobs=-1, verbose=0)

            # Fit the model
            search.fit(X_train[important_features], y_train)

            # Print the best hyperparameters
            print("Best hyperparameters: ", search.best_params_)
            print("Best cross-validation accuracy: ", search.best_score_)

            # Use the best model to predict the test set
            model_RF_label = search.best_estimator_
            model_RF_label.fit(X_train[important_features], y_train)
            dump(model_RF_label, model_RF_label_path)

        if(classification_method == 'RF'):
            if heldout_testset:
                y_pred_label = model_RF_label.predict(X_heldout_test[important_features])
                print("Part 2: Test set accuracy with important features: ", accuracy_score(y_heldout_test, y_pred_label))
                print(classification_report(y_heldout_test, y_pred_label))
            else:
                y_pred_label = model_RF_label.predict(X_test[important_features])
                print("Part 2: Test set accuracy with important features: ", accuracy_score(y_test, y_pred_label))
                print(classification_report(y_test, y_pred_label))

        # If the classification method is not specified, use the best model to predict the test set
        if not classification_method:
            if heldout_testset:
                y_pred_label = model_RF_label.predict(X_heldout_test[important_features])
                print("Part 2: Test set accuracy with important features: ", accuracy_score(y_heldout_test, y_pred_label))
                print(classification_report(y_heldout_test, y_pred_label))
            else:
                y_pred_label = model_RF_label.predict(X_test[important_features])
                print("Part 2: Test set accuracy with important features: ", accuracy_score(y_test, y_pred_label))
                print(classification_report(y_test, y_pred_label))

    def attack_category_classification(heldout_testset, classification_method, load_model_name):
        """ Part 3: Attack Category Classification """

        important_features = np.load('important_features.npy', allow_pickle=True)

        X_imputed = iterative_imputer.transform(df.drop(['attack_cat', 'Label'], axis=1))
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, df['attack_cat'], test_size=0.3, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


        if not classification_method:
            classification_method = 'RF'
        
        if classification_method == 'RF':
            # Use RandomForest to predict the attack_cat
            model_RF_label_path = 'trained_model_RF_label.joblib'
            model_RF_label = load(model_RF_label_path)
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
            X_imputed['pred_label'] = model_RF_label.predict(X_imputed[important_features])
            # Add 'pred_label' to important features
            important_features = np.append(important_features, 'pred_label')

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, df['attack_cat'], test_size=0.3, random_state=42)
            X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            model_RF_attack_cat = RandomForestClassifier(random_state=42)

            # If the model exists, load it, otherwise train and save the model
            if load_model_name:
                model_RF_attack_cat_path = load_model_name
            else:
                model_RF_attack_cat_path = 'trained_model_RF_attack_cat.joblib'
            if os.path.exists(model_RF_attack_cat_path):
                model_RF_attack_cat = load(model_RF_attack_cat_path)
                print("Load pre-trained model.")
            else:
                # Define the parameter grid
                param_dist = {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }

                # Define the search
                search = HalvingRandomSearchCV(model_RF_label, param_dist, random_state=42, cv=5, n_jobs=-1, verbose=0)

                # Fit the model
                search.fit(X_train[important_features], y_train, sample_weight=sample_weights)

                # Print the best hyperparameters
                print("Best hyperparameters: ", search.best_params_)
                print("Best cross-validation accuracy: ", search.best_score_)
                model_RF_attack_cat = search.best_estimator_

                model_RF_attack_cat.fit(X_train[important_features], y_train, sample_weight=sample_weights)

                dump(model_RF_attack_cat, model_RF_attack_cat_path)
            if heldout_testset:
                X_imputed = iterative_imputer.transform(heldout_testset.drop(['attack_cat', 'Label'], axis=1))
                X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
                X_imputed['pred_label'] = model_RF_label.predict(X_imputed[important_features])
                # Add 'pred_label' to important features
                important_features = np.append(important_features, 'pred_label')
                y_pred = model_RF_attack_cat.predict(X_imputed[important_features])
                print(classification_report(heldout_testset['attack_cat'], y_pred))
                print('Accuracy score',accuracy_score(heldout_testset['attack_cat'], y_pred))
            else:
                y_pred = model_RF_attack_cat.predict(X_test[important_features])

                print("Part 3: Test set accuracy with important features: ", accuracy_score(y_test, y_pred))
                print(classification_report(y_test, y_pred))

    if task == 'feature_selection':
        feature_selection()
    elif task == 'label_classification':
        feature_selection()
        label_classification(heldout_testset, classification_method, load_model_name)
    elif task == 'attack_category_classification':
        feature_selection()
        attack_category_classification(heldout_testset, classification_method, load_model_name)
    else:
        feature_selection()
        label_classification(heldout_testset, classification_method, load_model_name)
        attack_category_classification(heldout_testset, classification_method, load_model_name)

if __name__ == '__main__':

    heldout_testset = None
    classification_method = None
    task = None
    load_model_name = None

    if len(sys.argv) > 1:
        heldout_testset = pd.read_csv(heldout_testset, low_memory=False)
    if len(sys.argv) > 2:
        classification_method = sys.argv[2]
    if len(sys.argv) > 3:
        task = sys.argv[3]
    if len(sys.argv) > 4:
        load_model_name = sys.argv[4]

    main(heldout_testset, classification_method, task, load_model_name)