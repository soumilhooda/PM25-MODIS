import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib


# Read dataframes
df_terra_001 = pd.read_csv("terra-0.001.ascii", delim_whitespace=True, header=None)
df_terra_002 = pd.read_csv("terra-0.002.ascii", delim_whitespace=True, header=None)
df_terra_005 = pd.read_csv("terra-0.005.ascii", delim_whitespace=True, header=None)
df_terra_01 = pd.read_csv("terra-0.01.ascii", delim_whitespace=True, header=None)

df_aqua_001 = pd.read_csv("aqua-0.001.ascii", delim_whitespace=True, header=None)
df_aqua_002 = pd.read_csv("aqua-0.002.ascii", delim_whitespace=True, header=None)
df_aqua_005 = pd.read_csv("aqua-0.005.ascii", delim_whitespace=True, header=None)
df_aqua_01 = pd.read_csv("aqua-0.01.ascii", delim_whitespace=True, header=None)

# Combine dataframes
df_combined_001 = pd.concat([df_aqua_001, df_terra_001], axis=0, ignore_index=True)
df_combined_002 = pd.concat([df_aqua_002, df_terra_002], axis=0, ignore_index=True)
df_combined_005 = pd.concat([df_aqua_005, df_terra_005], axis=0, ignore_index=True)
df_combined_01 = pd.concat([df_aqua_01, df_terra_01], axis=0, ignore_index=True)

datasets = {
    'df_terra_001': df_terra_001,
    'df_terra_002': df_terra_002,
    'df_terra_005': df_terra_005,
    'df_terra_01': df_terra_01,
    'df_aqua_001': df_aqua_001,
    'df_aqua_002': df_aqua_002,
    'df_aqua_005': df_aqua_005,
    'df_aqua_01': df_aqua_01,
    'df_combined_001': df_combined_001,
    'df_combined_002': df_combined_002,
    'df_combined_005': df_combined_005,
    'df_combined_01': df_combined_01,
}

feature_names = ['PM25', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'RAD5', 'RAD6', 'RAD7', 'RAD17', 'RAD18', 'RAD19', 'RAD20',
                 'RAD21', 'RAD22', 'RAD23', 'RAD24', 'RAD25', 'RAD26', 'RAD27', 'RAD28', 'RAD29', 'RAD30', 'RAD31',
                 'RAD32', 'RAD33', 'RAD34', 'RAD35', 'RAD36', 'SATZEN', 'SATAZI', 'SOLZEN', 'SOLAZI', 'NDVI', 'TPW',
                 'ALAT', 'ALON', 'TER', 'MON', 'DAY', 'HR', 'DIST']

def preprocess_data(data):
    data.columns = feature_names
    X = data.drop(['PM25', 'SATZEN', 'SATAZI', 'NDVI', 'TPW', 'ALAT', 'ALON', 'TER', 'MON', 'DAY', 'HR', 'DIST'], axis=1)
    y = data[['PM25']]
    return X, y

def train_random_forest(X_train, y_train, save_path):
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train, y_train.values.ravel())

    # Save the model
    joblib.dump(model, save_path)

    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        feature_names = X_train.columns
        sorted_idx = np.argsort(feature_importance)[::-1]

        print(f"\nFeature Importance for Random Forest Regressor:")
        for i, idx in enumerate(sorted_idx):
            print(f"{i + 1}. {feature_names[idx]}: {feature_importance[idx]}")

def train_ann(X_train_scaled, y_train, save_path):
    nn_model = Sequential([
        Dense(units=512, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(units=256, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1),
    ])

    nn_model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = nn_model.fit(X_train_scaled, y_train, epochs=300, batch_size=32, validation_split=0.2, verbose=2,
                           callbacks=[early_stopping])

    # Save the model
    nn_model.save(save_path)

def plot_scatter_save_fig(y_true, y_pred, dataset_type, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([0, 800], [0, 800], '--k')
    sns.regplot(x=y_true.values.flatten(), y=y_pred.flatten(), scatter=False, color='red', ax=ax)

    # Calculate and plot 95% confidence interval
    residuals = y_true - y_pred.flatten()
    ci_lower = np.percentile(residuals, 2.5)
    ci_upper = np.percentile(residuals, 97.5)
    ax.axhline(ci_lower, linestyle='--', color='red', linewidth=2)
    ax.axhline(ci_upper, linestyle='--', color='red', linewidth=2)

    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_title(f'{dataset_type} Scatter Plot')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.grid(True)

    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train predictions
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Test predictions
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTrain RMSE:", train_rmse)
    print("Train R-squared:", train_r2)
    print("\nTest RMSE:", test_rmse)
    print("Test R-squared:", test_r2)

    # Scatter plots with 95% confidence interval
    plot_scatter_save_fig(y_train, y_train_pred, "Train", "train_scatter.png")
    plot_scatter_save_fig(y_test, y_test_pred, "Test", "test_scatter.png")

    # Percentage RMSE error
    train_percentage_rmse = (train_rmse / np.mean(y_train.values)) * 100
    test_percentage_rmse = (test_rmse / np.mean(y_test.values)) * 100

    print("\nTrain Percentage RMSE Error:", train_percentage_rmse)
    print("Test Percentage RMSE Error:", test_percentage_rmse)


def main():
    print("Choose a dataset:")
    for i, dataset_name in enumerate(datasets.keys()):
        print(f"{i + 1}. {dataset_name}")

    dataset_choice = int(input("Enter the number of your choice: ")) - 1
    selected_dataset_name = list(datasets.keys())[dataset_choice]
    selected_dataset = datasets[selected_dataset_name]

    X, y = preprocess_data(selected_dataset)

    print("\nChoose a training scheme:")
    print("1. Train, validate, and test on splits from the same dataset.")
    print("2. Train on one dataset and test on another.")
    print("3. Train, validate, and test on a dataset, and also select another dataset for complete testing.")
    print("4. Train, validate, and test on multiple different datasets one after the other.")

    training_scheme = int(input("Enter the number of your choice: "))

    if training_scheme == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    elif training_scheme == 2:
        print("Choose another dataset for testing:")
        for i, dataset_name in enumerate(datasets.keys()):
            if dataset_name != selected_dataset_name:
                print(f"{i + 1}. {dataset_name}")

        test_dataset_choice = int(input("Enter the number of your choice: ")) - 1
        test_dataset_name = [name for i, name in enumerate(datasets.keys()) if i == test_dataset_choice][0]
        test_dataset = datasets[test_dataset_name]

        X_train, y_train = X, y
        X_test, y_test = preprocess_data(test_dataset)
    elif training_scheme == 3:
        print("Choose another dataset for complete testing:")
        for i, dataset_name in enumerate(datasets.keys()):
            if dataset_name != selected_dataset_name:
                print(f"{i + 1}. {dataset_name}")

        test_dataset_choice = int(input("Enter the number of your choice: ")) - 1
        test_dataset_name = [name for i, name in enumerate(datasets.keys()) if i == test_dataset_choice][0]
        test_dataset = datasets[test_dataset_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    elif training_scheme == 4:
        print("Choose datasets for training and testing:")
        for i, dataset_name in enumerate(datasets.keys()):
            print(f"{i + 1}. {dataset_name}")

        train_dataset_choice = int(input("Enter the number of your choice for training: ")) - 1
        train_dataset_name = [name for i, name in enumerate(datasets.keys()) if i == train_dataset_choice][0]
        train_dataset = datasets[train_dataset_name]

        for i, dataset_name in enumerate(datasets.keys()):
            if dataset_name != train_dataset_name:
                print(f"{i + 1}. {dataset_name}")

        test_dataset_choice = int(input("Enter the number of your choice for testing: ")) - 1
        test_dataset_name = [name for i, name in enumerate(datasets.keys()) if i == test_dataset_choice][0]
        test_dataset = datasets[test_dataset_name]

        X_train, y_train = preprocess_data(train_dataset)
        X_test, y_test = preprocess_data(test_dataset)
    else:
        print("Invalid choice. Exiting.")
        return

    print("\nChoose a regression model:")
    print("1. Random Forest Regressor")
    print("2. Artificial Neural Network (ANN)")

    model_choice = int(input("Enter the number of your choice: "))

    if model_choice == 1:
        save_path = f"{selected_dataset_name}_random_forest_model.joblib"
        train_random_forest(X_train, y_train, save_path)
        model = joblib.load(save_path)
    elif model_choice == 2:
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_test_scaled = StandardScaler().fit_transform(X_test)

        save_path = f"{selected_dataset_name}_ann_model.h5"
        train_ann(X_train_scaled, y_train, save_path)
        model = Sequential()
        model = model.load_model(save_path)
    else:
        print("Invalid choice. Exiting.")
        return

    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
