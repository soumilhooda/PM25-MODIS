import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU 
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
import random
from scipy.stats import linregress
import seaborn as sns
from sklearn.metrics import r2_score

# Define PrintEpochProgress class to print only every 10th epoch info
class PrintEpochProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1} - {logs}")

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

np.random.seed(42)

df1 = pd.read_csv("terra-0.001.ascii", delim_whitespace=True, header=None)
df2 = pd.read_csv("terra-0.002.ascii", delim_whitespace=True, header=None)
df3 = pd.read_csv("terra-0.005.ascii", delim_whitespace=True, header=None)
df4 = pd.read_csv("terra-0.01.ascii", delim_whitespace=True, header=None)

df5 = pd.read_csv("aqua-0.001.ascii", delim_whitespace=True, header=None)
df6 = pd.read_csv("aqua-0.002.ascii", delim_whitespace=True, header=None)
df7 = pd.read_csv("aqua-0.005.ascii", delim_whitespace=True, header=None)
df8 = pd.read_csv("aqua-0.01.ascii", delim_whitespace=True, header=None)

# Change whichever file to be used to data and keep others set as datas, these are combined aqua+terra files
datas = pd.read_csv("fort.999", delim_whitespace=True, header=None)
datas = pd.concat([df1, df5], axis=0, ignore_index=True)
datas = pd.concat([df2, df6], axis=0, ignore_index=True)
datas = pd.concat([df3, df7], axis=0, ignore_index=True)
datas = pd.concat([df4, df8], axis=0, ignore_index=True)

# If you want to chose single aqua or terra files, set data as df here
data = df4 

feature_names = ['PM25', 'RAD1', 'RAD2', 'RAD3', 'RAD4', 'RAD5', 'RAD6', 'RAD7', 'RAD17', 'RAD18', 'RAD19', 'RAD20', 'RAD21', 'RAD22', 'RAD23', 'RAD24', 'RAD25','RAD26', 'RAD27', 'RAD28','RAD29', 'RAD30', 'RAD31','RAD32', 'RAD33', 'RAD34', 'RAD35', 'RAD36', 'SATZEN', 'SATAZI','SOLZEN', 'SOLAZI', 'NDVI', 'TPW', 'ALAT', 'ALON', 'TER', 'MON','DAY', 'HR', 'DIST']
data.columns = feature_names

columns_to_check = ['RAD2', 'RAD7', 'RAD23', 'RAD31']
# Create a boolean mask for rows with any negative value in the specified columns
mask_neg_values = (data[columns_to_check] >= 0).all(axis=1)
# Create a boolean mask for rows where 'PM25' is above 500
# mask_pm25 = (data['PM25'] <= 400)
# Combine the masks using logical AND to get the final mask
# final_mask = mask_neg_values & mask_pm25
final_mask = mask_neg_values 
# Apply the final mask to filter the DataFrame
data = data[final_mask]


# Prepare data
X = data.drop(['PM25', 'RAD1',  'RAD3', 'RAD5', 'RAD4',  'RAD6',  'RAD17', 'RAD18', 'RAD19', 'RAD20', 'RAD21', 'RAD22',  'RAD24', 'RAD25','RAD26', 'RAD27', 'RAD28','RAD29', 'RAD30', 'RAD32', 'RAD33', 'RAD34', 'RAD35', 'RAD36', 'SATZEN', 'SATAZI', 'NDVI', 'TPW', 'DIST'] ,axis=1)
# X = data.drop(['PM25', 'RAD1', 'RAD3', 'RAD4', 'RAD18', 'RAD19',  'RAD24', 'RAD26', 'RAD27', 'RAD28', 'RAD33', 'RAD34', 'RAD35', 'RAD36', 'SATZEN', 'SATAZI',  'NDVI', 'TPW', 'DIST'], axis=1)
# X = data.drop(['PM25','SATZEN', 'SATAZI','SOLZEN', 'SOLAZI','NDVI', 'TPW','DIST'], axis=1)
y = data[['PM25']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GRU Model
gru_model = Sequential([
    GRU(units=64, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    Dense(units=1)
])

# Compile and train GRU Model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train_scaled[:, :, np.newaxis], y_train, epochs=1000, batch_size=25,
              validation_data=(X_test_scaled[:, :, np.newaxis], y_test),
              verbose=10, callbacks=[PrintEpochProgress(), early_stopping])

# Evaluate on both train and test sets
predictions_train = gru_model.predict(X_train_scaled[:, :, np.newaxis])
predictions_test = gru_model.predict(X_test_scaled[:, :, np.newaxis])

# Calculate R^2 values
r2_train_gru = r2_score(y_train, predictions_train)
r2_test_gru = r2_score(y_test, predictions_test)

print(f"\nGRU Model Train R^2: {r2_train_gru:.4f}")
print(f"GRU Model Test R^2: {r2_test_gru:.4f}")

# Scatter plot for train data
plt.figure(figsize=(6, 6))
sns.scatterplot(x=np.ravel(y_train), y=np.ravel(predictions_train))
plt.title('GRU Model - Train Data')
plt.xlabel('Actual PM25')
plt.ylabel('Predicted PM25')

# Add zero line
plt.plot([min(np.ravel(y_train)), max(np.ravel(y_train))], [min(np.ravel(y_train)), max(np.ravel(y_train))], linestyle='--', color='red')

# Linear fit and confidence interval
slope_train, intercept_train, _, _, std_err_train = linregress(np.ravel(y_train), np.ravel(predictions_train))
line_train = slope_train * np.ravel(y_train) + intercept_train
plt.plot(np.ravel(y_train), line_train, color='green', label='Linear Fit')

# 95% Confidence Interval
ci_train = 1.96 * std_err_train  # 1.96 is the Z-score for a 95% confidence interval
plt.fill_between(np.ravel(y_train), line_train - ci_train, line_train + ci_train, color='black', alpha=0.2, label='95% CI')

plt.legend()
plt.show()

# Scatter plot for test data
plt.figure(figsize=(6, 6))
sns.scatterplot(x=np.ravel(y_test), y=np.ravel(predictions_test))
plt.title('GRU Model - Test Data')
plt.xlabel('Actual PM25')
plt.ylabel('Predicted PM25')

# Add zero line
plt.plot([min(np.ravel(y_test)), max(np.ravel(y_test))], [min(np.ravel(y_test)), max(np.ravel(y_test))], linestyle='--', color='red')

# Linear fit and confidence interval
slope_test, intercept_test, _, _, std_err_test = linregress(np.ravel(y_test), np.ravel(predictions_test))
line_test = slope_test * np.ravel(y_test) + intercept_test
plt.plot(np.ravel(y_test), line_test, color='green', label='Linear Fit')

# 95% Confidence Interval
ci_test = 1.96 * std_err_test  # 1.96 is the Z-score for a 95% confidence interval
plt.fill_between(np.ravel(y_test), line_test - ci_test, line_test + ci_test, color='black', alpha=0.2, label='95% CI')

plt.legend()
plt.show()

### Train and Test Stats + Outliers Print ###

y_stats = np.array(y_train)

# Basic statistics train
mean_value = np.mean(y_stats)
median_value = np.median(y_stats)
std_deviation = np.std(y_stats)
min_value = np.min(y_stats)
max_value = np.max(y_stats)

print(f"Train Mean: {mean_value}")
print(f"Train Median: {median_value}")
print(f"Train Standard Deviation: {std_deviation}")
print(f"Train Minimum Value: {min_value}")
print(f"Train Maximum Value: {max_value}")

y_stats = np.array(y_test)

# Basic statistics test
mean_value = np.mean(y_stats)
median_value = np.median(y_stats)
std_deviation = np.std(y_stats)
min_value = np.min(y_stats)
max_value = np.max(y_stats)

print(f"Test Mean: {mean_value}")
print(f"Test Median: {median_value}")
print(f"Test Standard Deviation: {std_deviation}")
print(f"Test Minimum Value: {min_value}")
print(f"Test Maximum Value: {max_value}")

# outlier printing
errors = abs(y_test['PM25'].values - predictions_test)

error_df = pd.DataFrame({
    'Actual': y_test['PM25'].values,
    'Predicted': predictions_test,
    'Error': errors
})

# Sort the DataFrame by the 'Error' column in descending order
error_df_sorted = error_df.sort_values(by='Error', ascending=False)

# Display the resulting DataFrame to find outliers
print(error_df_sorted.head(25))  

