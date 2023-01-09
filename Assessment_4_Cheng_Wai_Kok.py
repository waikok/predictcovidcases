#%%
#1. Import packages
import matplotlib.pyplot as plt
import numpy as np
import os,datetime
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# %%
#2. Set parameters
BATCH_SIZE = 64
dropout_rate = 0.3
EPOCHS = 100
SEED = 123
train_filename = 'cases_malaysia_train.csv'
test_filename = 'cases_malaysia_test.csv'
window_size = 30

#%%
#3. Data loading
FILE_PATH = os.path.join(os.getcwd(),'dataset',train_filename)
covid_df = pd.read_csv(FILE_PATH, sep=',')

# %%
#4. Data Inspection
print("The data has {} daily covid record".format(len(covid_df)))
print("DataFrame info:")
print(covid_df.info())
print("Number of NA values: ")
print(covid_df.isna().sum())
print("Numer of complete duplicates:")
print(covid_df.duplicated().sum())
print("First 5 rows of the Data\n")
print(covid_df.head())
print("Description of the DataFrame:")
print(covid_df.describe().T)

# %%
#5. Data preprocessing
"""Observation:
        a) Some data in clusters are missing
        b) Some value in cases_new are missing
        c) Data is not consistent, e.g. cases_child + cases_adolescent + cases_adult + cases_elderly not equal to cases_new
"""

#5.1 Data Cleaning
# Dropping unnecessary data to improve accuracy and performance
drop_labels = ['cluster_import','cluster_religious','cluster_community','cluster_highRisk','cluster_education','cluster_detentionCentre','cluster_workplace']
covid_df.drop(drop_labels,axis=1,inplace=True)

#%%
#5.2 Convert object into numeric
covid_df['cases_new'] = pd.to_numeric(covid_df['cases_new'],errors='coerce')

#%%
#5.3 Update the Missing Data/? in cases_new with total number of cases_child + cases_adolescent + cases_adult + cases_elderly
for i, row in covid_df.iterrows():
    if pd.isnull(row['cases_new']) or row['cases_new'] in [0, '?']:
        covid_df.at[i, 'cases_new'] = row['cases_child'] + row['cases_adolescent'] + row['cases_adult'] + row['cases_elderly']

#%%
#5.4 Split the data and target from the dataset
covid_data = covid_df.drop('cases_new',axis=1).values
covid_target = covid_df['cases_new'].values

new_cases = covid_df['cases_new'].values
mms = MinMaxScaler()
new_cases = mms.fit_transform(new_cases.reshape(-1,1))

# %%
#6. Train test split
#6.1 Select the feature and target
X = []
y = []

for i in range(window_size,len(new_cases)):
    X.append(new_cases[i-window_size:i])
    y.append(new_cases[i])

X = np.array(X)
y = np.array(y)

#6.2 Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = SEED)

# %%
#7. Model creation
model = keras.Sequential()
model.add(keras.layers.LSTM(64, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dropout(dropout_rate))
model.add(keras.layers.Dense(16,activation='relu'))
model.add(keras.layers.Dropout(dropout_rate))
model.add(keras.layers.Dense(8,activation='relu'))
model.add(keras.layers.Dropout(dropout_rate))
model.add(keras.layers.Dense(1,activation='relu'))

model.summary()

# %%
#8. Model compilation
model.compile(optimizer='adam',loss='mse',metrics=['acc'])

# Plot the architecture model
keras.utils.plot_model(model,show_shapes=True)

# %%
#9. Define the callback function to use
#TensorBoard callback
log_path = os.path.join('log_dir',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = keras.callbacks.TensorBoard(log_path)

# %%
#10. Perform model training
history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb])

#%%
#11. Model Analysis/Evaluation
TEST_CSV_PATH = os.path.join(os.getcwd(),'dataset',test_filename)
covid_df = pd.read_csv(TEST_CSV_PATH, sep=',')

# To concatenate the data
test_df = pd.read_csv(TEST_CSV_PATH)
concat = pd.concat((covid_df['cases_new'],test_df['cases_new']))
concat = concat[len(covid_df['cases_new'])-window_size:]

# Data normalization (min max transformation)
concat = mms.transform(concat[::,None]) # 0 to 1

Xtest_test = []
ytest_test = []
 
for i in range(window_size,len(concat)):
    Xtest_test.append(concat[i-window_size:i])
    ytest_test.append(concat[i])

Xtest_test = np.array(Xtest_test) # to convert into array
ytest_test = np.array(ytest_test)

predicted_cases = model.predict(Xtest_test) # to predict the unseen dataset

#%%
#12. Visualize the actual and predicted cases

plt.figure()
plt.plot(predicted_cases,color='red')
plt.plot(ytest_test,color='blue')
plt.legend(['Predicted','Actual'])
plt.xlabel('time')
plt.ylabel('New Covid Cases')
plt.show()

#%%
#12.1 Actual vs predicted
print(mean_absolute_percentage_error(ytest_test,predicted_cases)) 
print(mean_absolute_error(ytest_test,predicted_cases))
print(mean_squared_error(ytest_test,predicted_cases))

#%%
#13. Model Savings
model.save('saved_model/model.h5')


# %%
