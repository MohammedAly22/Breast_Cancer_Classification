import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('data.csv')


def dataSetAnalysis(df):
    # view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)

    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)

    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)

    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)

    # view distribution of categorical features across the data set
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 30)


dataSetAnalysis(dataset)

X = dataset.iloc[:, 2:32]  # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:, 1]  # [all rows, col one only which contains the classes of cancer]

print("Before encoding: ")
print(y[100:110])

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

print("\nAfter encoding: ")
print(y[100:110])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()  # Initialising the ANN

classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=1, epochs=100)

classifier.save('breast_cancer_model.h5')  # Save trained ANN

# classifier = load_model('breast_cancer_model.h5')  #Load trained ANN

y_pred = classifier.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy: " + str(accuracy * 100) + "%")

prediction = classifier.predict([[13.5, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]])
prediction = [1 if y >= 0.5 else 0 for y in prediction]
class_name = "Malignant" if prediction == 1 else "Benign"
print("Prediction is: ", class_name)
