import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

#these are necessary libarray to visualize results

#making a small dataset

#3 features height, weight, age and gender


data = {
    'Height': [170, 165, 180, 175, 160, 172, 168,177, 162, 158],
    'weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
    'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
    'Gender':[1,0,1,1,0,1,0,1,0,0]
    #1 is male 0 is female
}

df = pd.DataFrame(data)
print(df)

#Standarizing the Data

#now all features have different units so we standarize the data. Mkaing all features have mean = 0 and stndard deviation  = 1
# now no fetaure will dominate other because of its units.

X = df.drop('Gender', axis = 1) #drop gender col
y = df['Gender'] #seperatly storing gender col


# x is input features and x is target/op

scaler = StandardScaler() #performs standarization
X_scaled = scaler.fit_transform(X) #fit and transfrom combined / learn the statistics ie mean = 0 and standard deviation = 1 and transfrom the data


#APPLYING PCA ALGORITHM

pca = PCA (n_components=2) #telling py that i want to reduce my data to 2 principle components PC1 PC2 these are the new two combination sof the original features
X_pca = pca.fit_transform(X_scaled)


#split the data to check whether our model can correctly predict Gender
#dividing into 70 training and 30 testig percentage
#test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# X is the pca features and y is the actual gender labels
#  
model = LogisticRegression() #creates the logistic regression classifier

#train the model -> actual learning happens
model.fit(X_train, y_train)

y_pred = model.predict(X_test) # x test is the 30% data that was not trained on
# we test on that 30% data


#EVALUATING WITH CONFUSION MATRIX

# The confusion matrix compares actual vs predicted labels. This makes it easy to see where predictions were correct or wrong.


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (5,4))
sns.heatmap(cm,  annot = True, fmt='d', cmap = 'Blues',
            xticklabels=['Female','Male'], yticklabels=['Female','Male'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


#VISUALIZING PCA RESULT

y_numeric = pd.factorize(y)[0]

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
plt.scatter(X_scaled[:, 0], X_scaled[:,1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Original Feature 1')
plt.ylabel('Original Feature 2')
plt.title('Before PCA: Using First 2 standardized Features')
plt.colorbar(label='Target classes')

plt.subplot(1,2,2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('After PCA: Projected onto 2 Principal Components')
plt.colorbar(label='Target classes')
plt.tight_layout()
plt.show()