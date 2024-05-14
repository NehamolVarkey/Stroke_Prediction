# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# importing dataset
data = pd.read_csv(r"C:\DataScience\MyProjects\StrokePrediction\stroke_prediction_dataset.csv")
data = pd.DataFrame(data)
print(data.columns)

# renaming the columns
df = data.rename(columns={'Residence_type':'residence_type'})
print(df.head())

# dropping unnecessary columns
df.drop(columns='id', inplace=True)

# To generate summary statistics for the numerical columns in a Data.
print(df.info())

# checking the percentage of missing values
print((df.isna().sum()/len(data))*100)

# dropping the rows that contains the missing values
df.dropna(how='any', inplace=True)

# Checking for null values.
print(df.isnull().sum())

# Checking for duplicate values.
print(df.duplicated().sum())


# Correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidths=0.5,cmap="Greens")
plt.title('Heatmap')
plt.show()


# Feature Scaling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
categorical_cols = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
# preprocessing numerical and categorical columns
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
# preprocessor for both numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# Extracting Independent and dependent Variable
X = df.drop(columns=['stroke'])
y = df['stroke']

x = preprocessor.fit_transform(X)


# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# Logistic Regression Model
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

start_time = time.time()

logreg = LogisticRegression(solver='lbfgs', max_iter=10)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

accuracy_logreg = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_logreg)

end_time = time.time()

# Calculate execution time
logreg_time = end_time - start_time
print("Execution time:", logreg_time, "seconds")


# Support Vector Classifier
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

start_time = time.time()

svc = SVC(kernel='linear', random_state=0)
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

accuracy_svc = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_svc)

end_time = time.time()

# Calculate execution time
svc_time = end_time - start_time
print("Execution time:", svc_time, "seconds")



# DecisionTree Classifier
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

start_time = time.time()

dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)

accuracy_dtree = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_dtree)

end_time = time.time()

# Calculate execution time
dtree_time = end_time - start_time
print("Execution time:", dtree_time, "seconds")



# KNeighbors Classifier
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

start_time = time.time()

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy_knn = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_knn)

end_time = time.time()

# Calculate execution time
knn_time = end_time - start_time
print("Execution time:", knn_time, "seconds")



# Random Forest Classifier
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

start_time = time.time()

ranf = RandomForestClassifier(criterion='entropy', n_estimators=10)
ranf.fit(x_train, y_train)

y_pred = ranf.predict(x_test)

accuracy_ranf = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_ranf)

end_time = time.time()

# Calculate execution time
ranf_time = end_time - start_time
print("Execution time:", ranf_time, "seconds")

# checking which model is best
result = pd.DataFrame({
    'Algorithm' : ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier'],
    'Accuracy' : [accuracy_logreg, accuracy_svc, accuracy_dtree, accuracy_knn, accuracy_ranf],
    'Execution_time' : [logreg_time, svc_time, dtree_time, knn_time, ranf_time]
})
print(result)


# plotting the accuracy and time taken by each model
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x='Algorithm', y='Accuracy', data=result, palette='viridis', ax=ax[0])
ax[0].bar_label(ax[0].containers[0], fmt='%.3f')
ax[0].set_title('Accuracy of Classification Algorithms')
ax[0].set_xticklabels(labels=result.Algorithm, rotation=45)

sns.barplot(x='Algorithm', y='Execution_time', data=result, palette='viridis', ax=ax[1])
ax[1].bar_label(ax[1].containers[0], fmt='%.3f')
ax[1].set_title('Execution Time of Classification Algorithms')
ax[1].set_xticklabels(labels=result.Algorithm, rotation=45)
plt.show()
