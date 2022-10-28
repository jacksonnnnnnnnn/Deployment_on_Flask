import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
# load the csv file
df = pd.read_csv("iris.csv")


print(df.head())
# select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length" , "Petal_Width"]]
y = df["Class"]
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# make pickle file of our model
pickle.dump(rf_classifier, open("model.pkl", "wb"))


