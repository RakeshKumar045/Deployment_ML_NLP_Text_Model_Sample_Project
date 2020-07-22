import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv", encoding="latin-1")

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
# X = df['message']
# y = df['label']
#
#
# df_data = df[["CONTENT", "CLASS"]]
# # Features and Labels
df_x = df['message']
df_y = df['label']

# Extract Feature With CountVectorizer
corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus)  # Fit the Data

pickle.dump(cv, open('tranform.pkl', 'wb'))  # Save the bag of model (BOW)

X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
# Naive Bayes Classifier

clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# Alternative Usage of Saved Model
filename = 'spam_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

print("run success")
