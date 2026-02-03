import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("dataset.csv")


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]


model = MultinomialNB()
model.fit(X, y)


email = ["You have won a free prize"]
email_vector = vectorizer.transform(email)
result = model.predict(email_vector)

print("Email is:", result[0])
