import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df=pd.read_csv('Reviews.csv')
df = df.drop(columns=['Name'])
messages = df.Comment

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []

for i in messages:
  review = re.sub('[^a-zA-Z]', ' ', i)
  review = review.lower()
  review = review.split()
  review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

cv = CountVectorizer()
x= cv.fit_transform(corpus).toarray()

def isgood(x):
  if x>=3:return 1
  else: return 0

y = df.Rating.apply(isgood)
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)

# import pickle
# with open('transform_cv', 'wb') as f:
#   pickle.dump(cv, f)
# with open('model_pickle', 'rb') as f:
#     model=pickle.load(f)
  
# with open('transform_cv', 'rb') as t:
#      tf1 = pickle.load(t)

# text="Very bad watch with worst charging anf worst design and  less display quality"

# data=[text]
# vect = tf1.transform(data).toarray()
# print(len(vect[0]))
# model.predict(vect)

# y1=pd.get_dummies(df.Rating, drop_first=True)
# x_train1, x_test1,y_train1,  y_test1 = train_test_split(x, df.Rating, test_size=0.2) 
# model_2 = MultinomialNB()
# model_2.fit(x_train1, y_train1)
# y_pred=model_2.predict(x_test)
# accuracy_score(y_test1, y_pred)