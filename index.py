from flask import Flask, request, render_template
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import csv
import os
from chat import get_response
import openai
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'F:\Coding\Projects\Review Automation and Chatbot\chatbot-403615-c39643627579.json'

# define function that creates the bucket
def create_bucket(bucket_name, storage_class='STANDARD', location='us-central1'): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = storage_class
   
    bucket = storage_client.create_bucket(bucket, location=location) 
    # for dual-location buckets add data_locations=[region_1, region_2]
    
    return f'Bucket {bucket.name} successfully created.'

## Invoke Function
print(create_bucket('test_demo_storage_bucket', 'STANDARD', 'us-central1'))

app = Flask(__name__)
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)
with open('transform_cv', 'rb') as t:
    cv = pickle.load(t)
def preprocess_text(text):
    wordnet = WordNetLemmatizer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    return corpus
@app.route('/')
def welcome():
    return render_template('maindup.html')
@app.route('/signin', methods=['POST'])
def signin():
    csv_path = os.path.join('static', 'validemailpasswords.csv')
    email = request.form['email']
    with open(csv_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if email in row:
                return render_template('signup.html', error="Email already exists!")
    with open(csv_path, mode='a') as csvfile:
        writer = csv.writer(csvfile)
        email = request.form['email']
        crepassword = request.form['crepassword']
        writer.writerow([email, crepassword])
    return render_template('logup.html')
@app.route('/signup')
def signup():
    return render_template('signup.html')
@app.route('/submitsignin', methods=['POST'])
def submitsignin():
    email = request.form['email']
    return render_template('main.html', id=email)
@app.route('/home')
def home():
    return render_template('main.html')
@app.route('/logup')
def logup():
    return render_template('logup.html')
@app.route('/logout')
def logout():
    return render_template('maindup.html')
@app.route('/flipauto')
def flipauto():
    return render_template('flipauto.html')
@app.route('/submit', methods=['POST'])
def pr():
    if request.method == 'POST':
        text = request.form['text']
        text = str(text)
        corpus = preprocess_text(text)
        vect = cv.transform(corpus).toarray()
        ans = model.predict(vect)
        res = int(ans[0])
        if res==0:
            rating="Bad"
        else:
            rating="Good"
        return render_template('pr.html', rating=rating, text=text)
responses = {"user":["Hello"],
              "bot":["Hi How can i help you?"]}
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html',responses=responses)
@app.route('/get_response', methods=['POST'])
def chatbot_response():
    message = request.form['message']
    if message == 'exit':
         responses.clear()
         responses['bot'] = ['Hi How can i help you?']
         responses['user'] = ['Hello']
    else:
        response = get_response(message)
        print(response)
        responses['user'].append(message)
        responses['bot'].append(response)
        if len(responses['bot']) > 8:
            responses['bot'].pop(0)
        if len(responses['user']) > 8:
            responses['user'].pop(0)
        if message == 'exit':
            responses.clear()
            responses['bot'] = ['Hi How can i help you?']
            responses['user'] = ['Hello']
    return render_template("chatbot.html",responses=responses)
if __name__ == '__main__':
    app.run(debug=True)
