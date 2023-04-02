# Basic imports
import pickle
import random
import pandas as pd

# Importing sentence transformer
from sentence_transformers import SentenceTransformer

# Importing classifier from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity

# Flask utils
from flask import Flask, request, render_template

# For production environment
from gevent import pywsgi

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'bert_transformer.pkl'

# Load your trained model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Printing that the model has loaded
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

# Loading jokes.tsv
df_jokes = pd.read_csv('jokes.tsv', delimiter='\t')
sentences = df_jokes['Joke'].to_list()

# Printing sentences have been loaded
print('Sentences Loaded')

# Loading sentence embeddings
with open('sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

# Printing sentence embeddings have been loaded
print('Sentence Embeddings Loaded')

def model_predict(my_query, model):
    my_query = model.encode(my_query)
    similarity_score = cosine_similarity([my_query], sentence_embeddings).flatten()

    # df = pd.DataFrame({"Sentence":sentences,"Similarity_Score":similarity_score })
    # df = df.sort_values(by=["Similarity_Score"], ascending=False)

    # result = df.head(50).sample(n=1)['Sentence'].iloc[0]

    sentence_scores = list(zip(sentences, similarity_score))

    # Sort the list of tuples by similarity_score
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    # Get a random sentence from the top 20
    result = random.choice([x[0] for x in sentence_scores[:20]])

    return result

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get query from post request
        my_query = next(request.form.items())[0]
        
        # Make prediction
        result = model_predict(my_query, model)
        return result
    return None


if __name__ == '__main__':
    # Uncomment when going for production
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()
    # app.run(debug=True)