from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import os  # Import the os module to handle file paths

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')

def read_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        article = file.read()
    return article

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    words1 = [word.lower() for word in sent1]
    words2 = [word.lower() for word in sent2]

    all_words = list(set(words1 + words2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:
        if word not in stopwords:
            vector1[all_words.index(word)] += 1

    for word in words2:
        if word not in stopwords:
            vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stopwords)

    return similarity_matrix

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    if request.method == 'POST':
        file_path = request.form['file_path']  # Retrieve file path from form data
        top_n = int(request.form['top_n'])

        # Check if the file exists
        if os.path.exists(file_path):
            article = read_article(file_path)
            sentences = nltk.sent_tokenize(article)
            stop_words = stopwords.words('english')

            sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

            scores = np.zeros(len(sentences))
            for i in range(len(sentences)):
                scores[i] = sum(sentence_similarity_matrix[i])

            ranked_sentences = [sentences[i] for i in np.argsort(scores)[-top_n:]]
            summary = ' '.join(ranked_sentences)
        else:
            summary = f"File not found at path: {file_path}"

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
