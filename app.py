from flask import Flask
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# @app.route('/')
def hello_world():
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load your Q&A data from a JSON file
    with open('your_qa_data.json', 'r') as file:
        qa_data = json.load(file)

    # Preprocessing function
    def preprocess(text):
        # Tokenization
        tokens = word_tokenize(text.lower())  # Convert to lowercase for consistency

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    # Apply preprocessing to each Q&A pair
    preprocessed_data = []
    for pair in qa_data:
        question = preprocess(pair['question'])
        answer = preprocess(pair['answer'])
        preprocessed_data.append({'question': question, 'answer': answer})

    # Print preprocessed data for the first Q&A pair
    print("Original Question:", qa_data[0]['question'])
    print("Preprocessed Question:", preprocessed_data[0]['question'])
    print("\nOriginal Answer:", qa_data[0]['answer'])
    print("Preprocessed Answer:", preprocessed_data[0]['answer'])

    # Test case: Apply preprocessing to a custom example
    custom_example = "How can I learn machine learning?"
    preprocessed_example = preprocess(custom_example)
    print("\nCustom Example:", custom_example)
    print("Preprocessed Example:", preprocessed_example)
hello_world()
print("end")