import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Initialize Firebase Admin SDK
cred = credentials.Certificate("unfolding-aspirants-firebase-adminsdk-pb6dd-dce071f97b.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Fetch data from Firestore
def fetch_data_from_firestore(collection_name):
    docs = db.collection(collection_name).stream()
    data = []
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data['id'] = doc.id  # Add document ID to the data
        data.append(doc_data)
    return pd.DataFrame(data)

events_data = fetch_data_from_firestore('events')
courses_data = fetch_data_from_firestore('courses')
books_data = fetch_data_from_firestore('books')
users_data = fetch_data_from_firestore('users')

# Preprocess the text data
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

events_data['Description'] = events_data['Description'].apply(preprocess_text)
courses_data['Description'] = courses_data['Description'].apply(preprocess_text)
books_data['Description'] = books_data['Description'].apply(preprocess_text)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
all_descriptions = pd.concat([events_data['Description'], courses_data['Description'], books_data['Description']])
vectorizer.fit(all_descriptions)

events_features = vectorizer.transform(events_data['Description'])
courses_features = vectorizer.transform(courses_data['Description'])
books_features = vectorizer.transform(books_data['Description'])


# Recommendation Function
def get_recommendations(user_doc_id):
    user_doc = db.collection('users').document(user_doc_id).get()
    if not user_doc.exists:
        raise ValueError("User document does not exist")

    user_data = user_doc.to_dict()
    skills_text = ' '.join(user_data['skills'])
    interests_text = ' '.join(user_data['interests'])
    user_features = vectorizer.transform([skills_text + ' ' + interests_text])

    scores = []
    for item_type, item_features, item_data in [('events', events_features, events_data),
                                                ('courses', courses_features, courses_data),
                                                ('books', books_features, books_data)]:
        similarities = cosine_similarity(user_features, item_features).flatten()
        scores.extend([(item_type, item_data.iloc[idx]['id'], sim) for idx, sim in enumerate(similarities)])

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:5]

# Example Usage
user_doc_id = input('Enter User Document ID: ')
recommendations = get_recommendations(user_doc_id)
for item_type, doc_id, similarity in recommendations:
    print(f"{item_type.title()} Document ID - {doc_id}, Similarity Score: {similarity:.2f}")
