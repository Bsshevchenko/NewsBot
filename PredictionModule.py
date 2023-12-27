import numpy as np
from gensim.models import KeyedVectors
import joblib
from ParseNewsModule import parse_news
       
class MeanEmbeddingVectorizer(object):
    """Get mean of vectors"""
    def __init__(self, model):
        self.word2vec = model
        self.dim = model.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec.get_vector(w)
                for w in words if w in self.word2vec] or
                [np.zeros(self.dim)], axis=0)
            for words in X])

# Загрузка Word2Vec модели
word_vectors = KeyedVectors.load('C:\\Users\\админ\\Desktop\\MyProject\\Neural_Networks_and_NLP\\Hometasks\\Telegramm_bot\\word2vec_model', mmap='r')

# Загрузка общей модели
loaded_pipe = joblib.load('C:\\Users\\админ\\Desktop\\MyProject\\Neural_Networks_and_NLP\\Hometasks\\Telegramm_bot\\news_classification_model.joblib')


def prediction(data):
    # Преобразование текста в векторы
    mean_embedding_vectorizer = MeanEmbeddingVectorizer(word_vectors)
    X_transformed = mean_embedding_vectorizer.transform(data['content_clean'].apply(str.split))

    # Предсказание категорий
    predicted_categories = loaded_pipe.named_steps['clf'].predict(X_transformed)

    # Добавление предсказанных категорий к датафрейму
    data['predict_topic'] = predicted_categories
    return data

if __name__ == '__main__':
    print(prediction(parse_news('2023-12-26')))

