from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_topic_vector(x, topics=5):
    vector = np.zeros(topics)
    for topic, prob in x:
        vector[topic] = prob
    return vector


def find_similar_movies(movie_id, df, top=5):
    vector = df[df['movie_id']==movie_id]['topic_vector'].values[0]
    similarities = cosine_similarity([vector], list(df['topic_vector'].values))
    df['similarity'] = similarities[0]
    return df.sort_values(by='similarity', ascending=False)




if __name__ == "__main__":
    # ratings_path = "../data/ratings.dat"
    # tags_path = "../data/tags.dat"
    # ratings_col = ['user_id', "movie_id", "rating", "timestamp"]
    # tags_col = ['user_id', "movie_id", "tag", "timestamp"]
    # ratings_df = pd.read_csv(ratings_path, delimiter="::", engine="python", names=ratings_col)
    # tags_df = pd.read_csv(tags_path, delimiter="::", engine="python", names=tags_col)
    # # print(tags_df)
    #
    # merged_df = pd.merge(ratings_df, tags_df, on=['user_id', 'movie_id'], how="inner")
    # with open("../data/merged.pkl", 'wb+') as file:
    #     pickle.dump(merged_df, file)
    with open("../data/merged.pkl", 'rb') as file:
        merged_df = pickle.load(file)


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(merged_df['tag'])
    dictionary = corpora.Dictionary([tags.split() for tags in merged_df['tag']])
    corpus = [dictionary.doc2bow(tag.split()) for tag in merged_df['tag']]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42, passes=10)

    merged_df['top_distribution'] = [lda_model.get_document_topics(corp) for corp in corpus]
    # print(merged_df['top_distribution'])
    merged_df['topic_vector'] = merged_df['top_distribution'].apply(lambda x:get_topic_vector(x))
    # print(merged_df['topic_vector'])
    similar_movies = find_similar_movies(1, merged_df)
    print(similar_movies[['rating', 'tag']].iloc[0:10])

