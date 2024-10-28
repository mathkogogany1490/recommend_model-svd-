from gensim import corpora
from gensim.models import LdaModel
from db.dbCtr import insert_data_into_table
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_topic_vector(distribution, num_topics=5):
    """토픽 분포를 벡터 형식으로 변환하고 float32를 float로 변환"""
    vector = np.zeros(num_topics, dtype=float)
    for topic, prob in distribution:
        vector[topic] = float(prob)  # float32를 float로 변환
    return vector


def find_similar_movies(movie_id, df, top_n=5):
    """특정 영화와 유사한 영화를 찾기 위한 함수"""
    vector = df[df['movie_id'] == movie_id]['topic_vector'].values[0]
    similarities = cosine_similarity([vector], list(df['topic_vector'].values))
    df['similarity'] = similarities[0]
    return df.sort_values(by='similarity', ascending=False).head(top_n)


def build_lda_model(corpus, dictionary, num_topics=5):
    """LDA 모델을 학습하는 함수"""
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
    return lda_model


def preprocess_data(ratings_path, tags_path):
    """평점과 태그 데이터를 병합하고 pickle 파일로 저장 및 불러오기"""
    ratings_col = ['user_id', "movie_id", "rating", "timestamp"]
    tags_col = ['user_id', "movie_id", "tag", "timestamp"]
    ratings_df = pd.read_csv(ratings_path, delimiter="::", engine="python", names=ratings_col)
    tags_df = pd.read_csv(tags_path, delimiter="::", engine="python", names=tags_col)
    merged_df = pd.merge(ratings_df, tags_df, on=['user_id', 'movie_id'], how="inner")

    with open("../data/merged.pkl", 'wb') as file:
        pickle.dump(merged_df, file)

    return merged_df


def load_merged_data(file_path):
    """병합된 데이터를 pickle 파일에서 로드"""
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def prepare_corpus_and_dictionary(merged_df):
    """태그 데이터를 이용해 LDA 모델 학습을 위한 코퍼스와 사전을 준비"""
    dictionary = corpora.Dictionary([tags.split() for tags in merged_df['tag']])
    corpus = [dictionary.doc2bow(tag.split()) for tag in merged_df['tag']]
    return corpus, dictionary


def add_topic_vectors_to_df(df, lda_model, corpus, num_topics=5):
    """LDA 모델을 이용해 토픽 분포와 토픽 벡터를 데이터프레임에 추가하고 문자열로 변환"""
    df['top_distribution'] = [lda_model.get_document_topics(doc) for doc in corpus]

    # `top_distribution`과 `topic_vector`를 JSON 문자열로 변환
    df['top_distribution'] = df['top_distribution'].apply(
        lambda x: json.dumps([(int(topic), float(prob)) for topic, prob in x]))
    df['topic_vector'] = df['top_distribution'].apply(
        lambda x: json.dumps(get_topic_vector(json.loads(x), num_topics=num_topics).tolist()))

    return df



if __name__ == "__main__":
    # 데이터 병합 및 로드
    # ratings_path = "../data/ratings.dat"
    # tags_path = "../data/tags.dat"
    # merged_df = preprocess_data(ratings_path, tags_path)

    # Pickle 파일에서 병합된 데이터 로드
    merged_df = load_merged_data("../data/merged.pkl")

    # LDA 모델 학습을 위한 코퍼스와 사전 준비
    corpus, dictionary = prepare_corpus_and_dictionary(merged_df)

    # LDA 모델 생성
    lda_model = build_lda_model(corpus, dictionary, num_topics=5)

    # 데이터프레임에 토픽 분포 및 벡터 추가
    merged_df = add_topic_vectors_to_df(merged_df, lda_model, corpus, num_topics=5)

    # 데이터베이스에 삽입
    insert_data_into_table(merged_df, "lda_review_model")
    exit()

    # print(merged_df['topic_vector'])
    similar_movies = find_similar_movies(1, merged_df)
    print(similar_movies[['rating', 'tag']].iloc[0:10])

