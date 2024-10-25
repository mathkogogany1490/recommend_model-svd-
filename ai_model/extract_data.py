import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from db.dbCtr import *

import pandas as pd


def extract_movie_data():
    names = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('../data/movies.dat',
                         names=names, sep='::', engine='python', encoding='latin-1')

    # 각 영화의 장르를 리스트로 변환
    movies.genres = movies.genres.apply(lambda x: x.split('|'))

    # 유니크한 장르를 모두 추출
    unique_genres = set(genre for genres in movies.genres for genre in genres)

    return movies, unique_genres


movies, unique_genres = extract_movie_data()
print("Unique genres:", unique_genres)


def extract_user_data():
    names = ['user_id', "movie_id", 'rating', 'timestamp']
    ratings = pd.read_csv('../data/ratings.dat', names=names, sep='::', engine='python', encoding='latin-1')
    unique_users = ratings['user_id'].unique()
    customers = pd.DataFrame(unique_users, columns=['user_id'])
    return customers

def extract_rating_data():
    names = ['user_id', "movie_id", 'rating', 'timestamp']
    ratings = pd.read_csv('../data/ratings.dat', names=names, sep='::', engine='python', encoding='latin-1')
    ratings['id'] = list(range(len(ratings)))
    # UNIX timestamp를 MySQL에서 사용 가능한 datetime 형식으로 변환
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings = ratings[['id', 'user_id', "movie_id", 'rating', 'timestamp']]
    # print(ratings.keys())
    # exit()
    return ratings

def extract_tag_data():
    names = ['user_id', "movie_id", 'tag', 'timestamp']
    tags = pd.read_csv('../data/tags.dat',
                       names=names, sep='::', engine='python', encoding='latin-1')
    tags['id'] = list(range(len(tags)))
    # UNIX timestamp를 MySQL에서 사용 가능한 datetime 형식으로 변환
    tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
    tags = tags[['id', 'user_id', "movie_id", 'tag']]
    # print(ratings.keys())
    # exit()
    return tags


def insert_data_into_table(df, table):
    # connecting fields
    fields = postgres_connect_field()  # Assuming this retrieves PostgreSQL connection info

    # PostgreSQL 연결 문자열 생성 (SQLAlchemy에서 psycopg2 사용)
    conn = f"postgresql+psycopg2://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"

    # 예외 처리를 통한 안전한 PostgreSQL 연결 및 데이터 삽입
    try:
        # with 문으로 engine 자동 종료 처리
        with create_engine(conn).connect() as connection:

            # DataFrame 데이터를 PostgreSQL 테이블에 삽입
            df.to_sql(table, con=connection, if_exists='append', chunksize=1000, index=False)
            print(f"Data inserted successfully into table {table}.")
    except SQLAlchemyError as e:
        # 에러 처리 및 로그 출력
        print(f"Error while inserting data: {e}")
    else:
        print("Data inserted and connection closed!!!")



if __name__ == "__main__":
    _, unique_genres = extract_movie_data()
    print(unique_genres)
    exit()
    # insert_data_into_table(customers, 'customers')
    ratings = extract_rating_data()
    insert_data_into_table(ratings, "ratings")
    # ratings = bring_dataframe_from_table('ratings')
    # print(ratings)
    # tags = extract_tag_data()
    # insert_data_into_table(tags, 'tags')
    # movies = extract_movie_data()
    # print(movies)
    # movies['id'] = list(range(len(movies)))
    # movies = movies[['id', 'movie_id', 'title', 'genres']]
    # insert_data_into_table(movies, 'recommend_movies')


