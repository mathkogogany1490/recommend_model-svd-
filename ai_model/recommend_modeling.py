import pandas as pd
import numpy as np
from db.dbCtr import *
from ai_model.extract_data import *
import scipy
import ast
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset as SurpriseDataset, Reader
from implicit.als import AlternatingLeastSquares
from collections import defaultdict

# 평균 제곱 오차
def MSE(origins, predicts):
    return np.sqrt(mean_squared_error(origins, predicts))
# 재현율
def recall_at_k(origins, predicts, k):
    scores = []
    for user in origins.keys():
        origin = origins[user]
        predict = predicts[user]
        if len(origin) == 0 or k == 0:
            return 0.0
        recall = (len(set(origin) & set(predict[:k]))) / len(origin)
        scores.append(recall)
    return np.mean(scores)
# 정밀도
def precision_at_k(origins, predicts, k):
    scores = []
    for user in origins.keys():
        origin = origins[user]
        predict = predicts[user]
        if k == 0:
            return 0.0
        precision = (len(set(origin) & set(predict[:k]))) / k
        scores.append(precision)
    return np.mean(scores)
def save_to_mysql(df, table_name):
    try:
        # Get database connection credentials from the connect_field() function
        fields = connect_field()

        # Create the MySQL connection string
        conn_str = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"

        # Create the SQLAlchemy engine
        engine = create_engine(conn_str)

        # Save the DataFrame to the specified MySQL table
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=True)

        print(f"DataFrame successfully saved to table '{table_name}' in the MySQL database.")

    except SQLAlchemyError as e:
        print(f"An error occurred while saving the DataFrame to MySQL: {e}")

    finally:
        # Ensure the connection is closed
        engine.dispose()
        print("Database connection closed.")
def save_pred_model_to_mysql(pred_model_df, table_name, chunk_size=5000):
    # MySQL 연결 정보 가져오기
    fields = connect_field()
    # MySQL 연결 문자열 생성
    conn_str = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"
    # SQLAlchemy 엔진 생성
    engine = create_engine(conn_str)
    try:
        # DataFrame을 행 기반 구조로 변환
        pred_model_melted = pred_model_df.reset_index().melt(id_vars='user_id', var_name='movie_id',
                                                             value_name='predicted_rating')

        # 데이터를 chunk로 나눔
        chunks = [pred_model_melted[i:i + chunk_size] for i in range(0, pred_model_melted.shape[0], chunk_size)]

        # tqdm을 사용하여 진행 상황 표시
        for i, chunk in enumerate(tqdm(chunks, desc="Inserting data into MySQL")):
            chunk.to_sql(name=table_name, con=engine, if_exists='append', index=False)
            print(f"Chunk {i + 1}/{len(chunks)} inserted successfully.")

        print(f"Data inserted successfully into '{table_name}' table.")

    except SQLAlchemyError as e:
        print(f"Error while inserting data: {e}")
    finally:
        # 연결 해제
        engine.dispose()
def get_movie_ids(user_id):
    fields = connect_field()
    conn = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"
    engine = create_engine(conn)
    query = f"select movie_ids from customers where user_id={user_id}"
    movie_ids = pd.read_sql(query, engine)['movie_ids'].to_list()
    return movie_ids

# 인기도 평가 모델
def popularity_predict_model(min_rating_num):
    train, test = train_test_split()
    # movie_id로 그룹화하여 rating의 size와 mean을 계산
    movie_stats = train.groupby('movie_id').agg(size=('rating', 'size'), mean=('rating', 'mean'))
    # 최소 평점 수 이상의 영화만 선택
    flags = movie_stats['size'] >= min_rating_num
    # 조건에 맞는 데이터를 평점 평균으로 내림차순 정렬
    pop_movies_sorted = movie_stats[flags].sort_values(by='mean', ascending=False)
    # movie_id를 열로 다시 추가 (reset_index)
    pop_movies_sorted = pop_movies_sorted.reset_index()
    return pop_movies_sorted
def save_to_database(df, table_name="popular_movies"):
    # MySQL 연결 정보 가져오기
    fields = connect_field()
    # MySQL 연결 문자열 생성
    conn = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"
    # SQLAlchemy 엔진 생성
    engine = create_engine(conn)
    try:
        # DataFrame을 MySQL 테이블에 삽입
        df.to_sql(table_name, engine, if_exists='replace', index=True)
    except SQLAlchemyError as e:
        print(f"Error while inserting data: {e}")
    finally:
        # 연결 해제
        engine.dispose()
# 특이값 분해 (차원 축소): 노이즈 많아요, 모르는 값이 많아요
def svd_predict_model(degree):
    train, test = train_test_split()
    # 사용자 또는 영화별로 평가 수가 많은 것들만 필터링
    popular_users = train['user_id'].value_counts().index[:1000]  # 예시: 상위 1000명의 사용자만 사용
    popular_movies = train['movie_id'].value_counts().index[:500]  # 예시: 상위 500개의 영화만 사용
    # 필터링된 사용자 및 영화만 남기기
    train_filtered = train[train['user_id'].isin(popular_users) & train['movie_id'].isin(popular_movies)]

    matrix = train_filtered.pivot(index='user_id', columns='movie_id', values='rating').astype(np.float32)

    # Fill NaN values with the mean rating
    mean = train.rating.mean().astype(np.float32)
    matrix_np = matrix.fillna(mean).to_numpy()

    P, S, Qt = scipy.sparse.linalg.svds(matrix_np, k=degree)
    print("고윳값", S)
    pred_model = P@np.diag(S)@Qt

    # print("예측모델", pred_model[0, :])
    pred_model_df = pd.DataFrame(pred_model, index=matrix.index, columns=matrix.columns)

    return pred_model_df.iloc[:100, :]
def nmf_predict_model(degree):
    train, test = train_test_split()

    # 사용자 또는 영화별로 평가 수가 많은 것들만 필터링
    popular_users = train['user_id'].value_counts().index[:1000]  # 예시: 상위 1000명의 사용자만 사용
    popular_movies = train['movie_id'].value_counts().index[:500]  # 예시: 상위 500개의 영화만 사용

    # 필터링된 사용자 및 영화만 남기기
    train_filtered = train[train['user_id'].isin(popular_users) & train['movie_id'].isin(popular_movies)]

    # Create user-movie rating matrix
    matrix = train_filtered.pivot(index='user_id', columns='movie_id', values='rating').astype(np.float16)

    # Fill NaN values with the mean rating
    mean = train_filtered['rating'].mean().astype(np.float16)
    matrix_np = matrix.fillna(mean).to_numpy(dtype=np.float16)

    # Scale the data
    scaler = StandardScaler()
    matrix_np = scaler.fit_transform(matrix_np)

    # Ensure the matrix is non-negative (replace any negative values with zero)
    matrix_np = np.maximum(matrix_np, 0)

    # Initialize NMF model with the specified number of components (degree)
    nmf = NMF(n_components=degree, random_state=42, init='random', max_iter=500, tol=1e-5)

    # Fit the NMF model to the entire matrix
    P = nmf.fit_transform(matrix_np)  # User-feature matrix
    Q = nmf.components_  # Feature-movie matrix

    # Reconstruct the predicted matrix
    pred_model = P @ Q  # Matrix multiplication to get the predicted ratings

    # Convert the prediction back to a DataFrame for interpretability
    pred_model_df = pd.DataFrame(pred_model, index=matrix.index, columns=matrix.columns)

    return pred_model_df.iloc[:100, :]
def save_pred_model_to_mysql(pred_model_df, table_name, chunk_size=5000):
    # MySQL 연결 정보 가져오기
    fields = connect_field()
    # MySQL 연결 문자열 생성
    conn_str = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"
    # SQLAlchemy 엔진 생성
    engine = create_engine(conn_str)
    try:
        # DataFrame을 행 기반 구조로 변환
        pred_model_melted = pred_model_df.reset_index().melt(id_vars='user_id', var_name='movie_id',
                                                             value_name='predicted_rating')

        # 데이터를 chunk로 나눔
        chunks = [pred_model_melted[i:i + chunk_size] for i in range(0, pred_model_melted.shape[0], chunk_size)]

        # tqdm을 사용하여 진행 상황 표시
        for i, chunk in enumerate(tqdm(chunks, desc="Inserting data into MySQL")):
            chunk.to_sql(name=table_name, con=engine, if_exists='append', index=False)
            print(f"Chunk {i + 1}/{len(chunks)} inserted successfully.")

        print(f"Data inserted successfully into '{table_name}' table.")

    except SQLAlchemyError as e:
        print(f"Error while inserting data: {e}")
    finally:
        # 연결 해제
        engine.dispose()
def train_test_split():
    data = extract_rating_data()
    data['timestamp_rank'] = (data.groupby('user_id')['timestamp']\
                              .rank(ascending=False, method='first'))
    train = data[data['timestamp_rank']>10]
    test = data[data['timestamp_rank']<=10]
    # print(train.shape)
    # print(test.shape)
    return train, test
def mf_predict_model(degree=10, lr=0.005, epochs=50):
    # 데이터 분할 (train, test는 실제 데이터셋으로 대체)
    train, test = train_test_split()

    # 사용자 및 영화 필터링
    popular_users = train['user_id'].value_counts().index[:1000]
    popular_movies = train['movie_id'].value_counts().index[:500]

    # 필터링된 사용자 및 영화만 유지
    train_filtered = train[train['user_id'].isin(popular_users) & train['movie_id'].isin(popular_movies)]
    train_data = train_filtered.groupby('movie_id').filter(lambda x: len(x['movie_id']) > 100)

    # Reader 설정
    reader = Reader(rating_scale=(0.5, 5))

    # Surprise 데이터셋 생성
    trainData = SurpriseDataset.load_from_df(
        train_data[['user_id', 'movie_id', 'rating']], reader
    ).build_full_trainset()

    # SVD 모델 생성 및 학습
    matrix = SVD(n_factors=degree, n_epochs=epochs, lr_all=lr, biased=False)
    matrix.fit(trainData)
    return matrix, train_data
def save_predictions_to_mysql(model, train_data, table):
    fields = postgres_connect_field()
    # MySQL 연결 문자열 생성
    conn = f"postgresql+psycopg2://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"

    # 훈련 데이터에서 고유한 user_id, movie_id 추출
    user_ids = train_data['user_id'].unique()
    movie_ids = train_data['movie_id'].unique()

    # Batch insert preparation
    insert_data = []

    # 각 사용자 및 영화에 대한 예측 평점 계산
    for user_id in tqdm(user_ids):
        for movie_id in movie_ids:
            # 예측 평점 계산
            prediction = model.predict(uid=int(user_id), iid=int(movie_id))
            predicted_rating = prediction.est

            # 데이터 리스트에 추가
            insert_data.append((int(user_id), int(movie_id), float(predicted_rating)))

    # insert_data 리스트를 DataFrame으로 변환
    df = pd.DataFrame(insert_data, columns=['user_id', 'movie_id', 'predicted_rating'])
    try:
        # with 문으로 engine 자동 종료 처리
        with create_engine(conn).connect() as connection:

            # DataFrame 데이터를 MySQL 테이블에 삽입
            df.to_sql(table, con=connection, if_exists='append', chunksize=1000, index=False)
            print(f"Data inserted successfully into table {table}.")

    except SQLAlchemyError as e:
        # 에러 처리 및 로그 출력
        print(f"Error while inserting data: {e}")
    else:
        print("Data inserted and connection closed!!!")

from tqdm import tqdm
from scipy.sparse import lil_matrix
from implicit.als import AlternatingLeastSquares

from tqdm import tqdm
from scipy.sparse import lil_matrix
from implicit.als import AlternatingLeastSquares

from tqdm import tqdm
from scipy.sparse import lil_matrix
from implicit.als import AlternatingLeastSquares

from tqdm import tqdm
from scipy.sparse import lil_matrix
from implicit.als import AlternatingLeastSquares

from tqdm import tqdm
from scipy.sparse import lil_matrix
from implicit.als import AlternatingLeastSquares

def imf_predict_model(degree=10, epochs=50, alpha=1.0):
    # 데이터 분할 (train, test는 실제 데이터셋으로 대체)
    train, test = train_test_split()

    # 사용자 및 영화 필터링
    popular_users = train['user_id'].value_counts().index[:1000]
    popular_movies = train['movie_id'].value_counts().index[:]


    train_filtered = train[train['user_id'].isin(popular_users) &
                           train['movie_id'].isin(popular_movies) &
                           (train['rating'] >= 4.0)
    ]
    train_data = (train_filtered.groupby('movie_id')
                  .filter(lambda x: len(x['movie_id']) > 100))

    # 사용자와 영화의 수를 기반으로 희소 행렬 크기 설정
    num_users = train_data['user_id'].nunique()
    num_movies = train_data['movie_id'].nunique()


    # 인덱스 매핑 생성
    user_id2index = {user_id: idx for idx, user_id in enumerate(train_data['user_id'].unique())}
    movie_id2index = {movie_id: idx for idx, movie_id in enumerate(train_data['movie_id'].unique())}


    # 희소 행렬 생성
    matrix = lil_matrix((num_users, num_movies))
    for _, row in tqdm(train_data.iterrows()):
        user_index = user_id2index[row["user_id"]]
        movie_index = movie_id2index[row["movie_id"]]
        matrix[user_index, movie_index] = 1.0 * alpha

    # CSR 형식으로 변환
    matrix_csr = matrix.tocsr()

    # 모델 학습
    model = AlternatingLeastSquares(factors=degree, iterations=epochs, calculate_training_loss=True, random_state=1)
    model.fit(matrix_csr)  # CSR 형식을 사용하여 학습

    # 사용자별 추천 생성
    recommendations = model.recommend_all(matrix_csr)
    df = pd.DataFrame(columns=["user_id", "movie_id", "rating"])
    idx = 0
    for (user_id, user_index), movie_idxs in tqdm(zip(user_id2index.items(), recommendations)):
        for movie_idx in movie_idxs:
            movie_id = [key for key, value in movie_id2index.items() if value == movie_idx][0]
            rating = train_data[(train_data['user_id']==user_id) &
                                (train_data['movie_id']== movie_id)]["rating"].values
            # rating 값을 설정: 기존 값이 존재하지 않으면 가상의 평점 0 사용
            rating_value = rating[0] if len(rating) > 0 else None
            df.loc[idx] = [user_id, movie_id, rating_value]
            idx += 1


    return df


if __name__ == "__main__":
    df = imf_predict_model()
    insert_data_into_table(df, 'imf_model')






    # # Sample CSV path (assumed structure, adjust path as necessary)
    # df = pd.read_csv("../data/imf_model3.csv")  # Adjusting the path to match your environment
    #
    # # 문자열로 저장된 리스트를 실제 리스트로 변환 후, PostgreSQL 형식의 배열 문자열로 변환
    # df['recommended_movies'] = df['recommended_movies'].apply(
    #     lambda x: '{' + ', '.join(map(str, ast.literal_eval(x))) + '}')
    #
    # # 수정된 내용을 새로운 CSV 파일로 저장
    # df.to_csv("../data/imf_model.csv", index=False)
    # Converting the 'recommended_movies' column from string representation to list
    # df['recommended_movies'] = df['recommended_movies'].apply(ast.literal_eval)
    # df.to_csv("../data/inf_model1.csv")
    # Checking the type of the first element to confirm conversion
    # data = df['recommended_movies'].to_list()
    # data_type = type(data[0])
    #
    # print(data_type)
    # matrix, train = mf_predict_model()
    # save_predictions_to_mysql(matrix, train, "mf_model")

    # df = popularity_predict_model(200)
    # save_to_database(df, table_name="popular_movies")
    # exit()

    # df = nmf_predict_model(10)
    # save_pred_model_to_mysql(df, "nmf_model")
    # exit()

    # df = svd_predict_model(10)
    # save_pred_model_to_mysql(df, "svd_model")
    # exit()
