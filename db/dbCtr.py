import mysql.connector as conn
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm


def filter_valid_users(tags, connection):
    # 데이터베이스에서 고객 목록을 가져와 유효한 user_id만 남김
    query = "SELECT user_id FROM customers"
    customers = pd.read_sql(query, connection)

    # tags 데이터에서 customers 테이블에 있는 user_id만 필터링
    valid_tags = tags[tags['user_id'].isin(customers['user_id'])]

    return valid_tags

def insert_data_into_table(df, table):
    # connecting fields
    fields = postgres_connect_field()
    # MySQL 연결 문자열 생성
    conn = f"postgresql+psycopg2://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"

    if 'movies' == table:
        create_movie_table()
    elif 'tags' == table:
        create_tag_table()
        df = filter_valid_users(df, conn)
    elif 'ratings' == table:
        create_rating_table()
    elif 'customers' == table:
        create_customer_table()

      # 예외 처리를 통한 안전한 MySQL 연결 및 데이터 삽입
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

def bring_dataframe_from_table(table):

    fields = connect_field()
    conn = f"mysql+mysqlconnector://{fields['username']}:{fields['password']}@{fields['host']}:{fields['port']}/{fields['database']}"
    try:
        engine = create_engine(conn)
        query = f"select * from {table}"
        df = pd.read_sql(query, con=engine)
    except SQLAlchemyError as e:
        # 에러 처리 및 로그 출력
        print(f"Error while inserting data: {e}")
        return
    else:
        engine.dispose()
        print("Data selected and connection closed!!!")

    return df

def updata_movie_ids_in_customers(movies):
    fields = connect_field()
    db = conn.connect(
        host=fields["host"],
        user=fields["username"],
        password=fields["password"],
        database=fields["database"],
        port=fields['port']
    )
    cursor = db.cursor()
    try:
        for _, row in tqdm(movies.iterrows()):
            user_id = row['user_id']
            movie_ids = row['movie_id']
            movies_ids_str = ','.join(map(str, movie_ids))

            cursor.execute(f"update customers \
                           set movie_ids = '{movies_ids_str}'\
                           where user_id = {user_id};"
                           )
        db.commit()
    except Exception as error:
        print(error)
    else:
        print("movie_ids field updated successfully!!!")
        cursor.close()
        db.close()
    return

def connect_field():
    fields = {
        "host": "3.36.78.179",
        "username": "kogo1490",
        "password": "math1106",
        "database": "mydb",
        "port": 3306
    }
    return fields

def postgres_connect_field():
    fields = {
        "host": "localhost",
        "username": "postgres",
        "password": "math1106",
        "database": "postgres",
        "port": 5432
    }
    return fields


def create_movie_table():
    fields = connect_field()
    db = conn.connect(
        host = fields["host"],
        user = fields["username"],
        password = fields["password"],
        database = fields["database"],
        port = fields['port']
    )
    cursor = db.cursor()
    cursor.execute("show tables like 'movies'")
    result = cursor.fetchone()
    if result:
        print("Table 'movies' already exists!!!!")
    else:
        create_table_query = '''
            create table movies(
                movie_id int primary key,
                title varchar(500),
                genre varchar(500)
            )
        '''
        # 테이블 생성 쿼리 실행
        cursor.execute(create_table_query)
        db.commit()
        print("Table 'movies created successfully!!!")
    cursor.close()
    db.close()

def create_customer_table():
    fields = connect_field()
    db = conn.connect(
        host = fields["host"],
        user = fields["username"],
        password = fields["password"],
        database = fields["database"],
        port = fields['port']
    )
    cursor = db.cursor()
    cursor.execute("show tables like 'customers'")
    result = cursor.fetchone()
    if result:
        print("Table 'customers' already exists!!!!")
    else:
        create_table_query = '''
            create table customers(
                user_id int primary key
            )
        '''
        # 테이블 생성 쿼리 실행
        cursor.execute(create_table_query)
        db.commit()
        print("Table 'customers created successfully!!!")
    cursor.close()
    db.close()

def create_tag_table():
    fields = connect_field()
    db = conn.connect(
        host = fields["host"],
        user = fields["username"],
        password = fields["password"],
        database = fields["database"],
        port = fields['port']
    )
    cursor = db.cursor()
    cursor.execute("show tables like 'tags'")
    result = cursor.fetchone()
    if result:
        print("Table 'tags' already exists!!!!")
    else:
        create_table_query = '''
            create table tags(
                id int primary key,
                user_id int,
                movie_id int,
                tag text,
                foreign key (user_id) references customers(user_id) 
                on delete cascade on update cascade,
                foreign key (movie_id) references movies(movie_id) 
                on delete cascade on update cascade
            )
        '''
        # 테이블 생성 쿼리 실행
        cursor.execute(create_table_query)
        db.commit()
        print("Table 'tags created successfully!!!")
    cursor.close()
    db.close()

def create_rating_table():
    fields = connect_field()
    db = conn.connect(
        host = fields["host"],
        user = fields["username"],
        password = fields["password"],
        database = fields["database"],
        port = fields['port']
    )
    cursor = db.cursor()
    cursor.execute("show tables like 'ratings'")
    result = cursor.fetchone()
    if result:
        print("Table 'ratings' already exists!!!!")
    else:
        create_table_query = '''
            create table ratings(
                id int primary key,
                user_id int,
                movie_id int,
                rating float,
                timestamp timestamp,
                foreign key (user_id) references customers(user_id) 
                on delete cascade on update cascade,
                foreign key (movie_id) references movies(movie_id) 
                on delete cascade on update cascade
            )
        '''
        # 테이블 생성 쿼리 실행
        cursor.execute(create_table_query)
        db.commit()
        print("Table 'ratings created successfully!!!")
    cursor.close()
    db.close()




if __name__ == "__main__":
    # movies = extract_movie_data()
    # insert_data_into_table(movies, "recommend_movies")
    create_customer_table()
    create_movie_table()
    create_tag_table()
    create_rating_table()