import tensorflow as tf
import numpy as np
import sys, os
import json, csv
from BurpyIC.settings import REC_DIR, CATEGORY

def save_train_data(payload):
    parsed = json.loads(payload.decode('utf-8'))
    user_id = parsed['id']
    products_data = parsed['data']

    # 전체 상품 목록을 카테고리 별 iterate
    for c in products_data:
        # 저장할 디렉토리 정의 및 생성
        train_data_path = os.path.join(REC_DIR, user_id, CATEGORY[c['category']], 'train_data.csv')
        os.makedirs(os.path.dirname(train_data_path), exist_ok=True)

        # csv 파일 생성 및 데이터 입력
        with open(train_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for item in c['items']:
                row = [item['productId'], *item['avgTaste'], item['score']]
                writer.writerow(row)

    return 'train data has saved'

def save_predict_data(payload):
    parsed = json.loads(payload.decode('utf-8'))
    user_id = parsed['id']
    products_data = parsed['data']

    for c in products_data:
        # 저장할 디렉토리 정의 및 생성
        predict_data_path = os.path.join(REC_DIR, user_id, CATEGORY[c['category']], 'predict_data.csv')
        os.makedirs(os.path.dirname(predict_data_path), exist_ok=True)

        # csv 파일 생성 및 데이터 입력
        with open(predict_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for item in c['items']:
                row = [item['productId'], *item['avgTaste']]
                writer. writerow(row)

    return 'predict data has saved'

def train_recommendation(user_id, category):
    tf.set_random_seed(777) # 시드 지정

    # 학습 데이터 csv 읽기
    train_data_path = os.path.join(REC_DIR, user_id, category, 'train_data.csv')
    filename_queue = tf.train.string_input_producer(
        [train_data_path], shuffle=True, name='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0], [0.], [0.], [0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(value, record_defaults=record_defaults)

    # 데이터 슬라이싱 및 batch 지정
    train_ID_batch, train_x_batch, train_y_batch = \
        tf.train.batch([xy[0:1], xy[1:-1], xy[-1:]], batch_size=5)

    # 플레이스홀더 & 변수 선언
    ID = tf.placeholder(tf.int32, shape=[None, 1], name="ID")
    X = tf.placeholder(tf.float32, shape=[None, 5], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")
    W = tf.Variable(tf.random_normal([5, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    # 모델 저장을 위한 saver 생성
    saver = tf.train.Saver()

    # 예측식 & 오차수정식 정의
    hypothesis = tf.matmul(X, W) + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # 학습 세션 생성
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 학습 데이터 큐 읽기 시작
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 학습 수행
    for step in range(2001):
        ID_batch, x_batch, y_batch = sess.run([train_ID_batch, train_x_batch, train_y_batch])
        ID_val, cost_val, hy_val, _ = sess.run(
            [ID, cost, hypothesis, train], feed_dict={ID: ID_batch, X: x_batch, Y: y_batch})
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nIDs:\n", ID_val, "\nPrediction:\n", hy_val)

    train_model_path = os.path.join(REC_DIR, user_id, category, 'trained_model')
    saver.save(sess, train_model_path) 

    # 학습 데이터 큐 사용 종료
    coord.request_stop()
    coord.join(threads)
    
    return 'train has done'

def predict_recommendation(user_id, category):
    train_model_path = os.path.join(REC_DIR, user_id, category, 'trained_model.meta')
    checkpoint_path = os.path.join(REC_DIR, user_id, category)
    predict_data_path = os.path.join(REC_DIR, user_id, category, 'predict_data.csv')
    predict_result_path = os.path.join(REC_DIR, user_id, 'predict_result.txt')
    
    # 저장된 모델 불러오기
    sess = tf.Session()
    saver = tf.train.import_meta_graph(train_model_path)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())

    # 저장된 변수들 불러오기
    ID = graph.get_tensor_by_name("ID:0")
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    W = graph.get_tensor_by_name("weight:0")
    b = graph.get_tensor_by_name("bias:0")
    hypothesis = tf.matmul(X, W) + b

    # 예측할 데이터 불러오기
    data = np.loadtxt(predict_data_path, delimiter=",")
    predict_ID, predict_X = data[:,0:1], data[:,1:]

    # 예측 수행
    resultItems = []
    for product, tastes in zip(predict_ID, predict_X):
        _product_ID, _score = sess.run([ID, hypothesis], feed_dict={ID: [product], X: [tastes]})
        product_ID, score = np.squeeze(_product_ID).item(), np.squeeze(_score).item()
        resultItems.append({'id': product_ID, 'score': process_score(score)})

    # 예측 결과 저장 (json)
    resultItems = sorted(resultItems, key=lambda i: i['score'], reverse=True)
    contents = {}

    if os.path.exists(predict_result_path):
        with open(predict_result_path, 'r') as file:
            contents = json.load(file)

    with open(predict_result_path, 'w', newline='') as file:
        contents[category] = resultItems
        json.dump(contents, file)

    return 'predict has done'

def fetch_predict_result(user_id):
    predict_result_path = os.path.join(REC_DIR, user_id, 'predict_result.txt')
    with open(predict_result_path) as file:
        payload = json.load(file)
    
    return payload

def process_score(score):
    return max(min(round(score, 1), 5.0), 0.5)
