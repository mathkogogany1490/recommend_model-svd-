import numpy as np
import tensorflow as tf
import pickle



if __name__ == "__main__":
    with open('../data/testdat.pkl', 'rb') as f:
        testX = pickle.load(f)
    with open('../data/testlabeldat.pkl', 'rb') as f:
        testy = pickle.load(f)

    model = tf.saved_model.load('data/tensor_2_17_0')
    infer = model.signatures["serving_default"]

    sample = np.expand_dims(testX[0], axis=0)
    # 예측 수행
    outputs = infer(tf.convert_to_tensor(sample, dtype=tf.float32))

    # 각 출력 값 처리
    logits = outputs['output_0'].numpy()  # logits
    probabilities = outputs['output_1'].numpy()  # softmax probabilities

    print(f"Predicted Output (Logits): {logits}")
    print(f"Softmax Output (Probabilities): {np.argmax(probabilities)+1}")
    print(f"True Label: {testy[0]}")  # 실제 라벨도 함께 출력

