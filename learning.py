import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.function
def train_step(model, criterion, optimizer, images, labels):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(images)
        # 2. Loss 계산
        loss = criterion(labels, predictions)

    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss와 accuracy를 업데이트 합니다.
    # train_loss(loss)
    # train_acc(labels, predictions)


@tf.function
def test_step(model, criterion, optimizer, images, labels):
    # 1. 예측 (prediction)
    predictions = model(images)
    # 2. Loss 계산
    loss = criterion(labels, predictions)

    # Test셋에 대해서는 gradient를 계산 및 backpropagation 하지 않습니다.

    # loss와 accuracy를 업데이트 합니다.
    # test_loss(loss)
    # test_acc(labels, predictions)