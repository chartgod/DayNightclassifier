#Author : lsh 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import tensorflow as tf
# 사용 가능한 GPU 목록 확인
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 어떤 GPU를 사용할지 지정
        tf.config.experimental.set_visible_devices([gpus[4], gpus[6]], 'GPU')

        # GPU 메모리 할당 문제를 방지하기 위해 메모리 성장 활성화
        for gpu in [gpus[4], gpus[6]]:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("사용 중인 GPU:", [gpus[4], gpus[6]])
    except RuntimeError as e:
        print(e)

# 랜덤 시드 고정
np.random.seed(3)

# ModelCheckpoint 콜백을 사용하여 50 에폭마다 모델 저장
checkpoint = ModelCheckpoint('daynight_epoch_{epoch:02d}.h5',
                             monitor='val_accuracy', save_best_only=True, period=30)

# EarlyStopping 콜백을 사용하여 검증 손실이 더 이상 감소하지 않을 때 훈련 중단
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ReduceLROnPlateau 콜백을 사용하여 학습률 동적 조절
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7)
# 이미지 데이터 증강을 위한 ImageDataGenerator 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # 이미지 회전 각도 범위
    width_shift_range=0.2,  # 가로 이동 범위
    height_shift_range=0.2,  # 세로 이동 범위
    zoom_range=0.2,  # 확대/축소 범위
    horizontal_flip=True,  # 수평 뒤집기
    vertical_flip=True  # 수직 뒤집기
)

# 검증 및 테스트 데이터는 증강하지 않음
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

with tf.distribute.MirroredStrategy().scope():
    train_generator = train_datagen.flow_from_directory(
        "Data/All/train",
        target_size=(224, 224),
        batch_size=256,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        "Data/All/val",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        "Data/All/test",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Model compilation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=1000,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

# Evaluate the model on the test set
print("-- Evaluate --")
scores = model.evaluate(test_generator, steps=len(test_generator))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Save the trained model
model.save('daynight.h5')

# 각각의 훈련과 검증에 대한 손실과 정확도를 저장
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot training history (accuracy and loss)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('.png')
plt.show()

plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('.png')
plt.show()
