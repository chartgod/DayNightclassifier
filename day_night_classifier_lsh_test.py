#Author : lsh 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 사용 가능한 GPU 목록 확인
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 어떤 GPU를 사용할지 지정
        tf.config.experimental.set_visible_devices([gpus[4], gpus[5], gpus[6]], 'GPU')

        # GPU 메모리 할당 문제를 방지하기 위해 메모리 성장 활성화
        for gpu in [gpus[4], gpus[5], gpus[6]]:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("사용 중인 GPU:", [gpus[4], gpus[5], gpus[6]])
    except RuntimeError as e:
        print(e)

# 랜덤 시드 고정
np.random.seed(3)

# ModelCheckpoint 콜백을 사용하여 50 에폭마다 모델 저장
checkpoint = ModelCheckpoint('/home/lsh/share/C2PNet-main/DayNight/bct_11_daynight_epoch_{epoch:02d}.h5',
                             monitor='val_accuracy', save_best_only=True, period=100)

# EarlyStopping 콜백을 사용하여 검증 손실이 더 이상 감소하지 않을 때 훈련 중단
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ReduceLROnPlateau 콜백을 사용하여 학습률 동적 조절
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7)

# Train set, validation set, and test set generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

with tf.distribute.MirroredStrategy().scope():
    train_generator = train_datagen.flow_from_directory(
        "/home/lsh/share/C2PNet-main/DayNight/Data/All/train",
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        "/home/lsh/share/C2PNet-main/DayNight/Data/All/val",
        target_size=(256, 256),
        batch_size=5,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        "/home/lsh/share/C2PNet-main/DayNight/Data/All/test",
        target_size=(256, 256),
        batch_size=5,
        class_mode='binary'
    )

    # CNN model with output activation as sigmoid
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Model compilation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 훈련에 추가
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
model.save('/home/lsh/share/C2PNet-main/DayNight/bct_11_daynight.h5')

# Plot training history (accuracy and loss)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('/home/lsh/share/C2PNet-main/DayNight/training_accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('/home/lsh/share/C2PNet-main/DayNight/training_loss.png')
plt.show()
