#LEE SEUNG HEON
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load the trained model
model = tf.keras.models.load_model('/home/lsh/Haeundae_daynight.h5')

# Sample images for classification
sample_images = [
    '/home/lsh/.jpg',

]

for fn in sample_images:
    # Load and preprocess the image
    img = tf.keras.utils.load_img(fn, target_size=(256, 256))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  

    # Predict the class
    classes = model.predict(x)

    # Display the image
    plt.imshow(mpimg.imread(fn))
    
    # 결과 표시 (진하게)
    result = "밤" if classes[0][0] >= 0.5 else "낮"
    plt.title(f'결과: {result}', fontweight='bold')
    
    plt.show()
    print('--------------------')
