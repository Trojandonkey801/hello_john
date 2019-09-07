"""
import keras
iport numpy as np
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
import numpy as np

resnet_model = resnet50.ResNet50()
filename = '/home/kim1/Qaunt/Cat03.jpg'
original_image = load_img(filname,target_size(224,224))
numpy_image=img_to_array(original_image)
input_image=np.expand_dims(numpy_image,axis=0)
print('PIL image size = ',original_image.size)
print('NumPy image size',numpy_image.size)
print('Input image size',input_image.size)
processed_image_resnet50=resnet50.preprocess_input(input_image.copy())
predictions_resnet50=resnet_model.predict(processed_image_resnet50)
lab_resnet50=decode_predictions(predictions_resnet50)
print('lable_resnet50=',lab_resnet50)

"""
import tensorflow as tf
import numpy as np
import keras
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = tf.keras.applications.resnet50.ResNet50(include_top=True)
#inception_model = inception_v3.InceptionV3(weights=’imagenet’) 
model.load_weights('writeout.h5')
"""
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
"""

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#tf.keras.models.save_model(model,"writeout.h5")
img_path = "./Cat03.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
