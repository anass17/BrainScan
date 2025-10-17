import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img):

    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    img_resized = img.resize((224, 224))
    
    img_array = image.img_to_array(img_resized)
    
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_name = class_names[predicted_class]
    
    return class_name, predictions[0]