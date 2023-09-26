import keras
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj wytrenowany model TensorFlow
model = keras.models.load_model('cat_vs_dogs_model_3')


def preprocess_image(img):
    resized_image = cv2.resize(img, (150, 150))
    # Wyświetl oryginalny obraz
    plt.subplot(1, 2, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Wyświetl przeskalowany obraz
    plt.subplot(1, 2, 2)
    plt.title("Przeskalowany obraz")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Pokaż oba obrazy
    plt.show()
    image = resized_image
    return image


def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    #plt.imshow(img_array)
    prediction = model.predict(img_array)
    print(prediction)
    if prediction[0][0] > 0.5:
        result = "To jest pies!"
    else:
        result = "To jest kot!"

    return result


# if __name__ == "__main__":
#     # Ścieżka do zdjęcia na twoim komputerze
#     image_path = "/archive/unseen-cat.jpeg"
#
#     # Dokonaj predykcji
#     result = predict_image(image_path)
#
#     # Wyświetl wynik
#     print(result)


# Ścieżka do zdjęcia na twoim komputerze
image_path = "/home/marcin/PycharmProjects/pythonProject2/archive/unseen-dog.jpg"

# Dokonaj predykcji
result = predict_image(image_path)

# Wyświetl wynik
print(result)