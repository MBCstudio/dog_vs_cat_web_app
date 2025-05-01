from flask import Flask, render_template, request, url_for
import numpy as np
from keras.preprocessing import image
import keras
import os

app = Flask(__name__)

#zaladaowanie modelu
model = keras.models.load_model('cat_vs_dogs_model_3')

#tworzenie folderu z zdjeciami
UPLOADS_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/przetworz_zdjecie", methods=["POST"])
def przetworz_zdjecie():
    #pobieranie zdjecia
    zdjecie = request.files["zdjecie"]

    if zdjecie:
        # Zapisanie przesłanego zdjęcia w folderze uploads
        zdjecie_path = os.path.join("static",app.config['UPLOAD_FOLDER'], zdjecie.filename)
        zdjecie.save(zdjecie_path)

        #dkoładana sciezka dla html
        image_url = url_for("static",filename=app.config['UPLOAD_FOLDER'] + "/" + zdjecie.filename)

        #aktywacja modelu
        img = image.load_img(zdjecie_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        prediction = model.predict(img_array)
        print(prediction)
        if prediction[0][0] > 0.5:
            wynik = "To jest pies!"
        else:
            wynik = "To jest kot!"

        return render_template("result.html", result = wynik, image_url = image_url)
    else:
        'Nie wybrano zdjecia'

if __name__ == "__main__":
    #Uruchomienie aplikacji z włączonym trybem debugowania
    app.run(debug=True)


