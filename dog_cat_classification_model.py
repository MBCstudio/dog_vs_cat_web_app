import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from data_export import train_dir, validation_dir, test_dir
from keras.applications import VGG16


#MODEL 1

#BULDING MODEL

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3))) #using 32 filters and 3x3 window to 'scan' photo
# model.add(layers.MaxPooling2D((2,2))) #decresing size image (/2)
# model.add(layers.Conv2D(64, (3,3), activation='relu')) #adding more filters coz we decresed img size
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# #model.summary()
#
# #COMPILE
# model.compile(optimizer= optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
#
# #PREPROCESING DATA
# train_datagen = ImageDataGenerator(rescale=1./255) #przesaklowywujemy wszyskie dane przez 1/255 by były w zakresie (0;1)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# #zmieniamy dane wejsciowe
# train_generator = train_datagen.flow_from_directory(
#     train_dir, #katalog docelowy
#     target_size= (150,150), #na jakie przeskalowanie
#     batch_size= 20,
#     class_mode= 'binary' #potrzebujemy binarnych etykier bo bedziemy uzywac funkcji straty 'binary_crossentropy'
# )
#
# validation_generator = train_datagen.flow_from_directory(
#     validation_dir,
#     target_size= (150,150),
#     batch_size= 20,
#     class_mode= 'binary'
# )
#
# #FITTING MODEL
# #do dopasowania modeli opartych na danych wejsciowych przetwarznych przez ImageDataGenerator używamy fit z odpowiednimi agumentami do generatorów
# history = model.fit(
#     train_generator, #co tranujemy
#     steps_per_epoch= 100, #co ile wykomujemy krok (w tym przypadku jeżeli nasze próbki mają rozmiar 20 i jest 100 karków to bedziemy mieli 2000 próbek)
#     epochs= 30,
#     validation_data= validation_generator,
#     validation_steps= 50 #nalogicznie jak steps_per_epoch
# )
# model.save("cats_vs_dogs_model_1")

# #MODEL 2 (dropout + argumentacja danych)
#
# #BULDING MODEL
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# #COMPILE
# model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
#
# #PREPORCESING DATA
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip= True,
#     shear_range=0.2,
#     zoom_range=0.2,
# )
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150,150),
#     batch_size= 32,
#     class_mode= 'binary'
# )
#
# vaidation_generator = train_datagen.flow_from_directory(
#     validation_dir,
#     target_size= (150,150),
#     batch_size= 32,
#     class_mode= 'binary'
# )
#
# #FITTING
# history = model.fit(
#     train_generator,
#     steps_per_epoch= 62,
#     epochs=100,
#     validation_data= vaidation_generator,
#     validation_steps= 25,
# )
# model.save('cats_vs_dogs_model_2')
#
#
#
# #MAKING PLOTS
# acc = history.history["acc"]
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))

#MODEL 3 (EKSTRAKCJA CECH Z ARGUMENTACJA DANYCH)

#USING VGG16
conv_base = VGG16(weights='imagenet', include_top= False, input_shape=(150,150,3))
#conv_base.summary() #jaką strukture4 ma model VGG16
#BULDING MODEL
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False #zamrazamy warsty w celu unikiniecia zniszczenia wag wyuczonego juz modelu
#COMPAILING
model.compile(loss='binary_crossentropy', optimizer= optimizers.RMSprop(lr= 2e-5), metrics=['acc'])

#FITTING
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
    rotation_range= 45,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    fill_mode= 'nearest',
    shear_range= 0.2,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= (150,150),
    batch_size= 20,
    class_mode= 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size= 20,
    class_mode= 'binary',
    target_size= (150,150)
)

history = model.fit(
    train_generator,
    epochs= 30,
    steps_per_epoch= 100,
    validation_data= validation_generator,
    validation_steps=50,
    verbose=2
)

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1' or layer.name == 'block5_conv2' or layer.name == 'block5_conv3':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer= optimizers.RMSprop(lr= 2e-5), metrics=['acc'])
history = model.fit(
    train_generator,
    epochs= 30,
    steps_per_epoch= 100,
    validation_data= validation_generator,
    validation_steps=50,
    verbose=2
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size= 20,
    target_size= (150,150),
    class_mode= 'binary'
)

test_loss, test_acc = model.evaluate(test_generator)
print("dokladnosc trenowania: ", test_acc)
print(model.predict())

model.save('cat_vs_dogs_model_3')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Dokładkość trenowania')
plt.plot(epochs,val_acc, 'b', label='Dokładność walidacji')
plt.title("Dokładność trenowania i walidacji")
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Strata ternowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title("Strata trenowania i walidacji")
plt.legend()
plt.show()