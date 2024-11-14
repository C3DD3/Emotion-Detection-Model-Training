import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Sequential model oluşturuyorum
emotion_model = Sequential()

#2 farklı konvolüsyon katmanı kullanma sebebim overfitting azaltma ve daha karmaşık özellikleri çıkarma
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#maxpooling katmanında boyutları küçültmek için kullanıyorum mesela 4*4 lük matrisi 2*2 matrise indirgeye biliyorum
emotion_model.add(MaxPooling2D(pool_size=(2, 2))) 
#Aşırı uyumu önlemek için rastgele nöronları devre dışı bırakıyorum.
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

#2-b konvolüsyon katmanı çıktısını tek boyuta düşürdüm
emotion_model.add(Flatten())
#Girdi özelliklerini işleyerek sınıflandırma için gerekli olan temsilcileri öğrenmesi için 
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Modeli derleyin
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']  #metrik olarak doğruluğu kullanıyorum duyarlılılk, kesinlik, log loss gibi çok sayıda metriktte vardır 
)

# Eğitim ve doğrulama için veri oluşturucularını tanımladım.
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # %80 eğitim, %20 doğrulama oranı veriyorum.
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\Eyup\\Desktop\\Bitirme Deneme 5\\Flask\\dataset\\data\\train',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    'C:\\Users\\Eyup\\Desktop\\Bitirme Deneme 5\\Flask\\dataset\\data\\test',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# checkpoint oluşturarak modeli kaydettim modelsave() yerine bunu kullanmammın sebebi en uygun modeli kaydetmesi
checkpoint = ModelCheckpoint(
    'emotion_model.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
) 

# son aşamada modeli eğitiyorum.
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=60,
    callbacks=[checkpoint]
)

# OpenCV'nin OpenCL kullanmasını devre dışı bıraktım ki daha efektif olarak bir eğitim sağlayabileyim
cv2.ocl.setUseOpenCL(False)
