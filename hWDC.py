import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

#MNIST veri setini yükleme ve ön işleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#CNN Modeli oluşturma
from tensorflow.keras.layers import Input

#CNN Modeli oluşturma
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#Modeli derleme ve eğitme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(X_test, y_test))

#Eğitim sonuçlarını görselleştirme
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

#Test setinde model performansını değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Doğruluğu: {test_acc}")

#Hata analizi (yanlış tahmin edilen örnekleri bulma)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Hatalı tahmin edilen örnekler
errors = np.where(predicted_classes != true_classes)[0]
print(f"Hatalı tahmin edilen örnek sayısı: {len(errors)}")

# Hatalı tahmin edilen birkaç örneği görselleştirme
for i in range(5):  # İlk 5 hatalı tahmin
    idx = errors[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Gerçek: {true_classes[idx]}, Tahmin: {predicted_classes[idx]}')
    plt.axis('off')
    plt.show()

# Doğru tahmin edilen örnekler
correct_indices = np.where(predicted_classes == true_classes)[0]
print(f"Doğru tahmin edilen örnek sayısı: {len(correct_indices)}")

# Doğru tahmin edilen birkaç örneği görselleştirme
for i in range(5):  # İlk 5 doğru tahmin
    idx = correct_indices[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Gerçek: {true_classes[idx]}, Tahmin: {predicted_classes[idx]}')
    plt.axis('off')
    plt.show()