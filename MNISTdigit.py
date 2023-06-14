import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.datasets import mnist

(train_X,train_y),(test_X,test_y) = mnist.load_data()
train_X.shape
plt.imshow(train_X[1],cmap='gray')
train_y
#normalize and reshaping our X data
train_X = train_X.reshape(-1,28,28,1)
test_X  = test_X.reshape(-1,28,28,1)

train_X = train_X.astype('float32')
test_X  = test_X.astype('float32')

train_X = train_X/255
test_X  = test_X/255

#one hot encode our y data
from keras.utils import np_utils 
train_y = np_utils.to_categorical(train_y)
test_y  = np_utils.to_categorical(test_y)

train_y[1]

#creating our model
input_shape=(28,28,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,Dropout,Dense,MaxPooling2D
from tensorflow.keras.optimizers import SGD
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape,padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu',padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(0.01),
              metrics=['accuracy'])

print(model.summary())

# training our data
batch_size=32
epochs=10

plotting_data = model.fit(train_X,
                          train_y,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(test_X,test_y))

loss,accuracy = model.evaluate(test_X,test_y,verbose=0)

print('Test loss ---> ',str(round(loss*100,2)) +str('%'))
print('Test accuracy ---> ',str(round(accuracy*100,2)) +str('%'))

plotting_data_dict = plotting_data.history
print(plotting_data_dict)
test_loss = plotting_data_dict['val_loss']
training_loss = plotting_data_dict['loss']
test_accuracy = plotting_data_dict['val_accuracy']
training_accuracy = plotting_data_dict['accuracy']

epochs = range(1,len(test_loss)+1)

plt.plot(epochs,test_loss,marker='X',label='test_loss')
plt.plot(epochs,training_loss,marker='X',label='training_loss')
plt.legend()

plt.plot(epochs,test_accuracy,marker='X',label='test_accuracy')
plt.plot(epochs,training_accuracy,marker='X',label='training_accuracy')
plt.legend()

model.save('MNIST_10_epochs.h5')
print('Model Saved !!!')

classifier = load_model('MNIST_10_epochs.h5')


drawing = False
cv2.namedWindow('win')
black_image = np.zeros((256, 256, 3), np.uint8)
ix, iy = -1, -1

def draw_circles(event, x, y, flags, param):
    global ix, iy, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(black_image, (x, y), 5, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.setMouseCallback('win', draw_circles)

while True:
    cv2.imshow('win', black_image)
    if cv2.waitKey(1) == 27:
        break
    elif cv2.waitKey(1) == 13:
        input_img = cv2.resize(black_image, (28, 28))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = input_img.reshape(1, 28, 28, 1)
        res = np.argmax(classifier.predict(input_img), axis=1)[0]
        cv2.putText(black_image, text=str(res), org=(205, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2)
    elif cv2.waitKey(1) == ord('c'):
        black_image = np.zeros((256, 256, 3), np.uint8)
        ix, iy = -1, -1

cv2.destroyAllWindows()
