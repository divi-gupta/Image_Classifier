#Importing the Libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Input,Dropout,MaxPooling2D,BatchNormalization,Flatten,GlobalMaxPooling2D,Dense
from tensorflow.keras.models import Model

#Data Preprocessing
dataset = tf.keras.datasets.cifar10
(x_train,y_train) , (x_test,y_test) = dataset.load_data()
y_train,y_test = y_train.flatten(),y_test.flatten()
x_train , x_test = x_train/255.0 , x_test/255.0

#Getting number of classes in dataset
k=len(set(y_train))

# Data Augmentation
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

#Building the model using functional API
i = Input(shape = x_train[0].shape)
x = Conv2D(32, (3,3), padding = "same", activation = "relu")(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), padding = "same", activation =  "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
x = BatchNormalization()(x) 
x = Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.2)(x)
x = Dense(k, activation = "softmax")(x)
model = Model(i,x)

#Compiling the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#fitting the model
r = model.fit(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=80)

# Plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Loss per Iteration")
plt.show()

# Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.title("Accuracy per Iteration")
plt.show()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))

#Label Mapping
labels = '''airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck'''.split()

#Some misclassified examples
misclassified_idx = np.where(p_test != y_test)[0]
i = np.random.choice(misclassified_idx)
plt.imshow(x_test[i], cmap='gray')
plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));

#Saving the model
model_save=model.save("model_save.h5")

#Model summary
model.summary()

