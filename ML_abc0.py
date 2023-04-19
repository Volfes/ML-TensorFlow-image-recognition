import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random


dir_data = '/content/drive/MyDrive/data/'
labels_dir = os.listdir(dir_data)
images_labels = []
img_data = []


# change str value of label to int value A=1, B=2, C=3, 0=0
def change_to_num(label, letter):
    if label[letter] == "A":
        label[letter] = 1
    elif label[letter] == "B":
        label[letter] = 2
    elif label[letter] == "C":
        label[letter] = 3
    elif label[letter] == "0":
        label[letter] = 0


# change int value of label to str value
def change_to_letter(number):
    if number == 1:
        return "A"
    elif number == 2:
        return "B"
    elif number == 3:
        return "C"
    elif number == 0:
        return "0"


# randomly flip given image
def random_flip(image_matrix):
    operation = random.randint(0, 3)
    if operation == 0:
        return image_matrix
    elif operation == 1:
        return np.flipud(image_matrix)
    elif operation == 2:
        return np.fliplr(image_matrix)
    else:
        image_matrix = np.fliplr(image_matrix)
        return np.flipud(image_matrix)


# load and normalize images (image/255)
for label in labels_dir:
    pictures = os.listdir(dir_data+label)
    for pic in pictures:
        img = cv2.imread(dir_data+label+'/'+pic)
        img_array = np.asarray(img)
        img_array = random_flip(img_array)
        normalized_img_array = img_array/255
        img_data.append(normalized_img_array)
        images_labels.append(label)


# Change labels
for item in range(len(images_labels)):
    change_to_num(images_labels, item)


# divide dataset to train 80%, validate 10% and test 10% and batch to to size
# of 20
full_dataset = tf.data.Dataset.from_tensor_slices((img_data, images_labels))
full_dataset = full_dataset.shuffle(buffer_size=len(img_data))
train_size = int(0.8 * len(img_data))
val_size = int(0.1 * len(img_data))
test_size = int(0.1 * len(img_data))
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)
train_dataset = train_dataset.batch(20)
test_dataset = test_dataset.batch(20)
val_dataset = val_dataset.batch(20)


# Neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4)
])


# RMSprop optimizer with changed value of leanring_rate to 0.0001
opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=100,
    jit_compile=True,
    name="RMSprop",
)


# training for 12 epochs
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=12, validation_data=val_dataset)


# # save model
# model.save('/content/drive/MyDrive/TFML/1h_1acc_0.04loss.h5')


# predict for all data in test set
predictions = model.predict(test_dataset)


# take one batch for check
fixed_test = test_dataset.take(1)
for images, labels in fixed_test:
    plt.figure(figsize=(10, 10))

    for i in range(10):
        img = (tf.keras.utils.img_to_array(images[i]))
        img = img.reshape((1,) + img.shape)
        guess = model.predict(img)
        guess_text = change_to_letter(guess.argmax())
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(images[i].numpy().astype("float32"))
        plt.title("real: "+change_to_letter(labels[i].numpy().astype("uint8")))
        plt.xlabel("prediction: "+guess_text)
plt.savefig("/content/drive/MyDrive/TFML/Test.png")


# Obtain values of true positive/negative and false positive/negative
TP = 0  # 1
TN = 0  # 0
FP = 0  # 0 |pred: 1
FN = 0  # 1 |pred: 0
for images, labels in test_dataset.take(6):
    for i in range(20):
        img = (tf.keras.utils.img_to_array(images[i]))
        img = img.reshape((1,) + img.shape)
        guess = model.predict(img)
        if guess.argmax() == labels[i].numpy().astype("uint8"):
            if guess.argmax() != 0:
                TP += 1
            else:
                TN += 1
        else:
            if guess.argmax() == 0:
                FN += 1
            else:
                FP += 1


# Accuracy, recall, precision, f1-score of taken dataset batches
def test_goodness(TruePositive, TrueNegative, FalsePositive, FalseNegative):
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    recall = (TP)/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*(precision * recall)/(precision + recall)
    return print(f"accuracy: {accuracy*100}, recall: {recall*100},"
                 f" precision: {precision}, f1-score: {f1}")


test_goodness(TP, TN, FP, FN)


# Accuracy plot
fig, ax = plt.subplots()
plt.title("Accuracy")
plt.xlabel("Epoches")
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend(loc='lower right')
fig.savefig("/content/drive/MyDrive/TFML/Accuracy3.png")


# Loss plot
fig, ax = plt.subplots()
plt.title("Loss")
plt.xlabel("Epoches")
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(loc='lower right')
fig.savefig("/content/drive/MyDrive/TFML/Loss.png")
