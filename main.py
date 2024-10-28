import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


np.random.seed(0)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10


def display_sample_images():
    f, ax = plt.subplots(1, num_classes, figsize=(20, 20))
    for i in range(num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title(f"Label: {i}", fontsize=16)
    plt.show()

display_sample_images()


print("Sample training labels:", y_train[:10])


x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

y_train = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)


batch_size = 512
epochs = 10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical)
print(f'Test loss: {test_loss}, Test Accuracy: {test_accuracy}')


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Predicted classes:", y_pred_classes[:10])


def plot_sample_prediction():
    random_index = np.random.choice(len(x_test))
    x_sample = x_test[random_index].reshape(28, 28)
    y_sample_true = y_test[random_index]
    y_sample_pred_class = y_pred_classes[random_index]
    
    plt.title(f'Predicted: {y_sample_pred_class}, True: {y_sample_true}', fontsize=16)
    plt.imshow(x_sample, cmap='gray')
    plt.show()

plot_sample_prediction()


def plot_confusion_matrix():
    cm = confusion_matrix(y_test, y_pred_classes)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    ax.set_title('Confusion Matrix', fontsize=16)
    plt.show()

plot_confusion_matrix()

def analyze_misclassifications():
    error = (y_pred_classes - y_test != 0)
    y_pred_classes_error = y_pred_classes[error]
    y_pred_error = y_pred[error]
    y_true_error = y_test[error]
    x_test_error = x_test[error]

    y_pred_error_prob = np.max(y_pred_error, axis=1)
    true_prob_error = np.diagonal(np.take(y_pred_error, y_true_error, axis=1))
    diff_error_pred_true_prob = y_pred_error_prob - true_prob_error
    sorted_index_diff = np.argsort(diff_error_pred_true_prob)
    top_index_diff = sorted_index_diff[-6:]


    num = len(top_index_diff)
    f, ax = plt.subplots(1, num, figsize=(20, 20))
    for i in range(num):
        index = top_index_diff[i]
        sample = x_test_error[index].reshape(28, 28)
        y_t = y_true_error[index]
        y_p = y_pred_classes_error[index]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title(f'Predicted label: {y_p}\nTrue label: {y_t}', fontsize=22)
    plt.show()

analyze_misclassifications()
