import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def evaluation(y_true, y_pred, labels, path_experiment):
    # Confusion matrix
    confusion = [confusion_matrix(y_true, y_pred)]
    print(confusion)

    for i, v in enumerate(confusion):
        tn, fp, fn, tp = v.ravel()
        print('')
        print(labels[i])
        print("Confusion matrix:" + str({"Real Pos": {"tp": tp, "fn": fn}, "Real Neg": {"fp": fp, "tn": tn}}))
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(tp + tn) / float(np.sum(confusion))
        print("Accuracy: " + str(accuracy))
        specificity = 0
        if float(tn + fp) != 0:
            specificity = float(tn) / float(tn + fp)
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(tp + fn) != 0:
            sensitivity = float(tp) / float(tp + fn)
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(tp + fp) != 0:
            precision = float(tp) / float(tp + fp)
        print("Precision: " + str(precision))
        NPV = 0
        if float(tn + fn) != 0:
            NPV = float(tn) / float(tn + fn)
        print("NPV: " + str(NPV))
        dice = 0
        if float(tp + fp + fn) != 0:
            dice = float((2. * tp)) / float((2. * tp) + fp + fn)
        print("Dice coefficient: " + str(dice))
        error_rate = 0
        if float(np.sum(confusion)) != 0:
            error_rate = float(fp + fn) / float(np.sum(confusion))
        print("Error Rate: " + str(error_rate))

        jaccard_index = jaccard_score(y_true,y_pred,average='weighted')
        print("Jaccard similarity score: " + str(jaccard_index))

        # corrcoef = matthews_corrcoef(y_true.flatten(),y_pred.flatten())
        corrcoef = matthews_corrcoef(y_true,y_pred)
        print("The Matthews correlation coefficient: " + str(corrcoef))

        fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='micro')
        print("fbeta micro: " + str(fbeta))
        fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='macro')
        print("fbeta macro: " + str(fbeta))
        fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='weighted')
        print("fbeta weighted: " + str(fbeta))

        fbeta = fbeta_score(y_true, y_pred, beta=2.0, average='micro')
        print("fbeta micro: " + str(fbeta))
        fbeta = fbeta_score(y_true, y_pred, beta=2.0, average='macro')
        print("fbeta macro: " + str(fbeta))
        fbeta = fbeta_score(y_true, y_pred, beta=2.0, average='weighted')
        print("fbeta weighted: " + str(fbeta))
        # fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='samples')
        # print("fbeta samples: " + str(fbeta))
        # fbeta = fbeta_score(y_true, y_pred, beta=0.5, average='binary')
        # print("fbeta binary: " + str(fbeta))

        file_perf = open(path_experiment + 'performances-'+str(labels[i])+'.txt', 'w')
        file_perf.write("Jaccard similarity score: " + str(jaccard_index)
                        + "\nConfusion matrix: " + str({"Real Pos": {"tp": tp, "fn": fn}, "Real Neg": {"fp": fp, "tn": tn}})
                        + "\nACCURACY: " + str(accuracy)
                        + "\nSENSITIVITY: " + str(sensitivity)
                        + "\nSPECIFICITY: " + str(specificity)
                        + "\nPRECISION: " + str(precision)
                        + "\nNPV: " + str(NPV)
                        + "\nError Rate: " + str(error_rate)
                        + "\nThe Matthews correlation coefficient: " + str(corrcoef)
                        + "\nDice coefficient: " + str(dice))
        file_perf.close()

    rep = classification_report(y_true, y_pred, target_names=labels)
    print(rep)
    file_perf = open(path_experiment + 'classification_report-asmr.txt', 'w')
    file_perf.write(rep)
    file_perf.close()

    # Compute ROC curve metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)  # Random chance line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('roc-resnet50.png')


# Load images and labels
image_dir = 'D:/ROP/dataset/'

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['0', '1']:
        path = os.path.join(folder, 'clahe', label)
        for filename in os.listdir(path):
            images.append([os.path.join(path, filename), os.path.join(folder, 'clahe', label, filename)])
            labels.append(int(label))

    return images, labels

image_paths, labels = load_images_from_folder(image_dir)

# Split data into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
train_paths = np.array(train_paths)
val_paths = np.array(val_paths)

print('labels 0', len([l for l in labels if l == 0]))
print('labels 1', len([l for l in labels if l == 1]))
print('train_labels 0', len([l for l in train_labels if l == 0]))
print('train_labels 1', len([l for l in train_labels if l == 1]))
print('val_labels 0', len([l for l in val_labels if l == 0]))
print('val_labels 1', len([l for l in val_labels if l == 1]))


# Preprocessing function for data loading
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))

    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Data generator
def data_generator(image_paths, labels, batch_size):
    while True:
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_indices = indices[start:end]
            batch_images = np.array([preprocess_image(image_paths[i]) for i in batch_indices])
            batch_labels = np.array([labels[i] for i in batch_indices])
            yield batch_images, tf.keras.utils.to_categorical(batch_labels, num_classes=2)

# Parameters
batch_size = 16
num_epochs = 200
learning_rate = 0.0001

# Build the model (using ResNet50 as the backbone)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# model = VGG16(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(224, 224, 3),
#     pooling=None,
#     classes=2,
#     classifier_activation="softmax"
# )

# Create data generators
train_gen = data_generator(train_paths[:, 1], train_labels, batch_size)
val_gen = data_generator(val_paths[:, 1], val_labels, batch_size)

# Train the model
steps_per_epoch = len(train_paths[:, 1]) // batch_size
validation_steps = len(val_paths[:, 1]) // batch_size

# Unfreeze base model layers for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# Fine-tune the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=num_epochs
)

# Evaluate the model
val_images = np.array([preprocess_image(path) for path in val_paths])
val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes=2)
val_preds = np.argmax(model.predict(val_images), axis=1)

print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=["Normal", "Plus"]))

print("Confusion Matrix:")
print(confusion_matrix(val_labels, val_preds))

# Plot training metrics
epochs_range = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["loss"], label="Train Loss")
plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["accuracy"], label="Train Accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig('loss-resnet50-amsr.png')

# Save the model
model.save(os.path.join(image_dir, "rop_classifier_resnet50_amsr_mask.h5"))

evaluation(val_labels, val_preds, ['Normal', 'Plus'], '')