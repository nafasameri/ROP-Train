from sklearn.metrics import classification_report, fbeta_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import matplotlib.pyplot as plt


def evaluation(y_true, y_pred, labels, path_experiment):
    # Confusion matrix
    # confusion = multilabel_confusion_matrix(y_true, y_pred)
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
    file_perf = open(path_experiment + 'classification_report.txt', 'w')
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
    # plt.show()
    plt.savefig(path_experiment + 'roc-densenet121.png')



# main


# Custom Dataset Class
class ROPDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
        # image = cv2.imread(self.image_paths[idx])

        # # # pre-proccessing
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # clahe = cv2.createCLAHE(clipLimit=5)
        # final_img = clahe.apply(gray_image) + 30

        # # Ordinary thresholding the same image
        # _, ordinary_img = cv2.threshold(gray_image, 155, 255, cv2.THRESH_BINARY)


        # # Create a blank mask (same size as image)
        # mask = np.zeros_like(gray_image)

        # # Get image dimensions
        # height, width = gray_image.shape[:2]
        # center = (width // 2, height // 2)
        # radius = min(center[0], center[1])  # Make the circle fit within the image dimensions

        # # Draw a white circle in the mask
        # cv2.circle(mask, center, radius, (255), thickness=-1)

        # # Apply mask to the image
        # masked_image = cv2.bitwise_and(image, image, mask=mask)

        # masked_image_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        # # Convert the OpenCV NumPy array to PIL Image
        # image = Image.fromarray(masked_image_rgb)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# CNN Model (Using a pretrained model for transfer learning)
class ROPClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ROPClassifier, self).__init__()
        # self.model = models.resnet18(pretrained=True)  # Using a pretrained ResNet18
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # self.model = models.vgg16(pretrained=True)  # Using a pretrained ResNet18

        # self.model = models.inception_v3(pretrained=True)

        self.model = models.densenet121(pretrained=True)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 200

# Data Transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # Preprocessing for the input image
# transform = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# Load Data
image_dir = 'D:/ROP/dataset/'
# label_file = "path_to_labels.csv"

# Assuming label_file contains image names and labels
# labels = [0, 1]  # Replace with actual labels loaded from your CSV
# image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name == "0" or img_name == "1"]

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['0', '1']:
        path = os.path.join(folder, 'Clahe_mask_' + label)
        # path = os.path.join(folder, 'AMSR_mask_' + label)
        #path = os.path.join(folder, label + ' cropped-preproccessing')
        #path = os.path.join(folder, label + 'preproccessing')

        for filename in os.listdir(path):
            # print(filename)
            images.append(os.path.join(path, filename))
            labels.append(int(label))

    return images, labels

image_paths, labels = load_images_from_folder(image_dir)
# print(image_paths)

# Split data into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Create Dataset and DataLoader
train_dataset = ROPDataset(train_paths, train_labels, transform=transform)
val_dataset = ROPDataset(val_paths, val_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = ROPClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Early stopping setup
# early_stopping = EarlyStopping(patience=7, verbose=True)

history = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # if isinstance(outputs, tuple):
        #     logits, aux_logits = outputs  # Unpack the outputs
        # else:
        #     logits = outputs  # Some cases (like evaluation mode) may return only logits

        # logits = logits.squeeze(1)  # Match logits shape to labels

        # # Compute loss
        # loss = criterion(logits, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Average the loss
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
    history.append([running_loss/len(train_loader), val_loss])

    # Check early stopping condition
    # early_stopping(val_loss, model)
    # if early_stopping.early_stop:
    #     print("Early stopping triggered.")
    #     break

    # history.append(running_loss/len(train_loader))
    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# train_losses, val_losses, train_accs, val_accs = [], [], [], []
# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     running_loss = 0.0
#     correct_train, total_train = 0, 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Track training loss and accuracy
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels).sum().item()

#     train_loss = running_loss / len(train_loader)
#     train_acc = 100 * correct_train / total_train
#     train_losses.append(train_loss)
#     train_accs.append(train_acc)

#     # Validation
#     model.eval()
#     running_val_loss = 0.0
#     correct_val, total_val = 0, 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Track validation loss and accuracy
#             running_val_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total_val += labels.size(0)
#             correct_val += (predicted == labels).sum().item()

#     val_loss = running_val_loss / len(val_loader)
#     val_acc = 100 * correct_val / total_val
#     val_losses.append(val_loss)
#     val_accs.append(val_acc)

#     print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# # Plot training and validation metrics
# epochs_range = range(1, num_epochs + 1)
# plt.figure(figsize=(12, 5))

# # Plot loss
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, train_losses, label="Train Loss")
# plt.plot(epochs_range, val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.legend()

# # Plot accuracy
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, train_accs, label="Train Accuracy")
# plt.plot(epochs_range, val_accs, label="Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.legend()

# plt.show()


# Save the trained model
torch.save(model.state_dict(), os.path.join(image_dir, "rop_classifier_densenet121.pth"))
# Load the best model
# model.load_state_dict(torch.load(os.path.join(image_dir, 'checkpoint_resnet18.pt')))


# Load the saved model
model.eval()  # Set the model to evaluation mode
model.to(device)

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    with torch.no_grad():
        for images, labels in data_loader:
            image = images
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # preds = torch.nn.functional.softmax(outputs[0], dim=0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(image)
    return all_preds, all_labels, all_images

# Get predictions and ground truth labels
test_preds, test_labels, all_images = evaluate(model, val_loader)
print(test_preds)

# Calculate evaluation metrics
accuracy = accuracy_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
conf_matrix = confusion_matrix(test_labels, test_preds)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

evaluation(test_labels, test_preds, ['Normal', 'Plus'], '')

plt.figure()
plt.plot(history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('loss-densenet121.png')