import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, fbeta_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


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
    file_perf = open(path_experiment + 'classification_report-vgg16.txt', 'w')
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
    plt.savefig('roc-vgg16.png')




# main
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
import cv2

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
        self.model = models.vgg16(pretrained=True)  # Using a pretrained ResNet18
		
		# Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Modify the classifier part
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
		
    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 200

# Data Transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Data
image_dir = 'D:/ROP/dataset/'

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['0', '1']:
        path = os.path.join(folder, label)
        # path = os.path.join(folder, 'ML_' + label)
        # path = os.path.join(folder, 'Clahe_mask_' + label)
        # path = os.path.join(folder, 'AMSR_mask_' + label)
        # path = os.path.join(folder, label + 'preproccessing')
        # path = os.path.join(folder, label + ' cropped-preproccessing')

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


# Save the trained model
torch.save(model.state_dict(), os.path.join(image_dir, "rop_classifier_vgg16_Clahe.pth"))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

plt.plot(history, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss-vgg16.png')