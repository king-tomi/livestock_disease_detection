import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import os
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool
from torch.utils.data import Subset
import copy
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GCNDiseaseClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels * 16, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.pool = global_max_pool  # Pooling layer
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)  # Apply pooling here
        x = self.fc(x)
        return x

class LiveStockDataset(Dataset):
    def __init__(self, root_dir, transform=None, precompute_graphs=True, patch_size=4, split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.precompute_graphs = precompute_graphs
        self.patch_size = patch_size
        self.images, self.labels, self.graphs = [], [], []
        self.class_names = []
        self._load_data(root_dir)
        self._split_data(split_ratio)

    def _load_data(self, root_dir):
        for class_dir in os.listdir(root_dir):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                if class_dir not in self.class_names:
                    self.class_names.append(class_dir)
                for img_name in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_name)
                    if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(img_path)
                        self.labels.append(class_dir)
                        if self.precompute_graphs:
                            graph = self._load_image_as_graph(img_path)
                            self.graphs.append(graph)

    def _split_data(self, split_ratio):
        total_items = len(self.images)
        train_size = int(total_items * split_ratio)
        indices = torch.randperm(total_items).tolist()
        self.train_indices = indices[:train_size]
        self.test_indices = indices[train_size:]

    def _load_image_as_graph(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return self.image_to_graph(img)

    def image_to_graph(self, image):
        C, H, W = image.shape
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(C, num_patches, self.patch_size, self.patch_size)
        edge_index = []
        grid_width = W // self.patch_size
        for i in range(num_patches):
            row = i // grid_width
            col = i % grid_width
            if col < grid_width - 1:
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
            if row < (H // self.patch_size) - 1:
                down = i + grid_width
                edge_index.append([i, down])
                edge_index.append([down, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = patches.reshape(num_patches, -1)
        data = Data(x=x.float(), edge_index=edge_index)
        return data

    def get_subset(self, indices):
        return Subset(self, indices)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label_index = self.class_names.index(self.labels[idx])
        return graph, label_index  # Ensure this is a tuple

class GCNDiseaseClassifierEnhanced(GCNDiseaseClassifier):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)  # Dropout layer added
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)  # Apply pooling here
        x = self.fc(x)
        return x

def collate_fn(batch):
    data_list = [item[0] for item in batch]  # Data objects
    labels_list = torch.tensor([item[1] for item in batch], dtype=torch.long)
    batch_data = Batch.from_data_list(data_list)
    batch_data.y = labels_list  # Add labels to the batch
    return batch_data

def plot_metrics(train_losses, test_losses, precisions, recalls, accuracies, f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, precisions, label='Precision')
    plt.plot(epochs, recalls, label='Recall')
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Precision, Recall, Accuracy, and F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(train_dataset, test_dataset, model, optimizer, scheduler, criterion, epochs, batch_size, early_stopping_patience=3, checkpoint_interval=1):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    model.train()

    train_losses = []  # Track training loss
    test_losses = []   # Track test loss
    precisions = []    # Track precision
    recalls = []       # Track recall
    accuracies = []    # Track accuracy
    f1_scores = []     # Track F1 score

    best_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        scheduler.step()
        epoch_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        test_loss, precision, recall, accuracy, f1, _, _ = evaluate_model(model, test_loader, criterion)  # Adjusted here
        test_losses.append(test_loss)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f'Epoch {epoch+1}: Training Loss: {epoch_train_loss:.4f}, Test Loss: {test_loss:.4f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}')

        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")

    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), "best_model.pth")
    print("Best model saved.")
    
    plot_metrics(train_losses, test_losses, precisions, recalls, accuracies, f1_scores)



def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    average_loss = total_loss / len(data_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Evaluation Loss: {average_loss:.4f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}')
    return average_loss, precision, recall, accuracy, f1, all_labels, all_preds

def main():
    node_data_dirs = [
        'C:/Users\christian/Downloads/cow',
        'C:/Users\christian/Downloads/poultry'
    ]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    split_ratio = 0.8
    node_datasets = [LiveStockDataset(root_dir=dir, transform=transform, precompute_graphs=True, split_ratio=split_ratio) for dir in node_data_dirs]
    num_cattle_classes = 2
    num_poultry_classes = 2

    # Initialize models and optimizers
    global_model = GCNDiseaseClassifierEnhanced(3, 64, 32, num_cattle_classes + num_poultry_classes)
    nodes = [GCNDiseaseClassifierEnhanced(3, 64, 32, num_cattle_classes + num_poultry_classes) for _ in node_datasets]
    optimizers = [optim.Adam(node.parameters(), lr=0.001) for node in nodes]
    scheduler = optim.lr_scheduler.StepLR(optimizers[0], step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    final_model_params = federated_learning_rounds(node_datasets, nodes, global_model, optimizers, scheduler, criterion, num_rounds=2)
    
    def save_federated_model(final_model_params, filename):
        with open(filename, 'wb') as f:
            pickle.dump(final_model_params, f)
        print(f"Federated model saved as {filename}")
        
    save_federated_model(final_model_params, 'ls')

    print("Federated learning completed. Final model parameters have been computed.")

def plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

    
def federated_learning_rounds(node_datasets, nodes, global_model, optimizers, scheduler, criterion, num_rounds=2):
    for round in range(num_rounds):
        print(f"Round {round + 1}:")
        node_params = []
        for dataset, node, optimizer in zip(node_datasets, nodes, optimizers):
            train_set = dataset.get_subset(dataset.train_indices)
            test_set = dataset.get_subset(dataset.test_indices)
            train_model(train_set, test_set, node, optimizer, scheduler, criterion, epochs=10, batch_size=32)
            node_params.append(copy.deepcopy(node.state_dict()))

            _, _, _, _, _, test_labels, test_preds = evaluate_model(node, DataLoader(test_set, batch_size=32, collate_fn=collate_fn), criterion)
            cm = confusion_matrix(test_labels, test_preds)
            plot_confusion_matrix(cm, dataset.class_names, title=f'Confusion Matrix for {dataset.root_dir} - Round {round + 1}')
            print(f"Confusion Matrix for {dataset.root_dir}:\n{cm}")
            

        model_params = aggregate_parameters(node_params)
        global_model.load_state_dict(model_params)

        # Print the global model aggregated parameters for each round
        print("Global Model Aggregated Parameters:")
        for name, param in global_model.named_parameters():
            print(f"{name}: {param.data.numpy()}")
        print()

        # Update node models with the new global model parameters
        for node in nodes:
            node.load_state_dict(global_model.state_dict())

    return global_model.state_dict()

def aggregate_parameters(node_params):
    aggregated_params = {}
    for key in node_params[0].keys():
        aggregated_params[key] = torch.zeros_like(node_params[0][key])
    
    for params in node_params:
        for key, value in params.items():
            aggregated_params[key] += value
    
    num_nodes = len(node_params)
    for key in aggregated_params.keys():
        aggregated_params[key] /= num_nodes

    return aggregated_params

if __name__ == '__main__':
    main()



