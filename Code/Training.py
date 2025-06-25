import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import average_precision_score, coverage_error, label_ranking_loss, hamming_loss, zero_one_loss
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')



def load_data(file_paths, label_path):
    features = []
    for file in file_paths:
        data = pd.read_csv(file).values

        # Normalize each feature set independently
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Convert the scaled data to torch.tensor and append to the features list
        features.append(torch.tensor(scaled_data, dtype=torch.float32))

    all_features = torch.cat(features)  

    labels = torch.tensor(pd.read_csv(label_path).values, dtype=torch.float32)

    return all_features, labels



# Define the TextCNN module
class TextCNN(nn.Module):
    def __init__(self, out_channels_1, out_channels_2, out_channels_3):
        super(TextCNN, self).__init__()
        # Define three different convolution layers with different kernel sizes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=out_channels_2, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=out_channels_3, kernel_size=5, padding=1)
        # Define a pooling layer to reduce the output size to 1 using AdaptiveMaxPool1d
        self.pool = nn.AdaptiveMaxPool1d(1)
        # Define a Dropout layer for regularization to prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        # Apply pooling to each convolutional output 
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        # Concatenate the outputs of the three convolutions
        x = torch.cat([x1, x2, x3])

        x = self.dropout(x)
        return x


# Define the BiLSTM module
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(BiLSTM, self).__init__()
        # Define a Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Forward pass through the LSTM
        output, _ = self.lstm(x)
        output = self.dropout(output)


        return output[:, -1, :]

    # Define the FeatureModule for handling individual features


class FeatureModule(nn.Module):
    def __init__(self, input_dim, out_channels, bilstm_hidden_dim):
        super(FeatureModule, self).__init__()
        # Initialize the TextCNN and BiLSTM modules
        self.textcnn = TextCNN(out_channels[0], out_channels[1], out_channels[2])
        self.bilstm = BiLSTM(input_dim, bilstm_hidden_dim)
        # Calculate the total dimension after concatenating the outputs
        total_dim = sum(out_channels) + 2 * bilstm_hidden_dim
        # Define the multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=total_dim, num_heads=4, batch_first=True)
        # Define the fully connected layer for classification
        self.fc = nn.Linear(total_dim, 9)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Pass the input through the TextCNN module
        x_textcnn = self.textcnn(x)

        # Pass the input through the BiLSTM module
        x_bilstm = self.bilstm(x)

        # Concatenate the outputs of TextCNN and BiLSTM
        x_concat = torch.cat([x_textcnn, x_bilstm])

        # Apply multi-head attention to the concatenated features
        x_attention, _ = self.attention(x_concat, x_concat, x_concat)
        # Apply Dropout to the attention output
        x_attention = self.dropout(x_attention)
        x_fc = self.fc(x_attention)
        return x_fc


# Define the mRSubLoc model
class mRSubLoc(nn.Module):
    def __init__(self, bilstm_hidden_dim):
        super(mRSubLoc, self).__init__()
        # Initialize feature modules for each type of feature (One-hot, Word2Vec, RNAErnie)
        self.onehot_feature_module = FeatureModule(280, [16, 32, 64], bilstm_hidden_dim)
        self.word2vec_feature_module = FeatureModule(768, [16, 32, 64], bilstm_hidden_dim)
        self.rna_ernie_feature_module = FeatureModule(768, [16, 32, 64], bilstm_hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(27, 18),  # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer for regularization
            nn.Linear(18, 9)
        )

    def forward(self, x):
        onehot_features = x[:, :280]
        word2vec_features = x[:, 280:1048]
        rna_ernie_features = x[:, 1048:]

        # Process each feature set through its corresponding feature module
        onehot_prob = torch.sigmoid(self.onehot_feature_module(onehot_features))
        word2vec_prob = torch.sigmoid(self.word2vec_feature_module(word2vec_features))
        rna_ernie_prob = torch.sigmoid(self.rna_ernie_feature_module(rna_ernie_features))

        # Concatenate the outputs of all feature modules
        new_features = torch.cat([onehot_prob, word2vec_prob, rna_ernie_prob])

        # Pass the concatenated features through the MLP for final classification
        final_output = self.mlp(new_features)
        return final_output


def calculate_metrics(L, L_pred):
    n, m = L.shape  # n: number of samples, m: number of labels

    # Aiming
    aiming = 0
    for v in range(n):
        intersection = 0
        for h in range(m):
            if L_pred[v, h] == 1 and L[v, h] == 1:
                intersection += 1
        if sum(L_pred[v]) == 0:
            continue
        aiming += intersection / sum(L_pred[v])
    aiming /= n

    # Accuracy
    accuracy = 0
    for v in range(n):
        intersection = 0
        union = 0
        for h in range(m):
            if L_pred[v, h] == 1 or L[v, h] == 1:
                union += 1
            if L_pred[v, h] == 1 and L[v, h] == 1:
                intersection += 1
        if union == 0:
            continue
        accuracy += intersection / union
    accuracy /= n

    # Coverage
    coverage = 0
    for v in range(n):
        intersection = 0
        for h in range(m):
            if L_pred[v, h] == 1 and L[v, h] == 1:
                intersection += 1
        if sum(L[v]) == 0:
            continue
        coverage += intersection / sum(L[v])
    coverage /= n

    # AbsoluteTrue
    absolute_true = 0
    for v in range(n):
        if list(L_pred[v]) == list(L[v]):
            absolute_true += 1
    absolute_true /= n

    # AbsoluteFalse
    absolute_false = 0
    for v in range(n):
        intersection = 0
        union = 0
        for h in range(m):
            if L_pred[v, h] == 1 or L[v, h] == 1:
                union += 1
            if L_pred[v, h] == 1 and L[v, h] == 1:
                intersection += 1
        absolute_false += (union - intersection) / m
    absolute_false /= n

    return aiming, coverage, accuracy, absolute_true, absolute_false



def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device):


    y_true_list = []
    y_pred_list = []

    # Training mode
    model.train()
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(features)  # Forward pass through the model
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()  # Update the model parameters

    # Validation mode
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation during validation
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            preds = torch.sigmoid(outputs).round()  # Apply sigmoid and round to get binary predictions
            y_true_list.append(labels)  # Append true labels to the list
            y_pred_list.append(preds)  # Append predicted labels to the list

    # Stack true and predicted labels from the lists to form arrays
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Calculate evaluation metrics
    aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value = calculate_metrics(y_true,
                                                                                                                y_pred)

    return aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value



def main():
    feature_files = ['./Data/TrainingData_Onehot.csv',
                     './Data/TrainingData_word2vec.csv',
                     './Data/TrainingData_RNAErnie.csv']
    label_file = './Data/TrainingLabel.csv'
    bilstm_hidden_dim = 64  # Hidden dimension for BiLSTM
    batch_size = 64
    learning_rate = 0.0001
    k_folds = 5
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load data
    features, labels = load_data(feature_files, label_file)

    features, labels = shuffle(features, labels, random_state=43)

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=43)


    global_best_model = None

    # Dictionary to store metrics across all folds
    fold_metrics_all = {
        "aiming": [],
        "coverage": [],
        "accuracy": [],
        "absolute_true": [],
        "absolute_false": []
    }

    # K-Fold Cross-Validation outer loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(labels), 1):

        # Split the data into training and validation sets
        train_features = features[train_idx]
        val_features = features[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        # DataLoader for training and validation sets
        train_dataset = RNADataset(train_features, train_labels)
        val_dataset = RNADataset(val_features, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate the model
        model = mRSubLoc(
            bilstm_hidden_dim=bilstm_hidden_dim
        ).to(device)

        # Loss function and optimizer
        train_label_tensor = torch.tensor(labels, dtype=torch.float32)
        pos_counts = train_label_tensor.sum(dim=0)
        total_counts = train_label_tensor.shape[0]
        neg_counts = total_counts - pos_counts
        pos_weight = (neg_counts / (pos_counts + 1e-6)).to(device)

        # Loss function
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Dictionary to store metrics for each fold
        fold_metrics = {
            "aiming": [],
            "coverage": [],
            "accuracy": [],
            "absolute_true": [],
            "absolute_false": []
        }

        # Loop over epochs for the current fold
        for epoch in range(epochs):
            # Train and validate the model
            aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value = train_and_validate(
                model, train_loader, val_loader, criterion, optimizer, device)

            # Print metrics for the current fold and epoch
            print(f"Fold {fold}, Epoch {epoch + 1} : "
                  f"Aiming: {aiming_value:.4f}, "
                  f"Coverage: {coverage_value:.4f}, "
                  f"Accuracy: {accuracy_value:.4f}, "
                  f"Absolute True: {absolute_true_value:.4f}, "
                  f"Absolute False: {absolute_false_value:.4f}")

            # Store metrics for each epoch in the current fold
            fold_metrics["aiming"].append(aiming_value)
            fold_metrics["coverage"].append(coverage_value)
            fold_metrics["accuracy"].append(accuracy_value)
            fold_metrics["absolute_true"].append(absolute_true_value)
            fold_metrics["absolute_false"].append(absolute_false_value)


        # Store the metrics for the current fold in the global dictionary
        for metric in fold_metrics_all:
            fold_metrics_all[metric].append(np.mean(fold_metrics[metric]))

    # Calculate the average metrics across all folds
    avg_fold_metrics = {metric: np.mean(values) for metric, values in fold_metrics_all.items()}
    print(f"\nAverage metrics across all folds: "
          f"Aiming: {avg_fold_metrics['aiming']:.4f}, "
          f"Coverage: {avg_fold_metrics['coverage']:.4f}, "
          f"Accuracy: {avg_fold_metrics['accuracy']:.4f}, "
          f"Absolute True: {avg_fold_metrics['absolute_true']:.4f}, "
          f"Absolute False: {avg_fold_metrics['absolute_false']:.4f}")

    if not os.path.exists('models'):
        print("Creating 'models' directory.")
        os.makedirs('models')

    model_save_path = os.path.join('models', 'mRSubLoc.pth')
    print(f"Saving model to {model_save_path}")
    torch.save(global_best_model, model_save_path)


if __name__ == "__main__":
    main()


