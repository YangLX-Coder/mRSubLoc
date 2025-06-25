import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
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



# Test function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()

    # Initialize lists to store the true labels and predicted labels
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            preds = torch.sigmoid(outputs).round()

            # Append the true labels and predicted labels to the respective lists (moving to CPU for numpy compatibility)
            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

            # Concatenate all the true labels and predicted labels into arrays for metric calculation
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # Calculate various performance metrics using the true and predicted labels
    aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value = calculate_metrics(y_true,
                                                                                                                y_pred)

    # Return the calculated metrics
    return aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value


# Main function to load data, load the model, and evaluate on the test set
def main_test():
    # Define file paths for test data
    feature_files = [
        './Data/TestData_Onehot.csv',
        './Data/TestData_word2vec.csv',
        './Data/TestData_RNAErnieFT.csv'
    ]
    label_file = './Data/TestLabel.csv'
    model_path = './models/mRSubLoc.pth'
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test data from CSV files
    test_features, test_labels = load_data(feature_files, label_file)
    # Create a dataset and DataLoader for batching
    test_dataset = RNADataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model structure and move it to the selected device (GPU or CPU)
    model = mRSubLoc(bilstm_hidden_dim=64).to(device)

    # Load the model weights from the saved file
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")

    # Evaluate the model on the test data
    aiming_value, coverage_value, accuracy_value, absolute_true_value, absolute_false_value = evaluate_model(
        model, test_loader, device)

    # Print the test results with the calculated metrics
    print(
        f"Test Results: "
        f"Aiming: {aiming_value:.4f}, "
        f"Coverage: {coverage_value:.4f}, "
        f"Accuracy: {accuracy_value:.4f}, "
        f"Absolute True: {absolute_true_value:.4f}, "
        f"Absolute False: {absolute_false_value:.4f},"
    )


if __name__ == "__main__":
    main_test()

