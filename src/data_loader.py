import pandas as pd
import scipy.io
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def load_base_data():
    """Load the base data from the csv files.

    Returns:
        (x_train, y_train, x_predict): tuple of pandas.DataFrame objects containing the training features, training targets, and prediction features, respectively.
    """
    x_train = pd.read_csv('./data/train.csv', index_col=0)
    y_train = pd.read_csv('./data/train_targets.csv', index_col=0)
    x_predict = pd.read_csv('./data/test.csv', index_col=0)
    return x_train, y_train["AAC"], x_predict
    
    
def load_base_data_with_feature_subset():
    """Load the base data from the csv files.

    Returns:
        (x_train, y_train, x_predict): tuple of pandas.DataFrame objects containing the training features, training targets, and prediction features, respectively. 
        The features are a subset of the original features corresponding to features found online to be meaningful for Erlotinib drug response prediction.
    """
    x_train = pd.read_csv('./data/train.csv', index_col=0)
    y_train = pd.read_csv('./data/train_targets.csv', index_col=0)
    x_predict = pd.read_csv('./data/test.csv', index_col=0)

    x_augmented = pd.read_csv('./data/train_augmented.csv', index_col=0)
    x_train = x_train[x_augmented.columns] 
    x_predict = x_predict[x_augmented.columns]
    
    return x_train, y_train["AAC"], x_predict
    
    
def load_augmented_data():
    """Load the augmented data from the csv files.

    Returns:
        (x_train, y_train, x_predict): tuple of pandas.DataFrame objects containing the training features, training targets, and prediction features, respectively.
    """
    x_train = pd.read_csv('./data/train_augmented.csv', index_col=0)
    y_train = pd.read_csv('./data/train_targets_augmented.csv', index_col=0)
    x_predict = pd.read_csv('./data/test.csv', index_col=0)
    x_predict = x_predict[x_train.columns]
    return x_train, y_train["AAC"], x_predict


def create_mat_files_wpfs(x_train, y_train, x_predict):
    """Create the mat files for the WPFS. And convert the target values to integers in 10 different classes.

    Args:
        X_train: pandas.DataFrame containing the training features.
        y_train: pandas.DataFrame containing the training targets.
        X_test: pandas.DataFrame containing the prediction features.
    """
    data_dict = {
        'X' : x_train.values,
        'Y' : (y_train.round(1) * 10).values,
    }
    scipy.io.savemat('train.mat', data_dict)

    data_dict = {
        'X' : x_predict.values,
    }

    scipy.io.savemat('predict.mat', data_dict)
    
    
def train_test_split(x_train, y_train, test_size=0.2, random_state=42):
    """Split the data into training and testing sets.

    Args:
        x_train: pandas.DataFrame containing the training features.
        y_train: pandas.DataFrame containing the training targets.
        test_size: float, default=0.2. The proportion of the dataset to include in the test split.

    Returns:
        (x_train, x_test, y_train, y_test): tuple of pandas.DataFrame objects containing the training features, testing features, training targets, and testing targets, respectively.
    """
    return sklearn.model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)


def rescale_data_convert_to_tensor(X_train, X_test, x_predict, Y_train, Y_test, classes = False):
    """

    Args:
        X_train: pandas.DataFrame containing the training features.
        X_test: pandas.DataFrame containing the testing features.
        X_predict: pandas.DataFrame containing the prediction features.

    Returns:
        (X_train, X_test, X_predict): tuple of pandas.DataFrame objects containing the rescaled training features, testing features, and prediction features, respectively.
    """
    
    # Remove genes with median expression below 1
    features = X_train.columns
    genes_train_median_above_1_column_mask = np.median(X_train, axis=0) > 1
    features = features[genes_train_median_above_1_column_mask]

    X_train = X_train.values
    X_test = X_test.values
    X_predict = x_predict.values

    X_train = X_train[:, genes_train_median_above_1_column_mask]
    X_test = X_test[:, genes_train_median_above_1_column_mask]
    X_predict = X_predict[:, genes_train_median_above_1_column_mask]

    # Log2 transformation
    X_train = np.log2(X_train + 0.25)
    X_test = np.log2(X_test + 0.25)
    X_predict = np.log2(X_predict + 0.25)

    # MinMax scaling
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    genes_train_log2TPM_MinMaxScaler = scaler.transform(X_train)
    genes_test_log2TPM_MinMaxScaler = scaler.transform(X_test)
    X_predict = scaler.transform(X_predict)

    # Convert to tensors
    X_train = pd.DataFrame(genes_train_log2TPM_MinMaxScaler, columns=features)
    X_test = pd.DataFrame(genes_test_log2TPM_MinMaxScaler, columns=features)
    X_predict = pd.DataFrame(X_predict, columns=features)
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    X_predict = torch.tensor(X_predict.values, dtype=torch.float32)
    if classes:
        Y_train = torch.tensor(Y_train.values, dtype=torch.int8)
        Y_test = torch.tensor(Y_test.values, dtype=torch.int8)
    else:
        Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
        Y_test = torch.tensor(Y_test.values, dtype=torch.float32)

    return X_train, X_test, X_predict, Y_train, Y_test

    

def augment_and_concat_with_pca(data, noise_std=0.1):
    """Augment the data by adding Gaussian noise to the PCA components.

    Args:
        data : data to augment
        noise_std (float, optional): Noise to add. Defaults to 0.1.

    Returns:
        dataframe: Augmented data
    """
    
    pca = PCA(n_components=min(data.shape[0], data.shape[1]), random_state=42)
    pca_transformed = pca.fit_transform(data)
    
    # Add Gaussian noise to the PCA components
    noise = np.random.normal(0, noise_std, pca_transformed.shape)
    noisy_pca = pca_transformed + noise
    
    # Reconstruct the data from noisy PCA components
    reconstructed_data = pca.inverse_transform(noisy_pca)        
        
    # Concatenate original and augmented data
    combined_data = torch.cat([data, torch.tensor(reconstructed_data, dtype=torch.float32)], dim=0)
    
    return combined_data


def augment_data(X_train, X_test, Y_train, Y_test):
    """
    Augment the data by adding Gaussian noise to the PCA components.
    Done in 3 steps with increasing noise levels. The data is augmented by a factor of 8 in total.
    """
    X_train = augment_and_concat_with_pca(X_train, noise_std=0.1)
    X_test = augment_and_concat_with_pca(X_test, noise_std=0.1)
    Y_train = torch.cat([Y_train, Y_train], dim=0)
    Y_test = torch.cat([Y_test, Y_test], dim=0)

    X_train = augment_and_concat_with_pca(X_train, noise_std=0.2)
    X_test = augment_and_concat_with_pca(X_test, noise_std=0.2)
    Y_train = torch.cat([Y_train, Y_train], dim=0)
    Y_test = torch.cat([Y_test, Y_test], dim=0)

    X_train = augment_and_concat_with_pca(X_train, noise_std=0.3)
    X_test = augment_and_concat_with_pca(X_test, noise_std=0.3)
    Y_train = torch.cat([Y_train, Y_train], dim=0)
    Y_test = torch.cat([Y_test, Y_test], dim=0)
    
    return X_train, X_test, Y_train, Y_test


def apply_PCA(X_train, X_test, X_predict, n_components=500):
    """Apply PCA to the data.

    Args:
        X_train: pandas.DataFrame containing the training features.
        X_test: pandas.DataFrame containing the testing features.
        X_predict: pandas.DataFrame containing the prediction features.
        n_components: int, default=100. The number of components to keep.

    Returns:
        (X_train, X_test, X_predict): tuple of pandas.DataFrame objects containing the PCA-transformed training features, testing features, and prediction features, respectively.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_predict = pca.transform(X_predict)
    return X_train, X_test, X_predict


def spearmanr_neural_net(model, X_test, Y_test, device = 'cuda' if torch.cuda.is_available() else 'cpu', classes = False):
    """Calculate the Spearman correlation coefficient.
    
    Args:
        y_pred: predicted values.
        y_true: true values.
        
    Returns:
        float: Spearman correlation         
        
    """
    test_loader = DataLoader(X_test, batch_size=1, shuffle=False, num_workers=2)

    prediction = []
    with torch.no_grad():
        for batch_count, data in enumerate(test_loader):
            data = data.float()
            data = data.to(device)
            pred = model.forward(data)
            if classes:
                prediction.append(pred.argmax(dim=-1).cpu().numpy()[0])
            else:
                prediction.append(pred.cpu().numpy()[0][0][0])
            
    print(f"Spearmann R coefficient for neural net is : {spearmanr(prediction, Y_test)[0]}")
    
    
def spearmanr_AE(encoder, classification_head, X_test, Y_test, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Calculate the Spearman correlation coefficient for the AE architecure
    
    Args:
        y_pred: predicted values.
        y_true: true values.
        
    Returns:
        float: Spearman correlation         
        
    """
    test_loader = DataLoader(X_test, batch_size=1, shuffle=False, num_workers=2)

    prediction = []
    with torch.no_grad():
        for batch_count, data in enumerate(test_loader):
            data = data.to(device)
            data = data.float()
            encoded = encoder.forward(data)
            train_output = classification_head.forward(encoded)
            prediction.append(train_output.argmax(dim=-1).cpu().numpy()[0])
            
    print(f"Spearmann R coefficient for neural net is : {spearmanr(prediction, Y_test)[0]}")
                

def generate_submission_neural_net(X_predict, x_predict, model, device = 'cuda' if torch.cuda.is_available() else 'cpu', classes = False, path_to_file='./submissions/NeuralNet.csv'):
    test_loader = DataLoader(X_predict, batch_size=1, shuffle=False, num_workers=2)

    prediction = []
    with torch.no_grad():
        for batch_count, data in enumerate(test_loader):
            data = data.float()
            data = data.to(device)
            pred = model.forward(data)
            prediction.extend(pred.cpu().numpy())
            
    print(f"Predicted {len(prediction)} samples")
    
    submission = pd.DataFrame({"sampleId": x_predict.index, "AAC": prediction})
    submission["sampleId"] = submission["sampleId"].apply(lambda x: x.replace("CL", "TS"))
    if classes:
        submission["AAC"] = submission["AAC"].apply(lambda x: x[0])
    else:
        submission["AAC"] = submission["AAC"].apply(lambda x: x[0][0])
    submission.to_csv(path_to_file, index=False)
    
    
def generate_submission_AE(X_predict, x_predict, encoder, classification_head, device = 'cuda' if torch.cuda.is_available() else 'cpu', path_to_file='./submissions/AE.csv'):
    test_loader = DataLoader(X_predict, batch_size=8, shuffle=True, num_workers=2)

    prediction = []
    with torch.no_grad():
        for batch_count, data in enumerate(test_loader):
            data = data.float()
            data = data.to(device)
            encoded = encoder.forward(data)
            train_output = classification_head.forward(encoded)
            prediction.extend(train_output.argmax(dim=-1).cpu().numpy())
            
    print(len(prediction))
    
    submission = pd.DataFrame({"sampleId": x_predict.index, "AAC": prediction})
    submission["sampleId"] = submission["sampleId"].apply(lambda x: x.replace("CL", "TS"))
    submission["AAC"] = submission["AAC"].apply(lambda x: x/10)
    submission.to_csv(path_to_file, index=False)