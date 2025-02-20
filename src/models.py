import torch
from torch import nn
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader


def init_weights(m):
    """Initialize the weights of the model with Xavier initialization.
    
    Args:
        m: torch.nn.Module object.
        
    Returns:
        torch.nn.Module object with initialized weights.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
          
        
def count_parameters(model):
    """ Count the number of trainable parameters in the model.
    
    Args:
        model: torch.nn.Module object.
        
    Returns:
        int: Total number of trainable parameters.
        Result is shown in a pretty table.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params    


def load_model(model, model_path):
    """Load the model from the specified path.
    
    Args:
        model: torch.nn.Module object.
        model_path: str. Path to the model.
        
    Returns:
        torch.nn.Module object with loaded weights.
    """
    model = torch.load(model_path)

    for param in model.parameters():
        param.requires_grad = False
        
    return model


class CustomDataset(Dataset):
    """Custom Dataset class for the data.
    
    Args:
        X: torch.Tensor. Features.
        y: torch.Tensor. Labels.
        
    Returns:
        torch.Tensor: Features and labels."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def get_datasets(X_train, Y_train, X_test, Y_test):
    """Get the train and test datasets and dataloaders.
    
    Args:
        X_train: torch.Tensor. Training features.
        Y_train: torch.Tensor. Training labels.
        X_test: torch.Tensor. Testing features.
        Y_test: torch.Tensor. Testing labels.
        
    Returns:
        train_dataset: CustomDataset object. Training dataset.
        train_loader: DataLoader object. Training dataloader.
        test_dataset: CustomDataset object. Testing dataset.
        test_loader: DataLoader object. Testing dataloader.
    """
    train_dataset = CustomDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_dataset = CustomDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    return train_dataset, train_loader, test_dataset, test_loader

def get_auto_encoder_datasets(X_train, X_test):
    ae_train_loader = DataLoader(X_train, batch_size=8, shuffle=True, num_workers=2)
    ae_test_loader = DataLoader(X_test, batch_size=16, shuffle=False, num_workers=2)
    return ae_train_loader, ae_test_loader
    
   

class NeuralNet_1(nn.Module):
    """Neural Network model with 2 hidden layers of 512 and 256 units. Dropout is applied after each layer. ReLU activation is used.
    
    Args:
        features: int. Number of features.
        
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        
        self.enc = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        ) 
        
    def forward(self, x):
        if len(x.shape) == 2:  # (batch_size, sequence_length)
            x = x.unsqueeze(1)
        x = self.enc(x)
        x = torch.sigmoid(x)
        return x
    
    

class NeuralNet_2(nn.Module):
    """Neural Network model with 2 hidden layers of 128 and 32 units. Dropout is applied after each layer. ReLU activation is used.
    
    Args:
        features: int. Number of features.
        
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        
        self.enc =nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
        
    def forward(self, x):
        if len(x.shape) == 2:  # (batch_size, sequence_length)
            x = x.unsqueeze(1)
        x = self.enc(x)
        x = torch.sigmoid(x)
        return x
    
    
class CNN(nn.Module):
    """Convolutional Neural Network model with 3 convolutional layers and 1 linear layer. LeakyReLU activation is used.
    
    Args:
        features: int. Number of features.
        
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4),
            nn.LeakyReLU(),
            nn.Conv1d(16, 64, 9, padding=4),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 9, padding=4),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(64, 1, 1),
            nn.Flatten(),
        )
        
    def forward(self, x):
        if len(x.shape) == 2:  # (batch_size, sequence_length)
            x = x.unsqueeze(1)
        x = self.enc(x)
        x = torch.sigmoid(x)
        return x
    
class NeuralNetClasses(nn.Module):
    """
    Neural Network model with 3 hidden layers of 384, 256, and 64 units. Dropout is applied after each layer. ReLU activation is used.
    
    Args:
        features: int. Number of features.
        num_classes: int. Number of classes.
        
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self, features, num_classes=10):
        super().__init__()
        
        self.features = features
        self.num_classes = num_classes
        
        self.enc = nn.Sequential(
            nn.Linear(features, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),  
        )
    
        
    def forward(self, x):
        x = self.enc(x)
        x = torch.softmax(x, dim=-1)  
        return x
    
    
class Encoder(nn.Module):
    """
    Encoder model with 1 hidden layer of 128 units. ReLU activation is used.
    
    Args:
        features: int. Number of features.
        
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        
        self.enc = nn.Sequential(
            nn.Linear(features, 128),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.enc(x)
        return x
    
    
class Decoder(nn.Module):
    """
    Decoder model with 1 hidden layer of 128 units. ReLU activation is used.
    
    Args:
        features: int. Number of features.
        
    Returns:
        torch.nn.Module object."""
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        
        self.dec = nn.Sequential(
            nn.Linear(128, features),
            nn.ReLU(),
        )
        
        
    def forward(self, x):
        x = self.dec(x) 
        x = torch.sigmoid(x)  
        return x
    
    
class ClassificationHead(nn.Module):
    """
    Classification model with 2 hidden layers of 128 and 64 units. ReLU activation is used.
    
    Args:
        features: int. Number of features.
    
    Returns:
        torch.nn.Module object.    
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 11),
        )
        
        
    def forward(self, x):
        x = self.fc(x)
        x = torch.softmax(x, dim=-1)  
        return x 