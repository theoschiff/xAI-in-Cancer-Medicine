import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from scipy.stats import spearmanr


def train_neural_net(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    Y_test,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    epochs = 10,
    model_path = '../models/model.pth',
    ):
    """
    Train the neural network model.
    
    Args:
    
    model: torch.nn.Module object. The neural network model.
    optimizer: torch.optim object. The optimizer.
    scheduler: torch.optim.lr_scheduler object. The learning rate scheduler.
    train_loader: DataLoader object. The training data loader.
    test_loader: DataLoader object. The testing data loader.
    Y_test: torch.Tensor. The testing labels.
    device: str. The device to use for training. Default is 'cuda' if available, else 'cpu'.
    epochs: int. The number of epochs to train the model. Default is 10.
    model_path: str. The path to save the model. Default is '../models/model.pth'.
    """
    # Initialize tracking variables
    best_val_loss = float('inf') 
    best_spearman = float('-inf') # Start with a very high value
    train_loss_history = []
    val_loss_history = []

    do_train = do_test = True

    for epoch in range(epochs):
        if do_train:
            train_sum_loss = 0
            model.train()

            with tqdm(total=len(train_loader)) as pbar:
                for batch_idx, (data, y) in enumerate(train_loader):
                    data = data.float()
                    data = data.to(device)
                    data = data.view(-1, 1, data.size(-1))
                    y = y.float()
                    y = y.to(device)
                    
                    train_output = model.forward(data)
                    train_loss = F.mse_loss(train_output.view(-1), y)
                    train_sum_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    optimizer.step()
                    pbar.set_description(f'Epoch {epoch + 1} - Processing Batch {batch_idx + 1}')
                    pbar.update(1)

                train_avg_loss = train_sum_loss / len(train_loader)
                train_loss_history.append(train_avg_loss)
        
                if do_test:
                    test_sum_loss = 0
                    model.eval()
                    predictions = []
                    with torch.no_grad():
                        for batch_count, (data, y) in enumerate(test_loader):
                            data = data.float()
                            data = data.to(device)
                            y = y.float()
                            y = y.to(device)
                            test_output = model.forward(data)
                            mse_loss = F.mse_loss(test_output.view(-1), y)
                            test_sum_loss += mse_loss.item()    
                            predictions.append(test_output.cpu().numpy()[0][0])            


                    test_avg_loss = test_sum_loss / len(test_loader)
                    val_loss_history.append(test_avg_loss)
                    # print(predictions[:5])
                    # print(Y_test.numpy()[:5])
                    current_spearman = spearmanr(predictions, Y_test)[0]

                    # Check if this is the best validation loss
                    model_saved = False
                    if current_spearman > best_spearman:
                        best_spearman = current_spearman
                        # Save the model
                        torch.save(model, model_path)
                        model_saved = True
                        
                    pbar.set_postfix({
                        'Train Loss': f'{train_avg_loss:.4f}',
                        'Val Loss': f'{test_avg_loss:.4f}',
                        'Spearman': f'{current_spearman:.4f}',
                        'Saved': 'Yes' if model_saved else 'No'
                    })

                    scheduler.step(test_avg_loss)
                    
                    
def train_neural_net_classes(
    model,
    distance,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    Y_test,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    epochs = 10,
    model_path = '../models/model.pth',
):
    """
    Train the neural network model in a classification setting. This code is very similar to the regression setting, but with a few changes.
    
    Args:
    
    model: torch.nn.Module object. The neural network model.
    optimizer: torch.optim object. The optimizer.
    scheduler: torch.optim.lr_scheduler object. The learning rate scheduler.
    train_loader: DataLoader object. The training data loader.
    test_loader: DataLoader object. The testing data loader.
    Y_test: torch.Tensor. The testing labels.
    device: str. The device to use for training. Default is 'cuda' if available, else 'cpu'.
    epochs: int. The number of epochs to train the model. Default is 10.
    model_path: str. The path to save the model. Default is '../models/model.pth'.
    """
    # Initialize tracking variables
    best_val_loss = float('inf') 
    best_spearman = float('-inf') # Start with a very high value
    train_loss_history = []
    val_loss_history = []

    do_train = do_test = True

    for epoch in range(epochs):
        if do_train:
            train_sum_loss = 0
            model.train()

            with tqdm(total=len(train_loader)) as pbar:
                for batch_idx, (data, y) in enumerate(train_loader):
                    data = data.float()
                    data = data.to(device)
                    data = data.view(-1, 1, data.size(-1))
                    y = y.to(torch.long)
                    y = y.to(device)
                    train_output = model.forward(data)
                    train_output = train_output.squeeze(1)
                    train_loss = distance(train_output, y)
                    train_sum_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    optimizer.step()
                    pbar.set_description(f'Epoch {epoch + 1} - Processing Batch {batch_idx + 1}')
                    pbar.update(1)

                train_avg_loss = train_sum_loss / len(train_loader)
                train_loss_history.append(train_avg_loss)
        
                if do_test:
                    test_sum_loss = 0
                    model.eval()
                    predictions = []
                    with torch.no_grad():
                        for batch_count, (data, y) in enumerate(test_loader):
                            data = data.float()
                            data = data.to(device)
                            y = y.to(torch.long)
                            y = y.to(device)
                            test_output = model.forward(data)
                            test_output = test_output.squeeze(1)
                            test_loss = distance(test_output, y)
                            test_sum_loss += test_loss.item()

                            predictions.append(test_output.argmax(dim=-1).cpu().numpy()[0])            


                    test_avg_loss = test_sum_loss / len(test_loader)
                    val_loss_history.append(test_avg_loss)
                    # print(predictions[:5])
                    # print(Y_test.numpy()[:5])
                    current_spearman = spearmanr(predictions, Y_test)[0]

                    # Check if this is the best validation loss
                    model_saved = False
                    if current_spearman > best_spearman:
                        best_spearman = current_spearman
                        # Save the model
                        torch.save(model, model_path)
                        model_saved = True
                        
                    pbar.set_postfix({
                        'Train Loss': f'{train_avg_loss:.4f}',
                        'Val Loss': f'{test_avg_loss:.4f}',
                        'Spearman': f'{current_spearman:.4f}',
                        'Saved': 'Yes' if model_saved else 'No'
                    })

                    scheduler.step(test_avg_loss)
                    
                    
def train_auto_encoder(
    encoder,
    decoder,
    distance,
    optimizer,
    scheduler,
    ae_train_loader,
    ae_test_loader,
    do_train = True,
    do_test = True,
    add_sparsity = False,
    reg_param = 0.2,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    epochs = 10,
    model_path = '../models/model.pth',
):
    """
    Train the autoencoder model. Only the encoder is saved for future use.
    
    Args:
        encoder: torch.nn.Module object. The encoder model.
        decoder: torch.nn.Module object. The decoder model.
        distance: torch.nn.Module object. The distance function to use.
        optimizer: torch.optim object. The optimizer.
        scheduler: torch.optim.lr_scheduler object. The learning rate scheduler.
        ae_train_loader: DataLoader object. The training data loader.
        ae_test_loader: DataLoader object. The testing data loader.
        do_train: bool. Whether to train the model. Default is True.
        do_test: bool. Whether to test the model. Default is True.
        add_sparsity: bool. Whether to add sparsity to the model. Default is False.
        reg_param: float. The regularization parameter. Default is 0.2.
        device: str. The device to use for training. Default is 'cuda' if available, else 'cpu'.
        epochs: int. The number of epochs to train the model. Default is 10.
        model_path: str. The path to save the model. Default is '../models/model.pth'.    
    """
    # Initialize tracking variables
    best_val_loss = float('inf')  # Start with a very high value
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        if do_train:
            train_sum_loss = 0
            encoder.train()
            decoder.train()

            with tqdm(total=len(ae_train_loader)) as pbar:
                for batch_idx, data in enumerate(ae_train_loader):
                    data = data.float()
                    data = data.to(device)
                    encoded = encoder.forward(data)
                    train_output = decoder.forward(encoded)

                    mse_loss = distance(train_output, data)
                    
                    if add_sparsity : 
                        model_children = list(encoder.children()) + list(decoder.children())

                        l1_loss=0
                        values=data
                        for i in range(len(model_children)):
                            values = F.leaky_relu((model_children[i](values)))
                            l1_loss += torch.mean(torch.abs(values))
                        # add the sparsity penalty
                        train_loss = mse_loss + reg_param * l1_loss
                    else:
                        train_loss = mse_loss
                    
                    train_sum_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    optimizer.step()
                    pbar.set_description(f'Epoch {epoch + 1} - Processing Batch {batch_idx + 1}')
                    pbar.update(1)

                scheduler.step()
                train_avg_loss = train_sum_loss / len(ae_train_loader)
                train_loss_history.append(train_avg_loss)
        
                if do_test:
                    test_sum_loss = 0

                    with torch.no_grad():
                        for batch_count, data in enumerate(ae_test_loader):
                            data = data.float()
                            data = data.to(device)
                            encoded = encoder.forward(data)
                            test_output = decoder.forward(encoded)

                            mse_loss = distance(test_output, data)
                            test_sum_loss += mse_loss.item()
                            

                    test_avg_loss = test_sum_loss / len(ae_test_loader)
                    val_loss_history.append(test_avg_loss)

                    # Check if this is the best validation loss
                    model_saved = False
                    if test_avg_loss < best_val_loss:
                        best_val_loss = test_avg_loss
                        # Save the model
                        model_saved = True
                        torch.save(encoder, model_path)
                        
                pbar.set_postfix({
                        'Train Loss': f'{train_avg_loss:.4f}',
                        'Val Loss': f'{test_avg_loss:.4f}',
                        'Saved': 'Yes' if model_saved else 'No'
                    })
                
                
def train_classification_head(
    encoder,
    classification_head,
    distance,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    Y_test,
    do_train = True,
    do_test = True,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    epochs = 10,
    model_path = '../models/model.pth',
):
    # Initialize tracking variables
    best_val_loss = float('inf') 
    best_spearman = float('-inf') # Start with a very high value
    train_loss_history = []
    val_loss_history = []

    do_train = do_test = True

    for epoch in range(epochs):
        if do_train:
            train_sum_loss = 0
            classification_head.train()

            with tqdm(total=len(train_loader)) as pbar:
                for batch_idx, (data, y) in enumerate(train_loader):
                    data = data.float()
                    data = data.to(device)
                    y = y.to(torch.long)
                    y = y.to(device)
                    
                    encoded = encoder.forward(data)
                    train_output = classification_head.forward(encoded)
                    train_loss = distance(train_output, y)
                    train_sum_loss += train_loss.item()

                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    optimizer.step()
                    pbar.set_description(f'Epoch {epoch + 1} - Processing Batch {batch_idx + 1}')
                    pbar.update(1)

                train_avg_loss = train_sum_loss / len(train_loader)
                train_loss_history.append(train_avg_loss)
        
                if do_test:
                    test_sum_loss = 0
                    spearmanr_sum = 0
                    classification_head.eval()
                    predictions = []
                    with torch.no_grad():
                        for batch_count, (data, y) in enumerate(test_loader):
                            data = data.float()
                            data = data.to(device)
                            y = y.to(torch.long)
                            y = y.to(device)
                            
                            encoded = encoder.forward(data)
                            test_output = classification_head.forward(torch.relu(encoded))
                            mse_loss = distance(test_output, y)
                            test_sum_loss += mse_loss.item()                
                            predictions.append(test_output.argmax(dim=-1).cpu().numpy()[0])      

                    test_avg_loss = test_sum_loss / len(test_loader)
                    val_loss_history.append(test_avg_loss)
                    current_spearman = spearmanr(predictions, Y_test)[0]
                    
                    # print(predictions[:5])
                    # print(Y_test.numpy()[:5])

                    # Check if this is the best validation loss
                    model_saved = False
                    if current_spearman > best_spearman:
                        best_spearman = current_spearman
                        # Save the model
                        model_saved = True
                        torch.save(classification_head, model_path)

                    scheduler.step(test_avg_loss)
                    
                    pbar.set_postfix({
                        'Train Loss': f'{train_avg_loss:.4f}',
                        'Val Loss': f'{test_avg_loss:.4f}',
                        'Saved': 'Yes' if model_saved else 'No'
                    })