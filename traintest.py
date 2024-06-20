from data import *
from tqdm import tqdm


def train_model(model, DataClass, num_epochs = 300, batch_size = 32,
                learning_rate = 0.001, save=False, path=None, path_metrics=None, verbose=True):
    # Initialize the model, loss function, and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    ps = []
    recs = []
    accs = []
    f1s = []
    Loss = []
    for epoch in tqdm(range(num_epochs), disable=not(verbose)):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, targets in DataClass.train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
            # Collect predictions and true labels
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(targets, dim=1).squeeze().cpu().numpy())
    
        # Calculate training metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        if verbose:
            print(f'Epoch {epoch+1}, Training Loss: {train_loss / len(DataClass.train_loader)}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')
        ps.append(precision)
        recs.append(recall)
        accs.append(accuracy)
        f1s.append(f1)
        Loss.append(train_loss / len(DataClass.train_loader))
        
    if save:
        torch.save(model.state_dict(), path)
        dic = {'Loss':Loss,'Accuracy':accs,'Recall':recs,'Precision':ps,'F1':f1s}
        df = pd.DataFrame.from_dict(dic)
        df.to_csv(path_metrics)
        return
        
    return model

def Eval_Model(model, DataClass):
    
    model.eval()
    pred = model(DataClass.X_test)
    pred = torch.argmax(pred, dim=1)
    y_Test = torch.argmax(DataClass.y_test, dim=1)

    # Calculate test metrics
    accuracy = accuracy_score(y_Test, pred)
    precision = precision_score(y_Test, pred, average=None)
    recall = recall_score(y_Test, pred, average=None)
    f1 = f1_score(y_Test, pred, average=None)
    
    print(f' Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')
    return accuracy, precision, recall, f1