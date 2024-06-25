from data import *
from model import *
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

def Sub_TestResults_DF(subj = 0):
    
    DataClass = subjecData()
    DataClass.build(subj)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    NetDict = {'TimeAggNet':TimeAggNet,'TimeGraphNet':TimeGraphNet,'DeepTimeGraphNet':DeepTimeGraphNet,
               'SimpleTimeGraphNet':SimpleTimeGraphNet}
    j = 0
    ps = []
    recs = []
    accs = []
    f1s = []
    names = []
    
    for net_name in tqdm(NetDict.keys()):
        net = NetDict[net_name](device, DataClass.num_nodes, 1200)
        model_path = f"./training/models/sub{subj}/{net_name}.pth"
        #metrics_path = f"./training/metrics/sub{subj}/metrics{net_name}.csv"
        net.load_state_dict(torch.load(model_path))
        accuracy, precision, recall, f1 = Eval_Model(net, DataClass)
        ps.append(precision)
        recs.append(recall)
        accs.append(accuracy)
        f1s.append(f1)
        names.append(net_name)
        
        if j>=1:
            net = NetDict[net_name](device, DataClass.num_nodes, 1200, True)
            model_path = f"./training/models/sub{subj}/{net_name}Adap.pth"
            #metrics_path = f"./training/metrics/sub{subj}/metrics{net_name}Adap.csv"
            net.load_state_dict(torch.load(model_path))
            accuracy, precision, recall, f1 = Eval_Model(net, DataClass)
            ps.append(precision)
            recs.append(recall)
            accs.append(accuracy)
            f1s.append(f1)
            names.append(net_name+'Adap')
            
        j = j+1

    recs = np.array(recs)
    ps = np.array(ps)
    f1s = np.array(f1s)
    
    dic = {'Accuracy':accs,'Recall class 0':recs[:,0],'Recall class 1':recs[:,1],'Recall class 2':recs[:,2],
           'Precision class 0':ps[:,0],'Precision class 1':ps[:,1],'Precision class 2':ps[:,2],'F1 class 0':f1s[:,0],
           'F1 class 1':f1s[:,1],'F1 class 2':f1s[:,2]}
    df = pd.DataFrame.from_dict(dic)
    df.index = names

    return df