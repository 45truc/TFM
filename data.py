from utils import *
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold, train_test_split

class subjecData():
    def __init__(self,training_percentage=80):
        self.X_train = None
        self.X_test = None
        self.test_results = None
        self.training_percentage = training_percentage
        self.test_percentage = 100 - training_percentage
        self.num_nodes = None
        self.y_train = None
        self.y_test = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
    def build(self, subj=0, batch_size=32, data_path = f"E:\data/", Ffilter=False):

        #Load data
        X, Y_sample, Y_hot = Load_Data(subj=subj, data_path = data_path, Ffilter=Ffilter)
        self.num_nodes = X.shape[1]
        
        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y_hot, test_size=self.test_percentage/100, random_state=42)
        
        # Create training and test datasets
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        
        # Create DataLoader for training and test sets
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)