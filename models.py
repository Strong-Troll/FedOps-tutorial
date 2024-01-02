from sklearn.metrics import f1_score
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm 

# Define your custom Model    
class MNISTClassifier(nn.Module):
    # To properly utilize the config file, the output_size variable must be used in __init__().
    def __init__(self, output_size):
        super(MNISTClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)  # Image size is 28x28, reduced to 14x14 and then to 7x7
        self.fc2 = nn.Linear(1000, output_size)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        """
        Train the network on the training set.
        Model must be the return value.
        """
        print("Starting training...")
            
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    
    def custom_test_torch(model, test_loader, cfg):
        """
        Validate the network on the entire test set.
        Loss, accuracy values, and dictionary-type metrics variables are fixed as return values.
        """
        print("Starting evalutation...")
        
        
        
        # if you use metrics, you set metrics
        # type is dict
        # for example, Calculate F1 score
        # Add F1 score to metrics
        # metrics = {"f1_score": f1}
        
        # If you don't use it
        metrics=None    
        
        model.to("cpu")  # move model back to CPU
        return average_loss, accuracy, metrics
    
    return custom_test_torch
