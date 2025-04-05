import torch
import torch.nn as nn
import torchvision.transforms as transforms



from torch import optim
from torchvision import datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torch.autograd import Variable

#import matplotlib.pylab as plt
import time

import nnIndex 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = input('Enter the neural network model (CNN/MLP/RESNET18): ').strip().upper()

loss_list = []
DATASET_CONFIGS = {
    'MNIST': {
        'dataset': datasets.MNIST,
        'classes': 10,
        'in_chnls': 1,
        'batch_train': 32,
        'batch_test': 16,
        'n_epochs': 8,
        'transform': transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to match CIFAR
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
    },
    'CIFAR10': {
        'dataset': datasets.CIFAR10,
        'classes': 10,
        'in_chnls': 3,
        'batch_train': 32,
        'batch_test': 16,
        'n_epochs': 40,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
        ])
    },
    'CIFAR100': {
        'dataset': datasets.CIFAR100,
        'classes': 100,
        'in_chnls': 3,
        'batch_train': 32,
        'batch_test': 16,
        'n_epochs': 100,
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],[0.2673, 0.2564, 0.2762])
        ])
    }
}

if model_name == 'RESNET18':
    DATASET_CONFIGS['MNIST']['transform'] = transforms.Compose([
            transforms.Resize(224), 
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
    DATASET_CONFIGS['CIFAR10']['transform'] = transforms.Compose([
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616])
        ])
    DATASET_CONFIGS['CIFAR100']['transform'] = transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409],[0.2673, 0.2564, 0.2762])
        ])

def get_dataloader(dataset_name, test_true):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f'Invalid dataset name. Choose from {list(DATASET_CONFIGS.keys())}')

    config = DATASET_CONFIGS[dataset_name]  # Get dataset settings
    dataset = config['dataset'](root='data', train=test_true, transform=config['transform'], download=True)
    if test_true == True:
        batch_size = config['batch_test']
    else:
        batch_size = config['batch_train']
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    return dataloader, config['in_chnls'], config['classes'], config['n_epochs'], batch_size

dataset_name = input('Enter training dataset (MNIST/CIFAR10/CIFAR100): ').strip().upper()  # User selects dataset
print(f'Loaded {dataset_name} for training')

dataloader, in_chnls, classes, n_epochs, batch_size = get_dataloader(dataset_name, False)


if model_name == 'CNN':
    model = nnIndex.CNN(in_chnls, classes)
elif model_name == 'MLP':
    model = nnIndex.MLP(in_chnls, classes)
elif model_name == 'RESNET18':
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, classes)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(n_epochs, nn, dataloader):
    
    nn.to(device)
    nn.train()
        
    total_step = len(dataloader)
    for epoch in range(n_epochs):
        for i,(images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()    
            output = nn(images)      
            loss = loss_func(output, labels)       
            loss.backward()           
            optimizer.step()
            
            loss_ = loss.item()
            # if (i+1) % 100 == 0:
            #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            #             .format(epoch + 1, n_epochs, i + 1, total_step, loss.item()))
            # loss_list.append(loss_)   
            pass
        pass
    pass

def test(nn, dataloader):
    nn.to(device)
    nn.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            test_output = nn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item() 
            total += float(labels.size(0))
            
            pass
    pass
    accuracy = 100 * correct / total
    print(accuracy,'%')


for i in range(5):
    start_time = time.time()
    
    dataloader, in_chnls, classes, n_epochs, batch_size = get_dataloader(dataset_name, False)
    train(n_epochs, model, dataloader)
    end_time = time.time()
    print("Training Time: ", end_time - start_time)
    
    torch.cuda.empty_cache()
    dataloader, in_chnls, classes, n_epochs, batch_size = get_dataloader(dataset_name, True)
    test(model, dataloader)

# Plot the cost and accuracy
# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.plot(loss_list, color=color)
# ax1.set_xlabel('epoch', color=color)
# ax1.set_ylabel('Loss', color=color)
# ax1.tick_params(axis='y', color=color)

# plt.show()