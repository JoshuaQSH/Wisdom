import os
import time
import copy
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from prepare_selector_data import get_layer_info, extract_features, CustomDataset

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from utils import parse_args, normalize_tensor, get_model, get_model_cifar, get_trainable_modules_main, save_model, Logger

class Selector(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(Selector, self).__init__()
        
        # Create a list of fully connected layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Define the sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

def save_model(model, model_name):
    torch.save(model.state_dict(), model_name + '.pt')
    print("Model saved as", model_name + '.pt')
            
def string_to_one_hot(attribution_methods, string_labels):
    method_to_idx = {method: idx for idx, method in enumerate(attribution_methods)}
    integer_label = method_to_idx[string_labels]
    return torch.tensor(integer_label, dtype=torch.long)

def load_dataset(csv_file='prepared_data.csv', attributions=['lfa', 'ldl'], batch_size=32):
    dataset = CustomDataset(csv_file, attributions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def prapared_parameters(args):
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'Selector-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, TopK: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
    model_path += args.saved_model
    
    ## Loading models - either 1) from scratch or 2) pretrained
    if args.dataset == 'cifar10' and args.model != 'lenet':
        model, module_name, module = get_model_cifar(model_name=args.model, load_model_path=model_path)
    else:
        # We aussume that the SOTA models are pretrained with IMAGENET
        model, module_name, module = get_model(model_name=args.model)
    
    # TODO: A Hack here for model loading
    model.load_state_dict(torch.load(model_path))
    selector_model = copy.deepcopy(model)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    return model, selector_model, module_name, module, trainable_module, trainable_module_name, device, log


def trace_plots(num_epochs, epoch_accuracies, epoch_losses, save_file_name='training_loss_accuracy_1.pdf'):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_file_name, format='pdf', dpi=1200)

def train_whole(selector_model,
               num_epochs,
               dataloader,
               device,
               save_file_name,
               log):
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(selector_model.parameters(), lr=0.0002)
    optimizer = optim.SGD(selector_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    selector_model.train()
    
    epoch_losses = []
    epoch_accuracies = []
    
    selector_model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        optimizer.zero_grad()
        
        for images, labels, layer_infos, optimal_methods in dataloader:
            attr_label = optimal_methods.to(device)
            images = images.to(device)
            # labels = labels.to(device)

            outputs = selector_model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, attr_label)
            loss.backward()
            optimizer.step()
            # running_loss += loss.item() * images.size(0)
            running_loss += loss.item() # SGD

            attr_label_indices = torch.argmax(attr_label, dim=1)
            running_corrects += torch.sum(preds == attr_label_indices.data)
            total_samples += images.size(0)
        
        scheduler.step()         
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc.item())
        
        if log is not None:
            log.logger.info("Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}".format(epoch+1, num_epochs, epoch_loss, epoch_acc))
        else:
            print('Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
    
    start_time = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
    trace_plots(num_epochs, epoch_accuracies, epoch_losses, save_file_name='{}_{}.pdf'.format(save_file_name, timestamp))
    return selector_model

# attributions, model, classes, net_layer, layer_name, top_m_neurons
def train(model,
               selector_model,
               num_epochs,
               dataloader,
               device,
               save_file_name,
               log):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(selector_model.parameters(), lr=0.00005)
    # optimizer = optim.SGD(selector_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    selector_model.train()
    
    epoch_losses = []
    epoch_accuracies = []
    
    selector_model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        optimizer.zero_grad()
        
        for images, labels, layer_infos, optimal_methods in dataloader:
            attr_label_ex = optimal_methods.to(device)
            labels = labels.to(device)
    
            ### Extract features using transfer learning (freeze feature extractor)
            features = extract_features(model, images)
            # layer_info_repeated = layer_info.unsqueeze(0).repeat(features[0].shape[0], 1)
            
            # Normalize the features
            features_norm = normalize_tensor(features[0])
            combined_features = torch.cat((features_norm, layer_infos), dim=1)
        
            ##  Train the final fc layer only
            outputs = selector_model(combined_features.to(device))
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, attr_label_ex)
            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # running_loss += loss.item() * images.size(0)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            attr_label_indices = torch.argmax(attr_label_ex, dim=1)
            running_corrects += torch.sum(preds == attr_label_indices.data)
            total_samples += images.size(0)
        
        # scheduler.step()   
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc.item())
        
        if log is not None:
            log.logger.info("Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}".format(epoch+1, num_epochs, epoch_loss, epoch_acc))
        else:
            print('Epoch [{}/{}], Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
    
    start_time = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
    trace_plots(num_epochs, epoch_accuracies, epoch_losses, save_file_name='{}_{}.pdf'.format(save_file_name,timestamp))
    return selector_model

def test_whole(selector_model, dataloader, save_file_name, log):
    
    selector_model.cpu()
    selector_model.load_state_dict(torch.load(save_file_name + '.pt'))
    selector_model.eval()  
    
    total_loss = 0.0
    running_loss = 0.0
    total_samples = 0
    correct_attr_all = 0
    
    selector_model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels, layer_infos, optimal_methods in dataloader:
            outputs = selector_model(images)
            _, preds_attr = torch.max(outputs, 1)
                        
            loss = criterion(outputs, optimal_methods)
            running_loss += loss.item() * images.size(0)
            
            attr_label_indices = torch.argmax(optimal_methods, dim=1)
            running_corrects_attr = torch.sum(preds_attr == attr_label_indices.data)

            correct_attr_all += running_corrects_attr
            total_samples += images.size(0)

                    
        total_loss = running_loss / total_samples
        total_acc_attr = correct_attr_all.double() / total_samples      

        if log is not None:
            log.logger.info("Test Loss (Attribution): {:.4f} Test Acc (Attribution): {:.4f}".format(total_loss, total_acc_attr))
        else:
            print("Test Loss (Attribution): {:.4f} Test Acc (Attribution): {:.4f}".format(total_loss, total_acc_attr))

def test(selector_model, model, dataloader, extract_layer_name, save_file_name, log):
    
    selector_model.cpu()
    selector_model.load_state_dict(torch.load(save_file_name + '.pt'))
    selector_model.eval()    
    
    total_loss = 0.0
    total_acc = 0.0
    running_loss = 0.0
    total_samples = 0
    correct_attr_all = 0
    correct_class_all = 0
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels, layer_infos, optimal_methods in dataloader:
            outputs_class = model(images)
            features = extract_features(model, images)
            features_norm = normalize_tensor(features[0])
            combined_features = torch.cat((features_norm, layer_infos), dim=1)
            outputs_attr = selector_model(combined_features)
            
            _, preds_class = torch.max(outputs_class, 1)
            _, preds_attr = torch.max(outputs_attr, 1)
                        
            loss = criterion(outputs_attr, optimal_methods)
            running_loss += loss.item() * images.size(0) # Adam
            # running_loss += loss.item() # SGD
            
            running_corrects_class = torch.sum(preds_class == labels.data)
            attr_label_indices = torch.argmax(optimal_methods, dim=1)
            running_corrects_attr = torch.sum(preds_attr == attr_label_indices.data)

            correct_attr_all += running_corrects_attr
            correct_class_all += running_corrects_class
            
            acc_class = running_corrects_class / images.shape[0]
            acc_attr = running_corrects_attr / images.shape[0]
            
            total_samples += images.size(0)

            if log is not None:
                # log.logger.info("Label: {}".format(labels))
                # log.logger.info("Preds: {}".format(preds_class))
                log.logger.info("Image Acc: {:.4f}, Attribution Acc: {:.4f}".format(acc_class, acc_attr))
            else:
                # print("Label: {}".format(labels))
                # print("Preds: {}".format(preds_class))
                print("Image Acc: {:.4f}, Attribution Acc: {:.4f}".format(acc_class, acc_attr))
                    
        total_loss = running_loss / total_samples
        total_acc_attr = correct_attr_all.double() / total_samples      
        total_acc_class = correct_class_all.double() / total_samples      

        if log is not None:
            log.logger.info("Test Acc (Images): {:.4f}".format(total_acc_class))
            log.logger.info("Test Loss (Attribution): {:.4f} Test Acc (Attribution): {:.4f}".format(total_loss, total_acc_attr))
        else:
            print("Test Acc (Images): {:.4f}".format(total_acc_class))
            print("Test Loss (Attribution): {:.4f} Test Acc (Attribution): {:.4f}".format(total_loss, total_acc_attr))
        
"""
python selector_train_v1.py --dataset cifar10 --batch-size 128 --layer-index 2 --model lenet --top-m-neurons 10 --all-attr
"""

if __name__ == '__main__':
    args = parse_args()
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'Training-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, TopK: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
    
    ### Model settings
    if args.model_path != 'None':
        model_path = args.model_path
    else:
        model_path = os.getenv("HOME") + '/torch-deepimportance/models_info/saved_models/'
    model_path += args.saved_model
    
    ## Loading models - either 1) from scratch or 2) pretrained
    if args.dataset == 'cifar10' and args.model != 'lenet':
        model, module_name, module = get_model_cifar(model_name=args.model, load_model_path=model_path)
    else:
        # We aussume that the SOTA models are pretrained with IMAGENET
        model, module_name, module = get_model(model_name=args.model)
    
    # TODO: A Hack here for model loading
    model.load_state_dict(torch.load(model_path))
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    # model, module_name, module, trainable_module, trainable_module_name, log = prapared_parameters(args)
    
    ### Device settings    
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        
    ### Saved the index for comparing the Common Neurons across the attributions
    ## {'lc': [1, 2, 3, 4, 5], 'la': [1, 2, 3, 4, 5], ...}
    # attributions = ['lc', 'la', 'ii', 'ldl', 'ldls', 'lgs', 'lig', 'lfa', 'lrp']
    attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
    attr_dict = {key: [] for key in attributions}
    original_state = copy.deepcopy(model.state_dict())
    layer_info = get_layer_info(trainable_module_name[args.layer_index], trainable_module_name)
    
    ### Hyperparemeters and layer settings
    num_out = len(attributions)
    # num_ft = trainable_module[-1].in_features
    selector_model = copy.deepcopy(model)
    num_epochs = 20
    for param in selector_model.parameters():
        param.requires_grad = False
    # selector_model.fc3 = nn.Linear(trainable_module[-1].in_features+layer_info.shape[0], num_out)
    # selector_model.fc3 = nn.Linear(trainable_module[-1].in_features, num_out)
    # trained_fc3 = selector_model.fc3
    trained_fc3 = Selector(trainable_module[-1].in_features+layer_info.shape[0], num_out, [256, 64, 32])
    
    ### Loading dataloader
    time_start = time.time()
    trainloader = load_dataset("prepared_data_train_cifar.csv", attributions, args.batch_size)
    testloader = load_dataset("prepared_data_test_cifar.csv", attributions, args.batch_size)
    print("Data loading time: ", time.time() - time_start)
    
    save_file_name = 'selector_{}_{}_2'.format(args.model, 'h3')
    
    ### Training the selector model
    # selector_trained = train_whole(model, selector_model, num_epochs, trainloader, device, save_file_name, log)
    # save_model(selector_trained, 'selector_whole')  
    # test_whole(selector_trained, testloader, save_file_name, log)

    selector_trained = train(model, trained_fc3, num_epochs, trainloader, device, save_file_name, log)
    save_model(selector_trained, save_file_name)
    test(selector_trained, model, testloader, trainable_module_name[-2], save_file_name, log)