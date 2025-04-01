import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import load_MNIST, load_CIFAR, get_trainable_layers, filter_correct_classifications, filter_val_set
from model_nn.lenet5 import LeNet5
from coverages.idc import ImportanceDrivenCoverage
from TorchLRP.lrp import Sequential, Linear, Conv2d, MaxPool2d, convert_vgg
from TorchLRP import lrp

from coverages import coverage, tool

__version__ = 0.9

def nc_demo(args):
    train_loader, test_loader = load_CIFAR(batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512), 
        nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(512, 10))
    model = model.to(device)
    img_rows, img_cols = 224, 224
    input_size = (1, 3, img_rows, img_cols)
    random_input = torch.randn(input_size).to(device)
    layer_size_dict = tool.get_layer_output_sizes(model, random_input)
    criterion = coverage.NLC(model, layer_size_dict, hyper=None)
    criterion.build(train_loader)
    criterion.assess(test_loader)
    cov = criterion.current
    print(cov.cpu().numpy())


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded from", model_path)
    return model

def get_mnist_model():
    model = Sequential(
        Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        MaxPool2d(2,2),
        nn.Flatten(),
        Linear(14*14*64, 512),
        nn.ReLU(),
        Linear(512, 84),
        nn.ReLU(),
        Linear(84, 10)
    )
    return model

def get_lenet5_model():
    model = Sequential(
        Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        nn.ReLU(),
        MaxPool2d(2, 2),
        Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.ReLU(),
        MaxPool2d(2, 2),
        nn.Flatten(),
        Linear(in_features=16*4*4, out_features=120),
        nn.ReLU(),
        Linear(in_features=120, out_features=84),
        nn.ReLU(),
        Linear(in_features=84, out_features=10)
        # nn.Softmax(dim=1)
    )
    return model

def prepare_mnist_model(model, train_loader, test_loader, model_path="./examples/mnist_model.pth", epochs=1, lr=1e-3, train_new=False, device="cpu", is_test=False):

    if os.path.exists(model_path): 
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    if train_new: 
        model = model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for e in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
                if i%10 == 0: 
                    print("\r[%i/%i, %i/%i] loss: %.4f acc: %.4f" % (e, epochs, i, len(train_loader), loss.item(), acc.item()), end="", flush=True)
        torch.save(model.state_dict(), model_path)
    
    if is_test:
        model.eval()
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            acc = (y == y_hat.max(1)[1]).float().sum() / x.size(0)
            print("Testing Accuracy: %.4f" % acc.item())
    else:
        print("Skipping testing...")
    
    return model

def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Coverage Analyzer for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", default="mnist", help="Model name.\
                        The specified model will be used. choices=['mnist', 'lenet1','lenet4', 'lenet5']")
    parser.add_argument("-MP", "--model-path", default="./model_nn/lenet5_mnist.pth", help="Path to the model to be loaded.")
    parser.add_argument("-DS", "--dataset", default="mnist", help="The dataset to be used (mnist\
                        or cifar10).", choices=["mnist","cifar10"])
    parser.add_argument("-A", "--approach", default="idc", help="the approach to be employed \
                        to measure coverage", choices=['idc','nc','kmnc',
                        'nbc','snac','tknc','ssc', 'lsa', 'dsa'])
    parser.add_argument("-C", "--selected-class", default=-1, help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", default=3, help="quantization granularity for \
                        combinatorial other_coverage_metrics. (default: 3)", type= int)
    parser.add_argument("-L", "--layer", default=-1, help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-KS", "--k-sections", default=1000, help="number of sections used in \
                        k multisection other_coverage_metrics (default: 1000)", type=int)
    parser.add_argument("-KN", "--k-neurons", default=3, help="number of neurons used in \
                        top k neuron other_coverage_metrics (default: 3)", type=int)
    parser.add_argument("-RN", "--rel-neurons", default=10, help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics (default: 2)", type=int)
    parser.add_argument("-AT", "--act-threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", default=1, help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times) (default: 1)", type=int)
    parser.add_argument("-B", "--batch-size", type=int, default=256)
    parser.add_argument("-E", "--epochs", type=int, default=5)
    parser.add_argument("-LOG", "--logfile", default="result.log", help="path to log file")
    parser.add_argument("-ADV", "--advtype", default="fgsm", help="path to log file")
    parser.add_argument("--train-new", action="store_true", help="Train a new model")

    # parse command-line arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(vars(args))

    logfile = open(args.logfile, 'a')
    num_samples_plot = min(args.batch_size, 9)

    ####################
    # 0) Load data
    if args.dataset == 'mnist':
        train_loader, test_loader = load_MNIST(batch_size=args.batch_size, channel_first=False, train_all=True)
        img_rows, img_cols = 28, 28
    else:
        train_loader, test_loader = load_CIFAR(batch_size=args.batch_size)
        img_rows, img_cols = 224, 224

    ####################
    # 1) Setup the model
    if args.model == 'lenet5':
        model = get_lenet5_model()
        model = prepare_mnist_model(model, train_loader, test_loader, model_path="./examples/lenet5_model.pth", epochs=args.epochs, train_new=args.train_new)
            
    # Load a simple cnn (mnist training) for testing
    elif args.model == 'mnist':
        model = get_mnist_model()
        model = prepare_mnist_model(model, train_loader, test_loader, epochs=args.epochs, train_new=args.train_new)
    
    # Load a vgg16 for testing
    else:
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512), 
                      nn.ReLU(), 
                      nn.Dropout(0.5),
                      nn.Linear(512, 10))
        model.eval()
        model = lrp.convert_vgg(model)

    # For viewing
    trainable_layers, non_trainable_layers = get_trainable_layers(model)
    print("Trainable layers: ", trainable_layers)
    print("Non-trainable layers: ", non_trainable_layers)


    ####################
    # 3) Analyze Coverages
    if args.approach == 'nc':
        # A demo testing the Neuron Coverage [PASS]
        nc_demo(args)

    elif args.approach == 'idc':
        print("Running IDC for relevant neurons")
        idc = ImportanceDrivenCoverage(model=model, 
            batch_size=args.batch_size, 
            train_loader=train_loader, 
            test_loader=test_loader,
            subject_layer=args.layer,
            num_rel=args.rel_neurons,
            rule="epsilon",
            is_plot=False,
            device='cpu')
        all_patterns_path = None
        pos_patterns_path = None
        if args.model == 'mnist':
            all_patterns_path = "./examples/patterns/pattern_all.pkl"
            pos_patterns_path = "./examples/patterns/pattern_pos.pkl"
        elif args.model == 'lenet5':
            all_patterns_path = "./examples/patterns/lenet5_pattern_all.pkl"
            pos_patterns_path = "./examples/patterns/lenet5_pattern_pos.pkl"
        coverage, covered_combinations = idc.test(all_patterns_path=all_patterns_path, pos_patterns_path=pos_patterns_path)
        
        coverage, covered_combinations = idc.test(X_test)
        print("Your test set's coverage is: ", coverage)
        print("Number of covered combinations: ", len(covered_combinations))

        idc.set_measure_state(covered_combinations)
    
    elif args.approach == 'kmnc' or args.approach == 'nbc' or args.approach == 'snac':
        pass
    
    elif args.approach == 'tknc':
        pass

    elif args.approach == 'ssc':
        pass
    
    elif args.approach == 'lsa' or args.approach == 'dsa':
        pass

    logfile.close()