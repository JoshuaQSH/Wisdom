import os
import time
import copy
from pathlib import Path
import sys
from tqdm import tqdm

import torch

import matplotlib.pyplot as plt
from collections import defaultdict

from prepare_selector_data import get_layer_info, extract_features, test_model
from selector_train_v1 import Selector, load_dataset

# Add src directory to sys.path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from idc import IDC
from pruning_methods import prune_neurons, ramdon_prune
from attribution import get_relevance_scores
from utils import parse_args, normalize_tensor, load_CIFAR, get_model, get_model_cifar, get_class_data, get_trainable_modules_main, test_random_class, save_json, load_json, Logger
from visualization import plot_common_neurons_rate

def prapared_parameters(args):
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + 'SelectorPredTest-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
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

    selector_path = 'selector_lenet_h3_2.pt'
    layer_info = get_layer_info(trainable_module_name[args.layer_index], trainable_module_name)
    attributions = ['lc', 'la', 'ii', 'ldl', 'lgs', 'lig', 'lfa', 'lrp']
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_out = len(attributions)
    selector_model = Selector(trainable_module[-1].in_features+layer_info.shape[0], num_out, [256, 128, 64])
    selector_model.load_state_dict(torch.load(selector_path))

    return model, selector_model, module_name, module, trainable_module, trainable_module_name, attributions, classes, log

def calculate_common_neurons_rate(prediction_info):
    common_neurons_rate = defaultdict(list)
    common_neurons_indices = defaultdict(list)
    
    for truth_class, truth_neurons, pred_neurons in zip(prediction_info['truth_class'], prediction_info['truth_neuron_index'], prediction_info['pred_neuron_index']):
        common_neurons = set(truth_neurons).intersection(set(pred_neurons))
        common_rate = len(common_neurons) / len(truth_neurons)
        common_neurons_rate[truth_class].append(common_rate)
        common_neurons_indices[truth_class].append(list(common_neurons))
    
    return common_neurons_rate, common_neurons_indices

def extract_and_plot_common_neurons_rate(log_file, target_class):

    import re
    import ast
    
    with open(log_file, "r") as file:
        log_data = file.read()

    # Regular expression to extract the dictionary of common neurons rate
    pattern = r"Common neurons rate: (defaultdict\(.*?\))"
    match = re.search(pattern, log_data, re.DOTALL)
    if match:
        try:
            common_neurons_rate = ast.literal_eval(match.group(1)[28:-1])  # Convert to dictionary safely
            
            if target_class in common_neurons_rate:
                rates = common_neurons_rate[target_class]
                average_rate = sum(rates) / len(rates)
                # Plot the data
                plt.figure(figsize=(10, 5))
                plt.scatter(range(len(rates)), rates, color='blue', marker='o')
                plt.xlabel("#Samples")
                plt.ylabel("Common Neurons Rate")
                plt.title(f"Common Neurons Rate Change for Class '{target_class}', Average Rate: {average_rate:.4f}")
                plt.grid(True)
                plt.savefig("./logs/common_neurons_rate_{}.png".format(target_class))
                
            else:
                print(f"Class '{target_class}' not found in the log.")
        
        except (SyntaxError, ValueError) as e:
            print("Error parsing the extracted data:", e)
    else:
        print("No common neurons rate data found in the log file.")

def predition(args, idc, selector_model, model, original_state, trainable_module, trainable_module_name, dataloader, attributions, classes, log):
    
    selector_model.eval()    
    selector_model.cpu()
    
    model.eval()
    model.cpu()
    
    prediction_info = {
        'truth_class': [],
        'pred_class': [],
        'truth_attr': [],
        'pred_attr': [],
        'truth_neuron_index': [],
        'pred_neuron_index': []
    }
    
    correct_predictions = []
    accuracy_drops = {}
    loss_gains = {}

    with torch.no_grad():
        for images, labels, layer_infos, optimal_methods in tqdm(dataloader):
            outputs_class = model(images)
            features = extract_features(model, images)
            features_norm = normalize_tensor(features[0])
            combined_features = torch.cat((features_norm, layer_infos), dim=1)
            outputs_attr = selector_model(combined_features)
            
            _, preds_class = torch.max(outputs_class, 1)
            _, preds_attr = torch.max(outputs_attr, 1)
            
            attr_label_indices = torch.argmax(optimal_methods, dim=1)
            attr_pred_indices = torch.argmax(outputs_attr, dim=1)
            
            for index in tqdm(range(images.size(0))):
                prediction_info['truth_class'].extend([classes[labels[index]]])
                prediction_info['pred_class'].extend([classes[preds_class[index]]])
                prediction_info['truth_attr'].extend([attributions[attr_label_indices[index]]])
                prediction_info['pred_attr'].extend([attributions[attr_pred_indices[index]]])

                # Get the pruned neurons
                _, importance_scores_pred = get_relevance_scores(model, 
                                                                images[index].unsqueeze(0), 
                                                                labels[index], 
                                                                classes, 
                                                                trainable_module[args.layer_index],
                                                                layer_name=trainable_module_name[args.layer_index], 
                                                                attribution_method=attributions[attr_pred_indices[index]])
                _, importance_scores_truth = get_relevance_scores(model, 
                                                                images[index].unsqueeze(0), 
                                                                labels[index], 
                                                                classes, 
                                                                trainable_module[args.layer_index],
                                                                layer_name=trainable_module_name[args.layer_index], 
                                                                attribution_method=attributions[attr_label_indices[index]])
                
                indices_pred = idc.select_top_neurons(importance_scores_pred)
                indices_truth = idc.select_top_neurons(importance_scores_truth)
                
                prediction_info['pred_neuron_index'].extend([indices_pred.tolist()])
                prediction_info['truth_neuron_index'].extend([indices_truth.tolist()])
                
                if labels[index] == preds_class[index]:
                    # a pair-wise: 0 for truth, 1 for pred
                    correct_predictions.append((attributions[attr_label_indices[index]], attributions[attr_pred_indices[index]]))
                    original_accuracy, original_loss, f1_score = test_model(model, images[index].unsqueeze(0), labels[index].unsqueeze(0))
                    # Prune the neurons - ground truth
                    prune_neurons(model, trainable_module_name[args.layer_index], indices_truth)
                    pruned_accuracy_truth, pruned_loss_truth, f1_score_truth = test_model(model, images[index].unsqueeze(0), labels[index].unsqueeze(0))
                    model.load_state_dict(original_state)
                    # Prune the neurons - prediction
                    prune_neurons(model, trainable_module_name[args.layer_index], indices_pred)
                    pruned_accuracy_pred, pruned_loss_pred, f1_score_pred = test_model(model, images[index].unsqueeze(0), labels[index].unsqueeze(0))
                    model.load_state_dict(original_state)
                    
                    accuracy_drops['truth_neuron'] = original_accuracy - pruned_accuracy_truth
                    accuracy_drops['pred_neuron'] = original_accuracy - pruned_accuracy_pred                    
                    loss_gains['truth_neuron'] = pruned_loss_truth - original_loss
                    loss_gains['pred_neuron'] = pruned_loss_pred - original_loss
                                                        
    return prediction_info, accuracy_drops, loss_gains, correct_predictions
        
# python selector_pred_v1.py --dataset cifar10 --batch-size 256 --layer-index 2 --model lenet --top-m-neurons 10 --all-attr --logging                  
if __name__ == '__main__':
    args = parse_args()
    model, selector_model, module_name, module, trainable_module, trainable_module_name, attributions, classes, log = prapared_parameters(args)
    original_state = copy.deepcopy(model.state_dict())
    load_file = './saved_files/prepared_data_test_cifar.csv'
    testloader = load_dataset(load_file, attributions, args.batch_size)    
    trainloader_cifar, testloader_cifar, test_dataset_cifar, classes = load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)

    ### Step - 0: IDC Setup
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class)
    saved_pred_file = './logs/predit_info_h3.json'
    
    for target_class in classes:
        extract_and_plot_common_neurons_rate(log_file='./logs/SelectorPredTest-lenet-cifar10-L2-20250218-221627.log', target_class=target_class)

    ### Step - 1: Prediction
    if os.path.exists(saved_pred_file):
        combined_data = load_json(saved_pred_file)
        print(f"File {saved_pred_file} loaded.")
        prediction_info = combined_data['prediction_info']
        accuracy_drops = combined_data['accuracy_drops']
        loss_gains = combined_data['loss_gains']
        correct_predictions = combined_data['correct_predictions']
    else:
        prediction_info, accuracy_drops, loss_gains, correct_predictions = predition(args,
                                idc,
                                selector_model, 
                                model, 
                                original_state,
                                trainable_module, 
                                trainable_module_name, 
                                testloader, 
                                attributions, 
                                classes, 
                                log)
        combined_data = {
            'prediction_info': prediction_info,
            'accuracy_drops': accuracy_drops,
            'loss_gains': loss_gains,
            'correct_predictions': correct_predictions
        }
        save_json(saved_pred_file, combined_data)

    ### Step - 2: Common neurals
    common_neurons_rate, common_neurons_indices = calculate_common_neurons_rate(prediction_info)
    plot_common_neurons_rate(common_neurons_rate)
    if log is not None:
        log.logger.info(f"Common neurons rate: {common_neurons_rate}")
        log.logger.info(f"Common neurons indices: {common_neurons_indices}")
            
    else:
        print(f"Common neurons rate: {common_neurons_rate}")
        print(f"Common neurons indices: {common_neurons_indices}")
        
    ### Step - 3: Compare with ground truth (prunning acc diff)
    
    ### Step - 4: IDC Coverage rate
    ## Step - 4.1: Get the specific class 
    if args.all_class:
        ## Step - 4.2: Randomly select and test the class [Option 1]
        testloader_cifar, test_images, test_labels = test_random_class(test_dataset_cifar, test_all=args.idc_test_all, num_samples=args.num_samples)
    test_images, test_labels = get_class_data(testloader_cifar, classes, args.test_image)
