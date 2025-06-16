import copy
import random
import time
import os

import pandas as pd

import torch
import torch.nn as nn

from src.attribution import get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main


# Example command to run the script:
# python run_rq_1_demo.py --model resnet18 --saved-model '/torch-deepimportance/models_info/saved_models/resnet18_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --batch-size 32 --device 'cuda:0' --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/resnet18_cifar_b32.csv'

attribution_methods = {'lrp': 'LRP', 'ldl': 'DeepLIFT', 'lgs': 'SHAP', 'Wisdom': 'wisdom'}
N_list = [6, 8, 10, 15, 20]

def prapare_data_models(args):

    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name

def save_results_to_csv(relevance_records, accuracy_records, filename="rq1"):
    
    # Save relevance scores to CSV
    relevance_df = pd.DataFrame(relevance_records)
    relevance_df.to_csv(filename+"_relevance.csv", index=False)

    # Save accuracy drop results to CSV
    accuracy_df = pd.DataFrame(accuracy_records)
    accuracy_df.to_csv(filename+"_acc_drop.csv", index=False)
    

### Wisdom method ###
def wisdom_neurons(csv_file, top_k=10):
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_k)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)

    return top_k_neurons

def convert_top_k_neurons(top_k_neurons):
    """Convert top_k_neurons dictionary to a list of (layer_name, index) pairs."""
    converted = []
    for layer_name, indices in top_k_neurons.items():
        for index in indices:
            converted.append((layer_name, index.item()))
    return converted

def run_wisdom_test(model, test_loader, device, csv_file, original_acc, accuracy_records):
    
    for n_prune in N_list:
        pruned_model = copy.deepcopy(model)
        
        # Load the CSV file and get the top neurons
        top_k_neurons = wisdom_neurons(csv_file, top_k=n_prune)
        top_N = convert_top_k_neurons(top_k_neurons)
        
        for lname, idx in top_N:
            layer = dict(pruned_model.named_modules())[lname]

            with torch.no_grad():
                if isinstance(layer, nn.Conv2d):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
                elif isinstance(layer, nn.Linear):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
        
        acc, avg_loss, f1 = eval_model_dataloder(pruned_model, test_loader, device)
        acc_drop = original_acc - acc
        
        accuracy_records.append({
                "Attribution Method": "Wisdom",
                "Top-N": n_prune,
                "Accuracy Drop": acc_drop
        })
        
        print(f"Pruned Model Wisdom - Top {n_prune} Neurons - Accuracy: {acc:.2f}, Drop: {acc_drop*100:.2f}, Loss: {avg_loss:.2f}, F1 Score: {f1:.2f}")
    
    return accuracy_records

### Random pruning baseline ###
def record_acc_drop_random(total_neurons, 
                           global_neurons, 
                           model, 
                           test_loader, 
                           device, 
                           original_acc, 
                           final_layer, 
                           accuracy_records):
    
    for n in N_list:
        n_prune = min(n, total_neurons)
        # (layer_name, score, neuron_idx) tuples
        all_candidates = [x for x in global_neurons if x[0] != final_layer]
        random_sample = random.sample(all_candidates, n_prune)
        rand_model = copy.deepcopy(model)
        for lname, _, idx in random_sample:
            layer = dict(rand_model.named_modules())[lname]
            with torch.no_grad():
                if isinstance(layer, nn.Conv2d):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
                elif isinstance(layer, nn.Linear):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
        acc_random, avg_loss_random, f1_random = eval_model_dataloder(rand_model, test_loader, device)
        acc_drop = original_acc - acc_random
        
        print(f"Pruned Model Random - Top {n_prune} Neurons - Accuracy: {acc_random:.2f}, Drop: {acc_drop*100:.2f}, Loss: {avg_loss_random:.2f}, F1 Score: {f1_random:.2f}")
        
        accuracy_records.append({
                "Attribution Method": "Random",
                "Top-N": n_prune,
                "Accuracy Drop": acc_drop
        })

    return accuracy_records

### Attribution-guided pruning ###
def record_acc_drop(total_neurons, 
                    global_neurons, 
                    model, 
                    test_loader, 
                    device, 
                    original_acc, 
                    attr_method, 
                    accuracy_records):
    
    for n in N_list:
        n_prune = min(n, total_neurons)
        
        # Attribution-guided pruning
        top_N = global_neurons[:n_prune]
        pruned_model = copy.deepcopy(model)
        for lname, _, idx in top_N:
            layer = dict(pruned_model.named_modules())[lname]
            with torch.no_grad():
                if isinstance(layer, nn.Conv2d):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
                elif isinstance(layer, nn.Linear):
                    layer.weight[idx].zero_()
                    if layer.bias is not None:
                        layer.bias[idx].zero_()
        
        pruned_acc, avg_loss_pruned, f1_pruned = eval_model_dataloder(pruned_model, test_loader, device)
        acc_drop = original_acc - pruned_acc
        print(f"Pruned Model {attr_method} - Top {n_prune} Neurons - Accuracy: {pruned_acc:.2f}, Drop: {acc_drop*100:.2f}, Loss: {avg_loss_pruned:.2f}, F1 Score: {f1_pruned:.2f}")

        
        accuracy_records.append({
                "Attribution Method": attr_method,
                "Top-N": n_prune,
                "Accuracy Drop": acc_drop
        })
    
    return accuracy_records


### Main entry point ###
def run_single_train_attr(train_loader, test_loader, original_acc, model, device, csv_file, final_layer):
    # Prepare data structures to collect results
    relevance_records = []
    accuracy_records = []

    # Main loop: for each attribution method, compute relevance and evaluate pruning
    for attr_key, attr_name in attribution_methods.items():
        
        print(f"Computing relevance scores with: {attr_name}")
        
        ## For the Wisdom method, extract the voting results from the CSV file        
        if attr_key == 'Wisdom':
            accuracy_records = run_wisdom_test(model, test_loader, device, csv_file, original_acc, accuracy_records)
        
        ## Run with other single attribution methods
        else:
            # Get the relevance scores with the train data
            relevance = get_relevance_scores_dataloader(model, train_loader, device, attr_key)
            
            # Save all relevance scores for the chosen attribution method
            for layer_name, scores in relevance.items():
                for neuron_idx, score in enumerate(scores):
                    relevance_records.append({
                        "Attribution Method": attr_name,
                        "Layer Name": layer_name,
                        "Neuron Index": neuron_idx,
                        "Relevance Score": float(score)
                    })
            
            # Flatten into a global list of (layer_name, score, neuron_idx) tuples
            flat_scores = []
            for layer_name, scores in relevance.items():
                if layer_name == final_layer:
                    continue
                # Fully connected layer
                if scores.dim() == 1:
                    for i, score in enumerate(scores):
                        flat_scores.append((layer_name, score.item(), i))
                # Convolutional layer
                else:
                    mean_scores = torch.mean(scores, dim=[1, 2])
                    for i, score in enumerate(mean_scores):
                        flat_scores.append((layer_name, score.item(), i))
            
            # Sort descending by relevance score
            flat_scores.sort(key=lambda x: x[1], reverse=True)
            total_neurons = len(flat_scores)
            # Random pruning baseline
            accuracy_records = record_acc_drop_random(total_neurons=total_neurons, 
                            global_neurons=flat_scores, 
                            model=model, 
                            test_loader=test_loader, 
                            device=device, 
                            original_acc=original_acc, 
                            final_layer=final_layer, 
                            accuracy_records=accuracy_records)
            
            # Record the accuracy drop for each method [with random pruning]
            accuracy_records = record_acc_drop(total_neurons=total_neurons, 
                            global_neurons=flat_scores, 
                            model=model, 
                            test_loader=test_loader, 
                            device=device, 
                            original_acc=original_acc, 
                            attr_method=attr_name,
                            accuracy_records=accuracy_records)

    return relevance_records, accuracy_records
        

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")

    ### Model settings
    model, module_name, module, trainable_module, trainable_module_name = prapare_data_models(args)

    ### Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path, args.large_image)
    
    # Get the original accuracy
    original_acc, avg_loss, f1 = eval_model_dataloder(model, test_loader, device)
    print(f"Original accuracy: {original_acc:.2f}, Loss: {avg_loss:.2f}, F1 Score: {f1:.2f}")
    
    # Skip final classifier layer
    final_layer = trainable_module_name[-1]
    
    # RQ 1 run case
    relevance_records, accuracy_records = run_single_train_attr(train_loader, test_loader, original_acc, model, device, args.csv_file, final_layer)
    
    save_results_to_csv(relevance_records, accuracy_records, filename="rq1_"+ args.dataset + "_"+ args.model)
    print(" ==== Done with RQ1 ====")