import copy
import csv
import random
import time
import os

import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from src.attribution import get_relevance_scores_for_all_layers, get_relevance_scores_dataloader
from src.utils import get_data, parse_args, get_model, eval_model_dataloder, get_trainable_modules_main, Logger


# Example command to run the script:
# python3 run_rq_1_demo.py --model vgg16 --saved-model '/torch-deepimportance/models_info/saved_models/vgg16_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/'  --device 'cuda:0' --batch-size 128 --idc-test-all --num-samples 0

attribution_methods = {'lrp': 'LRP', 'ldl': 'DeepLIFT', 'ldls': 'SHAP'}
N_list = [6, 8, 10, 15, 20]

def prapared_parameters(args):
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + '-{}-{}-{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        # saved_log_name = args.log_path + 'PrepareData-{}-{}-L{}-{}.log'.format(args.model, args.dataset, args.layer_index, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, TopK: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None

    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)

    return model, module_name, module, trainable_module, trainable_module_name, log

def wisdom_neurons(csv_file, top_k=10):
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_k)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)

    return top_k_neurons

def save_results_to_csv(relevance_records, accuracy_records, filename="rq1"):
    
    # Save relevance scores to CSV
    relevance_df = pd.DataFrame(relevance_records)
    relevance_df.to_csv(filename+"_relevance.csv", index=False)

    # Save accuracy drop results to CSV
    accuracy_df = pd.DataFrame(accuracy_records)
    accuracy_df.to_csv(filename+"_acc_drop.csv", index=False)
    

def run_single_attr_test_demo(model, test_loader, original_acc, final_layer, N_list):
    results = []
    
    # Load all test data to CPU memory for attribution
    test_images, test_labels = next(iter(DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset))))
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    for attr_key, attr_name in attribution_methods.items():
        print(f"Computing relevance scores with: {attr_name}")
        # Get relevance scores across all layers (excluding final layer)
        relevance_scores = get_relevance_scores_for_all_layers(model.cpu(), 
                                                               test_images.cpu(),
                                                               test_labels.cpu(), 
                                                               attribution_method=attr_key)

        # Flatten all neurons (excluding final layer)
        flat_scores = []
        for layer_name, scores in relevance_scores.items():
            if layer_name == final_layer:
                continue
            
            # Fully connected layer
            if scores.dim() == 1:
                for i, score in enumerate(scores):
                    flat_scores.append((score.item(), layer_name, i))
            
            # Convolutional layer
            else:
                mean_scores = torch.mean(scores, dim=[1, 2])
                for i, score in enumerate(mean_scores):
                    flat_scores.append((score.item(), layer_name, i))

        # Sort by score descending
        flat_scores.sort(key=lambda x: x[0], reverse=False)
        
        for N in N_list:
            # Attribution-guided pruning
            top_N = flat_scores[:N]
            pruned_model = copy.deepcopy(model)

            for _, lname, idx in top_N:
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

            acc, avg_loss_pruned, f1_pruned = eval_model_dataloder(pruned_model, test_loader, device)
            print(f"Pruned model accuracy ({attr_name}): {acc:.2f}, Loss: {avg_loss_pruned:.2f}, F1 Score: {f1_pruned:.2f}")
            
            # acc = evaluate_model(pruned_model)
            drop = (original_acc - acc) * 100.0
            results.append([attr_name, N, f"{drop:.2f}", "Attribution"])
            print(f"Attribution: {attr_name}, N: {N}, Drop: {drop:.2f}%")
            
            # Random pruning baseline
            all_candidates = [x for x in flat_scores if x[1] != final_layer]
            random_sample = random.sample(all_candidates, N)
            rand_model = copy.deepcopy(model)

            for _, lname, idx in random_sample:
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
                            
            acc, avg_loss_pruned, f1_pruned = eval_model_dataloder(rand_model, test_loader, device)
            print(f"Pruned model accuracy (Random): {acc:.2f}, Loss: {avg_loss_pruned:.2f}, F1 Score: {f1_pruned:.2f}")

            # acc = evaluate_model(rand_model)
            drop = (original_acc - acc) * 100.0
            results.append([attr_name, N, f"{drop:.2f}", "Random"])
            print(f"Random, N: {N}, Drop: {drop:.2f}%")
    
    return results

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
        all_candidates = [x for x in global_neurons if x[1] != final_layer]
        random_sample = random.sample(all_candidates, n_prune)
        rand_model = copy.deepcopy(model)
        for _, lname, idx in random_sample:
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
        print(f"Random N: {n_prune}, Drop: {acc_drop*100:.2f}%")
        
        accuracy_records.append({
                "Attribution Method": "Random",
                "Top-N": n_prune,
                "Accuracy Drop": acc_drop
        })

    return accuracy_records

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
        for _, lname, idx in top_N:
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
        print(f"Attribution: {attr_method}, N: {n_prune}, Drop: {acc_drop*100:.2f}%")
        
        accuracy_records.append({
                "Attribution Method": attr_method,
                "Top-N": n_prune,
                "Accuracy Drop": acc_drop
        })
    
    return accuracy_records

def run_single_train_attr(train_loader, model, device, final_layer):
    # Prepare data structures to collect results
    relevance_records = []
    accuracy_records = []

    # Main loop: for each attribution method, compute relevance and evaluate pruning
    for attr_key, attr_name in attribution_methods.items():
        print(f"Computing relevance scores with: {attr_name}")
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
        
        # Flatten into a global list of (layer_name, neuron_idx, score) tuples
        flat_scores = []
        for layer_name, scores in relevance.items():
            if layer_name == final_layer:
                continue
            # Fully connected layer
            if scores.dim() == 1:
                for i, score in enumerate(scores):
                    flat_scores.append((score.item(), layer_name, i))
            # Convolutional layer
            else:
                mean_scores = torch.mean(scores, dim=[1, 2])
                for i, score in enumerate(mean_scores):
                    flat_scores.append((score.item(), layer_name, i))
        
        # Sort descending by relevance score
        flat_scores.sort(key=lambda x: x[2], reverse=False)
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
    model, module_name, module, trainable_module, trainable_module_name, log = prapared_parameters(args)

    ### Data settings
    train_loader, test_loader, train_dataset, test_dataset, classes = get_data(args.dataset, args.batch_size, args.data_path, args.large_image)
    
    # Get the original accuracy
    original_acc, avg_loss, f1 = eval_model_dataloder(model, test_loader, device)
    print(f"Original accuracy: {original_acc:.2f}, Loss: {avg_loss:.2f}, F1 Score: {f1:.2f}")

    # Skip final classifier layer
    final_layer = trainable_module_name[-1]
    
    # RQ 1 run case
    relevance_records, accuracy_records = run_single_train_attr(train_loader, model, device, final_layer)
    
    save_results_to_csv(relevance_records, accuracy_records, filename="rq1_"+ args.dataset + "_"+ args.model)
    print(" ==== Done with RQ1 ====")