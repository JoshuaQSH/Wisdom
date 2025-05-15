import time
import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, get_trainable_modules_main, eval_model_dataloder, extract_random_class, Logger
from src.attribution import get_relevance_scores, get_relevance_scores_for_all_layers, get_relevance_scores_for_all_classes
from src.idc import IDC

'''
This script is for IDC pipeline with the selected model and dataset given the important neurons that being chosen from either:
 - Selector
 - Voter
 
 Comparing with the baseline:
 - Run `run_baseline.py` to get the baseline results.
 
It includes the following steps:
1. Load the dataset (CIFAR10, MNIST, or ImageNet).
2. Load the model.
3. Get the trainable modules from the model.
4. Set the device (GPU or CPU) - most of cases, we use CPU.
5. IDC counting for the selected model and dataset.

python run_pre.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --device cpu --n-cluster 2 --top-m-neurons 6 --test-image plane --end2end --num-samples 0 --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar_t6.csv' --idc-test-all 

@ Author: Shenghao Qiu
@ Date: 2025-04-01
'''


def load_dataset(args):
    if args.dataset == 'cifar10':
        return load_CIFAR(batch_size=args.batch_size, root=args.data_path, large_image=args.large_image)
    elif args.dataset == 'mnist':
        return load_MNIST(batch_size=args.batch_size, root=args.data_path)
    elif args.dataset == 'imagenet':
        return load_ImageNet(batch_size=args.batch_size, 
                             root=os.path.join(args.data_path, 'ImageNet'), 
                             num_workers=2, 
                             use_val=False)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

def process_neurons(csv_file, top_k=10, visualize=False):
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='Score', ascending=False).head(top_k)
    top_k_neurons = {}
    for layer_name, group in df_sorted.groupby('LayerName'):
        top_k_neurons[layer_name] = torch.tensor(group['NeuronIndex'].values)
    
    if visualize:
        plt.figure(figsize=(10, 6))
        for layer_name, group in df_sorted.groupby('LayerName'):
            plt.scatter(group['NeuronIndex'], group['Score'], label=layer_name, alpha=0.7, s=70)
        
        plt.xlabel('Neuron Index')
        plt.ylabel('Score')
        plt.title(f'Top-{top_k} Neuron Scores Across All Layers')
        plt.legend()
        plt.savefig('top_k_neuron_scores.pdf', format='pdf', dpi=1200)
        # plt.show()
    
    return top_k_neurons
    
def idc_count(args, model, classes, test_images, test_labels, csv_file):
    # IDC pipeline
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans")
    top_k_neurons = process_neurons(csv_file, args.top_m_neurons, True)
    activation_values, selected_activations = idc.get_activation_values_for_model(test_images, classes[test_labels[0]], top_k_neurons)
    kmeans_comb = idc.cluster_activation_values_all(selected_activations)
    unique_cluster, coverage_rate = idc.compute_idc_test_whole(test_images, 
                        test_labels,
                        top_k_neurons,
                        kmeans_comb,
                        "Voter")
    if log:
        log.logger.info("#Testing Samples: {}, IDC Coverage: {}, Attribution: {}".format(len(test_images), coverage_rate, "Voter"))
    else:
        print("#Testing Samples: {}, IDC Coverage: {}, Attribution: {}".format(len(test_images), coverage_rate, "Voter"))
            
if __name__ == '__main__':
    args = parse_args()
    
    ### Logger settings
    if args.logging:
        start_time = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d-%H%M%S',time.localtime(start_time/1000))
        saved_log_name = args.log_path + '-{}-{}-{}.log'.format(args.model, args.dataset, timestamp)
        log = Logger(saved_log_name, level='debug')
        log.logger.debug("[=== Model: {}, Dataset: {}, Layers_Index: {}, Topk: {} ==]".format(args.model, args.dataset, args.layer_index, args.top_m_neurons))
    else:
        log = None
        
    ### Model settings
    model_path = os.getenv("HOME") + args.saved_model
    
    ### Dataset settings
    trainloader, testloader, train_dataset, test_dataset, classes = load_dataset(args)

    ### Device settings
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    
    ### Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    ### Model evaluation
    accuracy, avg_loss, f1 = eval_model_dataloder(model, testloader, device)
    if log:
        log.logger.info("Model test Acc: {}, Loss: {}, F1 Score: {}".format(accuracy, avg_loss, f1))
    else:
        print("Model test Acc: {}, Loss: {}, F1 Score: {}".format(accuracy, avg_loss, f1))

    ### IDC pipeline
    if args.num_samples != 0:
        # Sample the testset data for the IDC coverage
        subset_loader, test_images, test_labels = extract_random_class(test_dataset, test_all=args.idc_test_all, num_samples=args.num_samples)
    else:
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)  
    
    idc_count(args, model, classes, test_images, test_labels, args.csv_file)

