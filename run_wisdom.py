import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_CIFAR, load_MNIST, load_ImageNet, get_class_data, parse_args, get_model, get_trainable_modules_main, eval_model_dataloder, extract_random_class, extract_class_to_dataloder, _configure_logging
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

Example usage:
python run_wisdom.py --model lenet --saved-model '/torch-deepimportance/models_info/saved_models/lenet_CIFAR10_whole.pth' --dataset cifar10 --data-path '/data/shenghao/dataset/' --device cpu --n-cluster 2 --top-m-neurons 6 --test-image plane --end2end --num-samples 0 --csv-file '/home/shenghao/torch-deepimportance/saved_files/pre_csv/lenet_cifar10.csv' --idc-test-all 

@ Author: Shenghao Qiu
@ Date: 2025-04-01
'''


def load_dataset(args):
    if args.dataset == 'cifar10':
        return load_CIFAR(batch_size=args.batch_size, root=args.data_path)
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
    
def idc_count(args, logger, model, classes, train_loader, test_images, csv_file):
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_wisdom_clusters.pkl"
    
    extra = dict(
        n_clusters = args.n_clusters,    # same as IDC’s n_clusters, but OK to repeat
        random_state = 42,   # fixes RNG
        n_init = 10    # keep best of 10 centroid seeds
    )

    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", extra, cache_path)
    top_k_neurons = process_neurons(csv_file, args.top_m_neurons, True)
    activation_values, selected_activations = idc.get_activations_model_dataloader(train_loader, top_k_neurons)
    selected_activations = {k: v.half().cpu() for k, v in selected_activations.items()}
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole(test_images, 
                        top_k_neurons,
                        cluster_groups)
    
    logger.info("#Testing Samples: %d", len(test_images))
    logger.info("Attribution Method: %s", "WISDOM")
    logger.info("Total importance combinations: %d", total_combination)
    logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
    logger.info("Coverage Rate: %.6f%%", coverage_rate * 100)


def idc_count_dataloader(args, logger, model, classes, trainloader, testloader, csv_file):
    if args.use_silhouette:
        cluster_info = "silhouette"
    else:
        cluster_info = str(args.n_clusters)
    cache_path = "./cluster_pkl/" + args.model + "_" + args.dataset + "_top_" + str(args.top_m_neurons) + "_cluster_" + cluster_info + "_wisdom_clusters.pkl"
    
    extra = dict(
        n_clusters = args.n_clusters,    # same as IDC’s n_clusters, but OK to repeat
        random_state = 42,   # fixes RNG
        n_init = 10    # keep best of 10 centroid seeds
    )

    # IDC pipeline
    idc = IDC(model, classes, args.top_m_neurons, args.n_clusters, args.use_silhouette, args.all_class, "KMeans", extra, cache_path)
    top_k_neurons = process_neurons(csv_file, args.top_m_neurons, True)
    activation_values, selected_activations = idc.get_activations_model_dataloader(trainloader, top_k_neurons)
    cluster_groups = idc.cluster_activation_values_all(selected_activations)
    coverage_rate, total_combination, max_coverage = idc.compute_idc_test_whole_dataloader(testloader, top_k_neurons, cluster_groups)
    
    logger.info("Attribution Method: %s", "WISDOM")
    logger.info("Total importance combinations: %d", total_combination)
    logger.info("Max Coverage (the best we can achieve): %.6f%%", max_coverage * 100)
    logger.info("Coverage Rate: %.6f%%", coverage_rate * 100)

def run_wisdom(args):
    # Logger settings  
    logger = _configure_logging(args.logging, args, 'debug')
    # Model settings
    model_path = os.getenv("HOME") + args.saved_model
    # Dataset settings
    trainloader, testloader, train_dataset, test_dataset, classes = load_dataset(args)
    # Device settings
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    # Model loading
    model, module_name, module = get_model(model_path)
    trainable_module, trainable_module_name = get_trainable_modules_main(model)
    
    # Model evaluation
    accuracy, avg_loss, f1 = eval_model_dataloder(model, testloader, device)
    logger.info("Model test Acc: {}, Loss: {}, F1 Score: {}".format(accuracy, avg_loss, f1))

    # IDC pipeline
    if args.num_samples != 0:
        # Sample the testset data for the IDC coverage
        subset_loader, test_images, test_labels = extract_random_class(test_dataset, test_all=args.idc_test_all, num_samples=args.num_samples)
    else:
        test_images, test_labels = get_class_data(testloader, classes, args.test_image)
        testloader = extract_class_to_dataloder(test_dataset, classes, args.batch_size, args.test_image)
    
    idc_count(args, logger, model, classes, trainloader, test_images, args.csv_file)
    
    # A dataloader version of the IDC counting
    idc_count_dataloader(args, logger, model, classes, trainloader, testloader, args.csv_file)
    
if __name__ == '__main__':
    args = parse_args()
    run_wisdom(args)

