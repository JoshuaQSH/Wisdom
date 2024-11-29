# src/visualization.py
import torch
import matplotlib.pyplot as plt

def visualize_idc_scores(idc_scores, filename='./logs/idc_scores.pdf'):
    methods = list(idc_scores.keys())
    scores = [idc_scores[method] for method in methods]
    
    plt.figure(figsize=(12, 6))
    plt.bar(methods, scores, color='skyblue')
    plt.xlabel('Attribution Method')
    plt.ylabel('IDC Score (%)')
    plt.title('Comparison of IDC Scores for Different Attribution Methods')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=1200)
    # plt.show()

def visualize_activation(activation_values, selected_activations, layer_name, threshold=0, mode='fc'):
    saved_file = f'./images/mean_activation_values_{layer_name}.pdf'
    if mode == 'fc':
        mean_activation = activation_values.mean(dim=0)
        mean_selected_activation = selected_activations.mean(dim=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(activation_values[1])), mean_activation.numpy(), marker='o')
        plt.plot(range(len(selected_activations[1])), mean_selected_activation.numpy(), marker='p')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.legend(['All Neurons', 'Selected Neurons'])
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
        
    elif mode == 'conv':
        mean_activation = torch.mean(selected_activations, dim=[2, 3]).mean(dim=0)
        plt.plot(range(len(mean_activation)), mean_activation.numpy(), marker='o')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation Value')
        plt.title('Mean Activation Values of Neurons')
        plt.grid(True)
        plt.savefig(saved_file, dpi=1500)
        print("Mean Activation Values plotted, saved to {}".format(saved_file))
    else:
        raise ValueError(f"Invalid mode: {mode}")