import re
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import ast

from collections import defaultdict

markers = ['.', 'p', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
color_Z = ['#ECDFFF', '#D4C4EE', '#BDAADE', '#A690CF', '#8E78BF', '#8C75BC', '#6A579C']
color_B = ['#d2dee5', '##bdcbd7' '#b7d3dd', '#99b4cc', '#82a4ca']

# Function to parse the log file
def parse_log_file(log_file):
    with open(log_file, 'r') as file:
        print("Opening log file:", log_file)
        lines = file.readlines()
    
    # Extract model, dataset, and layers index from the filename
    filename = log_file.split('/')[-1]
    # AttriTest-lenet-cifar10-L1-20241123-222136
    match = re.match(r'AttriTest-(\w+)-(\w+)-(\w+)-(\d+)', filename)
    if match:
        model, dataset, layer_index, saved_date = match.groups()
    else:
        raise ValueError("Filename does not match expected pattern")
    
    # Extract the class, accuracy, and tensor values from the log file
    data = []
    before_acc = []
    layers_index = None
    attribution = None
    accuracy = None
    class_name = None
    is_random = False

    for line in lines:
        if 'Class:' in line:
            class_match = re.search(r'Class: (\w+)', line)
            if class_match:
                class_name = class_match.group(1)
        
        if 'Before Accuracy:' in line:
            acc_match = re.search(r'Before Accuracy: ([\d.]+)%', line)
            if acc_match:
                before_acc.append(float(acc_match.group(1)))
        
        if 'Layers_Index:' in line:
            layers_match = re.search(r'Layers_Index: (\d+)', line)
            if layers_match:
                layers_index = layers_match.group(1)
        
        if 'Random Prune:' in line:
            is_random_ = re.search(r'Random Prune: (\w+)', line)
            if is_random_:
                is_random = ast.literal_eval(is_random_.group(1))

        if 'Attribution:' in line:
            attr_match = re.search(r'Attribution: (\w+), Accuracy: ([\d.]+)%', line)
            if attr_match:
                # attribution.append(attr_match.group(1))
                # accuracy.append(attr_match.group(2))
                attribution = attr_match.group(1)
                accuracy = attr_match.group(2)
                
        if 'The chosen index' in line:
            tensor_match = re.search(r'tensor\(\[([0-9,\s]+)\]\)', line)
            if tensor_match:
                tensor_values = list(map(int, tensor_match.group(1).split(',')))
                data.append((class_name, attribution, accuracy, tensor_values))

    return model, dataset, layers_index, is_random, before_acc, data

def plot_chosen_neuron_index(model, dataset, layers_index, data):
    class_data = {}
    for class_name, attribution, accuracy, tensor_values in data:
        if class_name not in class_data:
            class_data[class_name] = []
        class_data[class_name].append((attribution, accuracy, tensor_values))
    
    for class_name, values in class_data.items():
        plt.figure(figsize=(10, 6))
        for attribution, accuracy, tensor_values in values:
            plt.scatter([attribution] * len(tensor_values), tensor_values, label=f'{attribution} ({accuracy}%)')
        
        plt.title(f'Class: {class_name}, Model: {model}, Dataset: {dataset}, Layers_Index: {layers_index}')
        plt.xlabel('Attribution Methods')
        plt.ylabel('The chosen neuron index')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.165), ncol=4)
        # plt.legend()
        plt.grid(True)
        plt.savefig(f'{model}_{dataset}_{layers_index}_{class_name}.pdf', format='pdf', dpi=1200)
        plt.close()

def plot_acc(model, dataset, layers_index, before_acc, data):
    
    # Organize data by method and class
    method_data = {}
    for class_name, method, accuracy, _ in data:
        if method not in method_data:
            method_data[method] = {}
        if class_name not in method_data[method]:
            method_data[method][class_name] = []
        method_data[method][class_name].append(float(accuracy))
    mean_accuracies = {method: {cls: np.mean(acc) for cls, acc in cls_data.items()} for method, cls_data in method_data.items()}
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 22})
    for method, cls_data in method_data.items():
        classes = list(cls_data.keys())
        accuracies = [np.mean(cls_data[cls]) for cls in classes]
        plt.plot(classes, accuracies, label=f'{method}')
        
    plt.plot(classes, before_acc, label='Origin', linestyle='--', color='black')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model} (Layers_Index: {layers_index}) Accuracy for each Method')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model}_{dataset}_{layers_index}_accuracy_plot.pdf', format='pdf', dpi=1200)
    plt.close()


def get_no_random(data):
    # Organize data by method and class
    method_data = {}
    for class_name, method, accuracy, _ in data:
        if method not in method_data:
            method_data[method] = {}
        if class_name not in method_data[method]:
            method_data[method][class_name] = []
        method_data[method][class_name].append(float(accuracy))
    
    return method_data
    

def get_random(log_file):

    with open(log_file, 'r') as file:
        log_data = file.readlines()

    # Extract accuracies
    accuracy_pattern = re.compile(r'Accuracy: (\d+\.\d+)%')
    class_pattern = re.compile(r'Class: (\w+) Before Accuracy')

    accuracies = {}
    current_class = None

    for line in log_data:
        class_match = class_pattern.search(line)
        if class_match:
            current_class = class_match.group(1)
            if current_class not in accuracies:
                accuracies[current_class] = []
        
        accuracy_match = accuracy_pattern.search(line)
        if accuracy_match and current_class:
            accuracy = float(accuracy_match.group(1))
            accuracies[current_class].append(accuracy)

    # Calculate mean and standard deviation for each class
    mean_accuracies = {cls: np.mean(acc) for cls, acc in accuracies.items()}
    std_accuracies = {cls: np.std(acc) for cls, acc in accuracies.items()}
    
    # Plot the accuracies with error bands
    classes = list(mean_accuracies.keys())
    mean_values = [mean_accuracies[cls] for cls in classes]
    std_values = [std_accuracies[cls] for cls in classes]
    
    return mean_values, std_values, classes

    
def plot_all_acc(is_random, data):
    
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 22})
    
    if is_random:
        mean_values, std_values, classes = get_random()
        plt.plot(classes, mean_values, marker='^', label='Random Pruning', color='blue')
        plt.fill_between(classes, 
                        np.array(mean_values) - np.array(std_values), 
                        np.array(mean_values) + np.array(std_values), 
                        color='blue', alpha=0.2)
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('LeNet on CIFAR-10: Random Pruning')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'random_demo.pdf', format='pdf', dpi=1200)
    plt.close()

def plot_w_random_acc(log_files_type1, log_files_type2):
    
    log_dict = {}
    
    # Group log files by LX value
    for log_file in log_files_type1:
        match = re.search(r'AttriTest-lenet-cifar10-(L\d+)-', log_file)
        if match:
            lx_value = match.group(1)
            if lx_value not in log_dict:
                log_dict[lx_value] = {'P': None, 'RP': None}
            log_dict[lx_value]['P'] = log_file

    for log_file in log_files_type2:
        match = re.search(r'AttriTest-lenet-cifar10-random-(L\d+)-', log_file)
        if match:
            lx_value = match.group(1)
            if lx_value not in log_dict:
                log_dict[lx_value] = {'P': None, 'RP': None}
            log_dict[lx_value]['RP'] = log_file
    
    # Process log files with the same LX value
    for lx_value, files in log_dict.items():
        if files['P'] and files['RP']:
            log_p = files['P']
            log_rp = files['RP']
            mean_values, std_values, classes = get_random(log_rp)
            model, dataset, layers_index, is_random, before_acc, data = parse_log_file(log_p)
            method_data = get_no_random(data)
            
            plt.figure(figsize=(14, 6))
            plt.rcParams.update({'font.size': 20})
            i = 0
            for method, cls_data in method_data.items():
                classes = list(cls_data.keys())
                accuracies = [np.mean(cls_data[cls]) for cls in classes]
                plt.plot(classes, accuracies, label=f'{method}', linewidth=2, markersize=10, marker=markers[i])
                i = i + 1
                
            plt.plot(classes, before_acc, label='Origin', marker='+', markersize=10, linestyle='--', linewidth=2, color='black')
            plt.plot(classes, mean_values, marker='X', markersize=10, label='Random Pruning', linewidth=2, color='blue')
            plt.fill_between(classes, 
                        np.array(mean_values) - np.array(std_values), 
                        np.array(mean_values) + np.array(std_values), 
                        color='blue', alpha=0.2)
            plt.xlabel('Class')
            plt.ylabel('Accuracy (%)')
            plt.title('LeNet on CIFAR-10 with Layer Index: ' + lx_value)
            # Adding a customized legend
            plt.legend(
                loc='upper center',
                bbox_to_anchor=(0.38, 1.02),  
                ncol=5,  
                framealpha=0,
                fontsize=16
            )
            plt.grid(True)
            plt.savefig('lenet_accplot_{}.pdf'.format(lx_value), format='pdf', dpi=1200)
            plt.close()


# Main function
def plot_main(plot_all=False, log_file=None, plot_type='neuron'):
    if plot_all:
        if plot_type == 'neuron':
            log_files = glob.glob('AttriTest-*.log')
            for log_file in log_files:
                model, dataset, layers_index, is_random, before_acc, data = parse_log_file(log_file)
                plot_chosen_neuron_index(model, dataset, layers_index, data)
        
        elif plot_type == 'accuracy':
            
            log_files_type1 = glob.glob('AttriTest-lenet-cifar10-L*-*.log')
            log_files_type2 = glob.glob('AttriTest-lenet-cifar10-random-L*-*.log')
            
            if log_files_type2:
                print("Has random pruning log files")
                plot_w_random_acc(log_files_type1, log_files_type2)

            else:
                print("No random pruning log files")
                for log_file in log_files_type1:
                    model, dataset, layers_index, is_random, before_acc, data = parse_log_file(log_file)
                    plot_acc(model, dataset, layers_index, before_acc, is_random, data)
        else:
            raise ValueError("Invalid plot type")
    
    else:
        print("Plotting only the first log file")
        model, dataset, layers_index, is_random, before_acc, data = parse_log_file(log_file)
        if plot_type == 'neuron':
            plot_chosen_neuron_index(model, dataset, layers_index, data)
        elif plot_type == 'accuracy':
            plot_acc(model, dataset, layers_index, before_acc, is_random, data)
        else:
            raise ValueError("Invalid plot type")

def analyze_preparedata_log_file(log_file_path):
    """
    Parse the log file and return four data structures:

    1) label_method_rates:    Dict of { (label-method) : rate } or nested dict { label: {method: rate} }
    2) layer_method_rates:    Dict of { (layer-method) : rate } or nested dict { layer: {method: rate} }
    3) label_common_neurons:  Dict { label: list_of_neuron_lists }
    4) layer_common_neurons:  Dict { layer: list_of_neuron_lists }
    """

    # Counters for how many times each (label, method) occurs
    label_method_counts = defaultdict(int)
    # Keep track of how many times each label appears
    label_counts = defaultdict(int)

    # Same for layers
    layer_method_counts = defaultdict(int)
    layer_counts = defaultdict(int)

    # For storing common-neuron lists
    label_common_neurons = defaultdict(list)
    layer_common_neurons = defaultdict(list)

    # Regex to identify lines
    layer_regex = re.compile(
        r"Layer:\s*(\S+),\s*Common Neurons:\s*(.*)"
    )
    label_regex = re.compile(
        r"Label:\s*(\S+),\s*Optimal method:\s*([a-zA-Z0-9]+)"
    )

    current_layer = None
    current_common_neurons = None

    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()

            # 1) Check if it is a Layer line
            m_layer = layer_regex.search(line)
            if m_layer:
                # Example: "Layer: features.10, Common Neurons: [114]"
                current_layer = m_layer.group(1)  # e.g. 'features.10'
                # Parse the bracketed neuron list. We can safely eval or parse ourselves:
                # e.g. "[32]", "[212]", "[17, 22]", etc.
                neuron_str = m_layer.group(2).strip()
                # If it's something like '[]', we want an empty list
                try:
                    current_common_neurons = eval(neuron_str)
                except:
                    current_common_neurons = []
                continue

            # 2) Check if it is a Label line
            m_label = label_regex.search(line)
            if m_label:
                # Example: "Label: plane, Optimal method: lrp, Accuracy drop..."
                label = m_label.group(1)   # e.g. 'plane'
                method = m_label.group(2)  # e.g. 'lrp'

                # We only record these if we have a current_layer set (from the most recent "Layer: ..." line).
                if current_layer is not None:
                    # Update counters for label->method
                    label_method_counts[(label, method)] += 1
                    label_counts[label] += 1

                    # Update counters for layer->method
                    layer_method_counts[(current_layer, method)] += 1
                    layer_counts[current_layer] += 1

                    # Store common neurons
                    label_common_neurons[label].append(current_common_neurons)
                    layer_common_neurons[current_layer].append(current_common_neurons)

                continue

    # Compute rates from the raw counts
    label_method_rates = {}
    for (label, method), count in label_method_counts.items():
        total_for_label = label_counts[label]
        if total_for_label > 0:
            rate = count / float(total_for_label)
        else:
            rate = 0.0
        label_method_rates[(label, method)] = rate

    layer_method_rates = {}
    for (layer, method), count in layer_method_counts.items():
        total_for_layer = layer_counts[layer]
        if total_for_layer > 0:
            rate = count / float(total_for_layer)
        else:
            rate = 0.0
        layer_method_rates[(layer, method)] = rate

    return label_method_rates, layer_method_rates, label_common_neurons, layer_common_neurons


def visualize_preparedata_results(label_method_rates, layer_method_rates,
                      label_common_neurons, layer_common_neurons):
    """
    Provide simple visualizations:
      1) Bar chart of method rates by label
      2) Bar chart of method rates by layer
      3) Textual listing of common neurons by label
      4) Textual listing of common neurons by layer
    """

    # 1) Visualize method rates by label
    #    We'll turn label_method_rates (label,method)->rate into a nested structure for plotting:
    from collections import defaultdict

    # Collect methods per label
    label_to_methods_rates = defaultdict(dict)
    for (label, method), rate in label_method_rates.items():
        label_to_methods_rates[label][method] = rate

    # One simple approach is to pick each label and plot a bar for the methods
    fig, axes = plt.subplots(nrows=1, ncols=len(label_to_methods_rates), figsize=(5*len(label_to_methods_rates), 4))
    if len(label_to_methods_rates) == 1:
        axes = [axes]  # If there's only one label, make it iterable

    for ax, (label, method_dict) in zip(axes, label_to_methods_rates.items()):
        methods = list(method_dict.keys())
        rates = list(method_dict.values())
        ax.bar(methods, rates, color=color_Z[2])
        ax.set_ylim([0, 1])
        ax.set_title(f"Label: {label}")
        ax.set_xlabel("Method")
        ax.set_ylabel("Rate")

    plt.tight_layout()
    plt.savefig("label_method_rates.pdf", format='pdf', dpi=1200)

    # 2) Visualize method rates by layer
    #    Similarly, layer_method_rates -> (layer,method)->rate
    layer_to_methods_rates = defaultdict(dict)
    for (layer, method), rate in layer_method_rates.items():
        layer_to_methods_rates[layer][method] = rate

    fig, axes = plt.subplots(nrows=1, ncols=len(layer_to_methods_rates), figsize=(5*len(layer_to_methods_rates), 4))
    if len(layer_to_methods_rates) == 1:
        axes = [axes]

    for ax, (layer, method_dict) in zip(axes, layer_to_methods_rates.items()):
        methods = list(method_dict.keys())
        rates = list(method_dict.values())
        ax.bar(methods, rates, color=color_B[2])
        ax.set_ylim([0, 1])
        ax.set_title(f"Layer: {layer}")
        ax.set_xlabel("Method")
        ax.set_ylabel("Rate")

    plt.tight_layout()
    plt.savefig("layer_method_rates.pdf", format='pdf', dpi=1200)

    # 3) Textual listing of common neurons by label
    fig, axes = plt.subplots(nrows=1, ncols=len(label_common_neurons), figsize=(6*len(label_common_neurons), 4))
    if len(label_common_neurons) == 1:
        axes = [axes]

    for ax, (label, neuron_lists) in zip(axes, label_common_neurons.items()):
        lengths = [len(nlist) for nlist in neuron_lists]
        ax.bar(range(len(lengths)), lengths, color=color_Z[3])
        ax.set_title(f"Common Neuron Counts\nLabel: {label}")
        ax.set_xlabel("Occurrence")
        ax.set_ylabel("Number of Common Neurons")

    plt.tight_layout()
    plt.savefig("label_common_neurons.pdf", format='pdf', dpi=1200)

    # 4) Bar charts of the number of common neurons by layer
    fig, axes = plt.subplots(nrows=1, ncols=len(layer_common_neurons), figsize=(6*len(layer_common_neurons), 4))
    if len(layer_common_neurons) == 1:
        axes = [axes]

    for ax, (layer, neuron_lists) in zip(axes, layer_common_neurons.items()):
        lengths = [len(nlist) for nlist in neuron_lists]
        ax.bar(range(len(lengths)), lengths, color=color_Z[3])
        ax.set_title(f"Common Neuron Counts\nLayer: {layer}")
        ax.set_xlabel("Occurrence")
        ax.set_ylabel("Number of Common Neurons")

    plt.tight_layout()
    plt.savefig("layer_common_neurons.pdf", format='pdf', dpi=1200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, default='AttriTest-lenet-cifar10-20241123-222131.log', help='Path to the log file')
    parser.add_argument('--plot-all', action='store_true', help='Plot all log files')
    parser.add_argument('--plot-type', type=str, default='accuracy', choices=['neuron', 'acc'], help='Ploting type for the log file')
    args = parser.parse_args()
    # plot_main(plot_all=args.plot_all, log_file=args.log_file, plot_type=args.plot_type)
    
    log_file_path = "PrepareTestTrainLayerLog-vgg16-cifar10-5-20250226-103933.log"
    label_method_rates, layer_method_rates, label_common_neurons, layer_common_neurons = analyze_preparedata_log_file(log_file_path)
    visualize_preparedata_results(label_method_rates, layer_method_rates, label_common_neurons, layer_common_neurons)