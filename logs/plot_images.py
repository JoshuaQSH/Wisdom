import re
import matplotlib.pyplot as plt
import argparse
import glob
import numpy as np
import ast

markers = ['.', 'p', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, default='AttriTest-lenet-cifar10-20241123-222131.log', help='Path to the log file')
    parser.add_argument('--plot-all', action='store_true', help='Plot all log files')
    parser.add_argument('--plot-type', type=str, default='accuracy', choices=['neuron', 'acc'], help='Ploting type for the log file')
    args = parser.parse_args()
    plot_main(plot_all=args.plot_all, log_file=args.log_file, plot_type=args.plot_type)
