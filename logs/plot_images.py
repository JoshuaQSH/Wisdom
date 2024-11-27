import re
import matplotlib.pyplot as plt
import argparse
import glob

# Function to parse the log file
def parse_log_file(log_file):
    with open(log_file, 'r') as file:
        print("Opening log file:", log_file)
        lines = file.readlines()
    
    # Extract model, dataset, and layers index from the filename
    filename = log_file.split('/')[-1]
    match = re.match(r'AttriTest-(\w+)-(\w+)-(\d+)', filename)
    if match:
        model, dataset, saved_date = match.groups()
    else:
        raise ValueError("Filename does not match expected pattern")
    
    # Extract the class, accuracy, and tensor values from the log file
    data = []
    layers_index = None
    attribution = None
    accuracy = None
    class_name = None
    for line in lines:
        if 'Class:' in line:
            class_match = re.search(r'Class: (\w+)', line)
            if class_match:
                class_name = class_match.group(1)
        
        if 'Layers_Index:' in line:
            layers_match = re.search(r'Layers_Index: (\d+)', line)
            if layers_match:
                layers_index = layers_match.group(1)

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

    return model, dataset, layers_index, data

def plot_data(model, dataset, layers_index, data):
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

# Main function
def plot_main(plot_all=False, log_file=None):
    if plot_all:
        log_files = glob.glob('AttriTest-*.log')
        for log_file in log_files:
            model, dataset, layers_index, data = parse_log_file(log_file)
            plot_data(model, dataset, layers_index, data)
    else:
        print("Plotting only the first log file")
        model, dataset, layers_index, data = parse_log_file(log_file)
        plot_data(model, dataset, layers_index, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, default='AttriTest-lenet-cifar10-20241123-222131.log', help='Path to the log file')
    parser.add_argument('--plot-all', action='store_true', help='Plot all log files')
    args = parser.parse_args()
    plot_main(plot_all=args.plot_all, log_file=args.log_file)