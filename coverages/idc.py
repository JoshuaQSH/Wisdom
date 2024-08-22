import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import TorchLRP.lrp
from TorchLRP.lrp import trace
from TorchLRP.lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns


# from utils import save_quantization, load_quantization, save_totalR, load_totalR
# from utils import save_layerwise_relevances, load_layerwise_relevances
# from utils import get_layer_outs_new
# from lrp_toolbox.model_io import write, read


experiment_folder = 'experiments'
model_folder = 'models_dir'


def get_trainable_layers(model, parent_name=''):
    trainable_layers = []
    non_trainable_layers = []
    for name, layer in model.named_children():
        current_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            trainable_layers.append(current_name)
        else:
            non_trainable_layers.append(current_name)
            if isinstance(layer, nn.Sequential):
                trainable, non_trainable = get_trainable_layers(layer, current_name)
                trainable_layers.extend(trainable)
                non_trainable_layers.extend(non_trainable)
    
    return trainable_layers, non_trainable_layers

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def limit_precision(values, prec=2):
    limited_values = []
    for v in values:
        limited_values.append(round(v,prec))
    
    return limited_values

def heatmap(X, cmap_name="seismic"):
    cmap = plt.cm.get_cmap(cmap_name)

    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()

    shape = X.shape
    tmp = X.sum(axis=-1) # Reduce channel axis

    tmp = project(tmp, output_range=(0, 255)).astype(int)
    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[-1] = 3
    return tmp.reshape(shape).astype(np.float32)

def heatmap_grid(a, nrow=3, fill_value=1., cmap_name="seismic", heatmap_fn=heatmap):
    # Compute colors
    a = heatmap_fn(a, cmap_name=cmap_name) 
    return grid(a, nrow, fill_value)

def grid(a, nrow=3, fill_value=1.):
    bs, h, w, c = a.shape

    # Reshape to grid
    rows = bs // nrow + int(bs % nrow != 0)
    missing = (nrow - bs % nrow) % nrow
    if missing > 0: # Fill empty spaces in the plot
        a = np.concatenate([a, np.ones((missing, h, w, c))*fill_value], axis=0)

    # Border around images
    a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.5)
    a = a.reshape(rows, nrow, h+2, w+2, c)
    a = np.transpose(a, (0, 2, 1, 3, 4))
    a = a.reshape( rows * (h+2), nrow * (w+2), c)
    return a

# Function to get layer outputs
def get_layer_outputs_new(model, inputs, skip=[]):
    outputs = []
    hooks = []

    def hook_fn(module, input, output):
        outputs.append(output)

    for index, layer in enumerate(model.features):  # Adjust if your model structure differs
        if index not in skip:
            hooks.append(layer.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    for hook in hooks:
        hook.remove()

    return outputs

class ImportanceDrivenCoverage:
    def __init__(self, model, batch_size, train_loader, test_loader, subject_layer=-1, num_rel=10, rule="epsilon", is_plot=False, device='cpu'):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.num_samples_plot = min(batch_size, 9)
        # gradient, epsilon, gamma+epsilon, alpha1beta0, alpha2beta1, patternnet, patternattribution
        self.rule = rule
        self.ax = None
        self.subject_layer = subject_layer
        self.num_rel = num_rel
        self.covered_combinations = ()
        self.num_samples = 0
    
    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    # Sample from train_loader
    def sample_one(self, dataloader, num_samples=1):
        for x, y in dataloader:
            break
        if num_samples != 0:
            x = x[:num_samples].to(self.device)
            y = y[:num_samples].to(self.device)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        x.requires_grad_(True)
        return x, y
    
    def sample_batch(self, dataloader):
        # Sample batch from train_loader
        for x, y in dataloader: 
            break
        x = x[:self.num_samples_plot].to(self.device)
        y = y[:self.num_samples_plot].to(self.device)
        x.requires_grad_(True)
        
        with torch.no_grad(): 
            y_hat = self.model(x)
            pred = y_hat.max(1)[1]
        
        return x, y, y_hat, pred
    
    def compute_explanation(self, rule, pattern=None, is_trace=True):
        x, y, y_hat, pred = self.sample_batch(self.train_loader)
        
        # Reset gradient
        x.grad = None

        # Forward pass with rule argument to "prepare" the explanation
        y_hat = self.model.forward(x, explain=True, rule=rule, pattern=pattern)
        # Choose argmax
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        # y_hat *= 0.5 * y_hat # to use value of y_hat as starting point
        y_hat = y_hat.sum()

        # Enable and clean trace
        if is_trace:
            trace.enable_and_clean()

        # Backward pass (compute explanation)
        y_hat.backward()
        explanation = x.grad

        all_relevances = trace.collect_and_disable()
        for i,t in enumerate(all_relevances):
            print(i,t.shape)

        return y_hat, explanation
                
    def plot_attribution(self, a, ax_, preds, title, cmap='seismic', img_shape=28, postprocess=None):
        ax_.imshow(a) 
        ax_.axis('off')

        cols = a.shape[1] // (img_shape+2)
        rows = a.shape[0] // (img_shape+2)
        for i in range(rows):
            for j in range(cols):
                ax_.text(28+j*30, 28+i*30, preds[i*cols+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
        ax_.set_title(title)
    
    def store_patterns(self, file_name, patterns):
        with open(file_name, 'wb') as f:
            pickle.dump([p.detach().cpu().numpy() for p in patterns], f)

    def load_patterns(self, file_name): 
        with open(file_name, 'rb') as f: 
            p = pickle.load(f)
            return p
    
    def get_layer_outs(self, model, inputs, skip=[]):
        outputs = []
        hooks = []

        def hook_fn(module, input, output):
            outputs.append(output)
        
        # Register hooks to the layers except those specified in skip
        for index, layer in enumerate(model.children()):
            if index not in skip:
                hooks.append(layer.register_forward_hook(hook_fn))

        # Forward pass to capture the outputs
        model.eval()
        with torch.no_grad():
            _ = model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs
    
    def find_relevant_neurons(self, x, y, lrpmethod=None, sample_method='all'):
        
        x.grad = None
        y_hat = self.model.forward(x, explain=True, rule=lrpmethod, pattern=None)
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        y_hat = y_hat.sum()
        trace.enable_and_clean()
        # Backward pass (compute explanation)
        y_hat.backward()
        explanation = x.grad

        all_relevances = trace.collect_and_disable()
        
        if self.subject_layer == -1 and sample_method == 'all':
            # Calculate the mean of all the inputs' relevances
            model_rel = {}
            for i, t in enumerate(all_relevances):
                layer_t = all_relevances[i].mean(dim=0)
                print(i, layer_t.shape)
                layer_relevance_np = layer_t.cpu().numpy()
                model_rel[i] = np.argsort(layer_relevance_np)
        
        elif self.subject_layer == -1 and sample_method == 'one':
            model_rel = {}
            for i, t in enumerate(all_relevances):
                print(i, t.shape)
                layer_relevance_np = t.cpu().numpy()
                model_rel[i] = np.argsort(layer_relevance_np)
                
                # most_rel[i] = np.argsort(layer_relevance_np)[::-1][:self.num_rel]
                # least_rel[i] = np.argsort(layer_relevance_np)[0][:self.num_rel]
        else:
            model_rel = np.argsort(all_relevances[self.subject_layer].cpu().numpy())[::-1]
        
        return model_rel, all_relevances

    def quantizeSilhouette(self, out_vectors, model_rel):
        quantized_ = []
        for key, value in model_rel.items():
            out_i = []
            if isinstance(self.model[int(out_vectors[key])], nn.Linear):
                out_i.append(list(value.squeeze()))
                flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
                out_i = flatten_list(out_i)
            else:
                out_i.append(np.mean(value))
                
            values = []
            if not len(out_i) < 10:
                clusterSize = range(2, 5)
                clustersDict = {}
                for clusterNum in clusterSize:
                    kmeans = cluster.KMeans(n_clusters=clusterNum)
                    clusterLabels = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                    silhouetteAvg = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                    clustersDict[silhouetteAvg] = kmeans
                maxSilhouetteScore = max(clustersDict.keys())
                bestKMean = clustersDict[maxSilhouetteScore]
                values = bestKMean.cluster_centers_.squeeze()
            values = list(values)
            values = limit_precision(values)
                
            if len(values) == 0:
                values.append(0)

            quantized_.append(values)
        return quantized_
    

    ## it measures how many distinct patterns of neuron activations (quantized into clusters) are triggered by the test data, 
    ## relative to the total number of possible patterns.
    def measure_idc(self, target_layer, target_trainable_layer, relevant_neurons, qtized):
        lout = []
        covered_combinations = []
        for x, y in self.test_loader:
            x.requires_grad_(True)
            max_comb = len(x)
            for test_idx in tqdm(range(len(x))):
                test_layer_outs = self.get_layer_outs(self.model, x[test_idx].unsqueeze(0))
                is_conv = isinstance(test_layer_outs[target_layer], torch.Tensor) and len(test_layer_outs[target_layer].shape) > 2
                
                if is_conv:
                    for r in relevant_neurons[target_trainable_layer][0][:self.num_rel]:
                        lout.append(torch.mean(test_layer_outs[target_layer][0, r]))
                else:
                    for r in relevant_neurons[target_trainable_layer][0][:self.num_rel]:
                        lout.append(test_layer_outs[target_layer][0][r])
            
                comb_to_add = determine_quantized_cover(lout, qtized, str(target_trainable_layer))
                # comb_to_add_ = [item for sublist in comb_to_add for item in sublist]
                # covered_combinations.add(tuple(comb_to_add))
                covered_combinations.append(comb_to_add)
                
            if self.num_samples:
                continue
            else:
                break
        
        covered_num = len(covered_combinations)
        coverage = float(covered_num)/max_comb
        
        return coverage*100, covered_combinations

    def test(self, all_patterns_path=None, pos_patterns_path=None, is_plot=False):
        if all_patterns_path is None:
            all_patterns_path = "./examples/patterns/pattern_all.pkl"
        if pos_patterns_path is None:
            pos_patterns_path = "./examples/patterns/pattern_pos.pkl"

        if not os.path.isfile(all_patterns_path):  # Either load or compute them
            patterns_all = fit_patternnet(self.model, self.train_loader, device=self.device)
            self.store_patterns(all_patterns_path, patterns_all)
        else:
            patterns_all = [torch.tensor(p, device=self.device, dtype=torch.float32) for p in self.load_patterns(all_patterns_path)]
        
        if not os.path.isfile(pos_patterns_path):
            patterns_pos = fit_patternnet_positive(self.model, self.train_loader, device=self.device)#, max_iter=1)
            self.store_patterns(pos_patterns_path, patterns_pos)
        else:
            patterns_pos = [torch.from_numpy(p).to(self.device) for p in self.load_patterns(pos_patterns_path)]
        
        inputs, outputs = self.sample_one(self.train_loader, num_samples=0)
        
        ### 1.Find Relevant Neurons ###
        ## the tf version will get relevance_neurons: array([83, 30, 22, 23, 24, 25, 26, 27, 28, 29])
        model_rel, all_relevances = self.find_relevant_neurons(inputs, outputs, self.rule)
        
        ### 2.Quantize Relevant Neuron Outputs ###
        trainable_layers, non_trainable_layers = get_trainable_layers(self.model)
        # Get layer outputs
        train_layer_outs = self.get_layer_outs(self.model, inputs)
        
        # Access by calling self.model[list(model_rel.keys())[i]]
        # qtized = self.quantizeSilhouette(train_layer_outs[self.subject_layer], all_relevances)
        # TODO: A hack here
        # target_layer = [1,5,7,9]
        # target_trainable_layer = [3,2,1,0]
        # 1->3, 5->2, 7->1, 9->0
        target_layer = [1,5,7,9]        
        target_trainable_layer = [3,2,1,0]
        
        # qtized = quantizeSilhouette_pytorch(train_layer_outs[7], 
        #     all_relevances, 
        #     model_rel, 
        #     subject_layer=1, 
        #     num_rel=self.num_rel)
        # trainable_layers = trainable_layers[::-1]
        # qtized = self.quantizeSilhouette(trainable_layers, model_rel)
        
        qtized = quantizeSilhouette_pytorch(train_layer_outs[target_layer[0]], 
                                            all_relevances, model_rel, 
                                            subject_layer=target_trainable_layer[0], 
                                            num_rel=self.num_rel)
        print(qtized)        

        # for i, j in zip(target_layer, target_trainable_layer):
        #     qtized = quantizeSilhouette_pytorch(train_layer_outs[i], all_relevances, model_rel, subject_layer=j, num_rel=self.num_rel)
        #     print(qtized)        
        
        ### 3.Measure coverage ### 
        print("Calculating IDC coverage")
        coverage, covered_combinations = self.measure_idc(
            target_layer[0],
            target_trainable_layer[0],
            model_rel,
            qtized)

        return coverage, covered_combinations


# Function to limit precision of values
def limit_precision(values, prec=2):
    return [round(v, prec) for v in values]

# TODO: Check the combination number
def determine_quantized_cover(lout, quantized, layer_key='1'):
    covered_comb = []
    quantized_centers = quantized[layer_key]
    eps = 0.1
    for idx, l in enumerate(lout):
        neuron_clusters = quantized_centers[0]
        # Find the closest cluster center, setup a threashold
        # closest_q = min(neuron_clusters, key=lambda x: abs(x - l))       
        closest_q = [item for item in neuron_clusters if abs(item - l) < eps]
        if len(closest_q) >= 2:
            closest_q = min(closest_q)
 
        # Append the closest cluster center to the covered combination
        covered_comb.append(closest_q)
    return covered_comb

# A little changes here, now choose the first num_rel neurons in subjected layer as relevant neurons
def quantizeSilhouette_pytorch(out_vector, all_relevances, relevant_neurons, subject_layer=1, num_rel=10):
    quantized_ = {}
    # Assume conv if more than 2 dimensions
    is_conv = isinstance(out_vector, torch.Tensor) and len(out_vector.shape) > 2

    most_rel_neurons = relevant_neurons[subject_layer].squeeze()[:num_rel]
    quantized_[str(subject_layer)] = []

    # Loop through the specific layer based on the most relevant neurons
    out_i = []
    for neuron_index in most_rel_neurons:
        if is_conv:
            conv_temp = torch.tensor(neuron_index).expand(1,1,-1,-1)
            out_temp = torch.gather(out_vector, 1, conv_temp)
            out_i.append([torch.mean(out_temp)])
        else:
            out_i.append(out_vector[:, neuron_index])
    cluster_sizes = range(2, 5)
    clusters_dict = {}
    for cluster_num in cluster_sizes:
        kmeans = cluster.KMeans(n_clusters=cluster_num)
        cluster_labels = kmeans.fit_predict(out_i)
        silhouette_avg = silhouette_score(out_i, cluster_labels)
        clusters_dict[silhouette_avg] = kmeans

    # Select the best clustering result based on silhouette score
    max_silhouette_score = max(clusters_dict.keys())
    best_kmeans = clusters_dict[max_silhouette_score]
    quantized_values = best_kmeans.cluster_centers_.squeeze()

    # Store quantized values (cluster centers)
    quantized_values = [round(val, 2) for val in quantized_values]  # Limit precision
    if len(quantized_values) == 0:
        quantized_values.append(0)
    quantized_[str(subject_layer)].append(quantized_values)

    return quantized_