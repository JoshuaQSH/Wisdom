
import sys
import os
import copy
import random
import numpy as np
import time
from tqdm import tqdm
import itertools
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

from style_operator import Stylized
import image_transforms

# -----------------------------------------------------------
# Fuzzer Logger
# Refers to the NLC: https://github.com/Yuanyuan-Yuan/NeuraL-Coverage/blob/main/utility.py
# -----------------------------------------------------------
class Logger(object):
    def __init__(self, args, engine):
        import time
        self.name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log'
        self.args = args
        self.log_path = os.path.join(args.log_dir, self.name)
        self.f = open(self.log_path, 'a')
        self.f.write('Dataset: %s\n' % args.dataset)
        self.f.write('Model: %s\n' % args.model)
        self.f.write('Class: %d\n' % args.num_class)
        self.f.write('Data in each class: %d\n' % args.num_per_class)
        self.f.write('Criterion: %s\n' % args.criterion)

        for k in engine.hyper_params.keys():
            self.f.write('%s %s\n' % (k, engine.hyper_params[k]))
    
    def update(self, engine):        
        print('Epoch: %d' % engine.epoch)
        print('Delta coverage: %f' % (engine.criterion.current - engine.initial_coverage))
        print('Delta time: %fs' % engine.delta_time)
        print('Delta batch: %d' % engine.delta_batch)
        print('AE: %d' % engine.num_ae)
        self.f.write('Delta time: %fs, Epoch: %d, Current coverage: %f, Delta coverage:%f, AE: %d, Delta batch: %d\n' % \
            (engine.delta_time, engine.epoch, engine.criterion.current, \
             engine.criterion.current - engine.initial_coverage,
             engine.num_ae, engine.delta_batch))

    def exit(self):
        self.f.close()


# -----------------------------------------------------------
# Image normalization
# -----------------------------------------------------------
def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
       transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return transform(image)

class Parameters(object):
    def __init__(self, base_args):
        self.model = base_args.model
        self.dataset = base_args.dataset
        self.data_path = base_args.data_path
        self.criterion = base_args.criterion
        self.use_sc = self.criterion in ['LSC', 'DSC', 'MDSC']
        self.num_workers = 4
        self.device = base_args.device
        self.seed = base_args.seed
        self.guided = base_args.guided if hasattr(base_args, 'guided') else False
        self.saved_model = base_args.saved_model
        self.wisdom_csv = base_args.wisdom_csv if hasattr(base_args, 'wisdom_csv') else None
        
        self.batch_size = 50
        self.mutate_batch_size = 1
        self.nc = 3
        self.image_size = 128 if self.dataset == 'ImageNet' else 32
        self.input_shape = (1, self.image_size, self.image_size, 3)
        self.num_class = 100 if self.dataset == 'ImageNet' else 10
        self.num_per_class = 1000 // self.num_class

        self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        self.alpha = 0.2 # default 0.2
        self.beta = 0.4 # default 0.4
        self.TRY_NUM = 50
        self.save_every = 100 # save every 100 images
        self.output_dir = './fuzz_guide/output/'

        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                            [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))        
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        # shear = list(itertools.product([getattr(image_transforms, "image_shear")], list(range(-3, 3))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation #+ shear
        self.P = contrast + brightness + blur
        self.S = list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        self.save_batch = False

class INFO(dict):
    @staticmethod
    def _k(arr):            # hash by identity, not by value
        return id(arr)

    def __getitem__(self, arr):
        return super().get(self._k(arr), (arr, 0))

    def __setitem__(self, arr, tpl):
        super().__setitem__(self._k(arr), tpl)

    def __missing__(self, arr):
        return (arr, 0)
    
# -----------------------------------------------------------
# Main Fuzzer
# -----------------------------------------------------------
class Fuzzer:
    def __init__(self, params, criterion, guided: bool = True):
        self.params = params
        self.criterion = criterion
        self.guided = guided
        self.time_slot = 60 * 10
        self.time_idx = 0
        self.epoch = 0
        self.info = INFO()
        self.delta_time = 0
        self.delta_batch = 0
        self.num_ae = 0
        self.initial_coverage = copy.deepcopy(criterion.current)

        # default hyper‑parameters from the original repo
        self.hyper_params = dict(alpha=0.4, # [0, 1], default 0.02, 0.1 # number of pix
                                 beta=0.8, # [0, 1], default 0.2, 0.5 # max abs pix
                                 TRY_NUM=50,
                                 p_min=0.01, 
                                 gamma=5, 
                                 K=64)

        self.logger = Logger(params, self)

    # -------------- public --------------------------------------------------
    def run(self, I_input, L_input):
        """
        I_input : list/ndarray raw images in range [0,1]
        L_input : list/ndarray  ground truth labels
        """
        # F = np.array([]).reshape(0, *(self.params.input_shape[1:])).astype('float32')
        T = self._preprocess(I_input, L_input)
        
        B, B_label, B_id = self._select_next(T)   # current batch
        self.epoch = 0
        start = time.time()

        while not self._should_stop():
            if self.epoch % 500 == 0:
                self.logger.update(self)

            # we mutate every img
            S, S_label = B, B_label                   
            Ps = self._power_schedule(S, self.hyper_params["K"])

            B_new, B_old, B_label_new = [], [], []

            # ---------------- mutation phase -------------------------------
            for s_i, (I, L) in enumerate(zip(S, S_label)):
                n_trials = Ps(s_i) if self.guided else 1
                accepted = False

                for _ in range(n_trials):
                    I_new, _op = self._mutate(I)

                    # random mode keeps any *changed* mutation
                    if not self.guided:
                        if self._is_changed(I, I_new):
                            accepted = True
                            break
                        else:
                            continue

                    # guided ‑‑ evaluate coverage gain
                    torch_img = self._to_tensor(np.stack([I_new]), norm=True)
                    torch_lbl = torch.tensor([L], device=self.params.device)

                    if self.params.criterion in ['LSC','DSC','MDSC']:
                        cov_dict = self.criterion.calculate(torch_img, torch_lbl)
                    elif self.params.criterion in ['Deepimportance', 'Wisdom']:
                        cov_dict = self.criterion.calculate(torch_img)
                    else:
                        cov_dict = self.criterion.calculate(torch_img)

                    gain = self.criterion.gain(cov_dict)
                    if self._coverage_gain(gain):
                        self.criterion.update(cov_dict, gain)
                        accepted = True
                        break   # stop further trials for this seed

                if accepted:
                    B_new.append(I_new);   B_old.append(I);   B_label_new.append(L)

            # ------------- post‑processing & bookkeeping -------------------
            if B_new:                                   # at least one new seed
                self._append_new_seeds(T, B_new, B_label_new)
                self.delta_batch += 1

                # adversarial statistics
                tensor_img  = self._to_tensor(np.stack(B_new), norm=True)
                tensor_lbl  = torch.tensor(B_label_new, device=self.params.device)
                wrong, wrong_idx = self._is_adversarial(tensor_img, tensor_lbl)
                self.num_ae += int(wrong)

                # periodically save montage
                if self.epoch % self.params.save_every == 0:
                    self._save_image(np.stack(B_new) / self.params.input_scale,
                                     f"{self.params.image_dir}{self.epoch:03d}_new.jpg")
                    self._save_image(np.stack(B_old)/self.params.input_scale,
                                     f"{self.params.image_dir}{self.epoch:03d}_old.jpg")
                    if wrong:
                        save_image(tensor_img[wrong_idx],
                                   f"{self.params.image_dir}{self.epoch:03d}_ae.jpg",
                                   normalize=True)

            B, B_label, B_id = self._select_next(T)
            self.epoch += 1
            self.delta_time = time.time() - start

    def exit(self):
        self.logger.update(self)
        self.criterion.save(self.params.coverage_dir + "coverage_final.pt")
        self.logger.exit()

    # -------------- internal helpers ---------------------------------------
    def _should_stop(self):
        return (self.epoch > 10_000) or (self.delta_time > 6*60*60)

    def _preprocess(self, imgs, labels):
        # shuffle & scale to [0,255] uint8 space (as in the original code)
        order = np.random.permutation(len(imgs))
        imgs = [imgs[i] * self.params.input_scale for i in order]
        labels = [labels[i] for i in order]
        Bs = self._to_batch(imgs)
        Bs_label = self._to_batch(labels)
        # B_c, Bs, Bs_label
        return [0]*len(Bs), Bs, Bs_label        

    def _to_batch(self, seq):
        batches, cur = [], []
        for x in seq:
            if cur and len(cur) % self.params.mutate_batch_size == 0:
                batches.append(np.stack(cur)); cur = []
            cur.append(x)
        if cur: batches.append(np.stack(cur))
        return batches

    # ---------- per‑epoch primitives ---------------------------------------
    def _select_next(self, T):
        B_c, Bs, Bs_label = T
        priorities = [self._priority(c) for c in B_c]
        idx = np.random.choice(len(Bs), p=np.array(priorities)/np.sum(priorities))
        return Bs[idx], Bs_label[idx], idx

    def _priority(self, B_ci):
        pmin, gamma = self.hyper_params["p_min"], self.hyper_params["gamma"]    
        if B_ci < (1 - pmin) * gamma:
            return 1 - B_ci / gamma
        else:
            return pmin

    def _power_schedule(self, S, K):
        beta = self.hyper_params["beta"]
        potentials=[]
        for I in S:
            I0, _state = self.info[I]
            p = beta* 255 * np.sum(I > 0) - np.sum(np.abs(I - I0))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)
        return lambda idx: int(np.ceil(potentials[idx] * K))

    # ---------- mutation & acceptance --------------------------------------
    def _mutate(self, I):
        G, P, S = self.params.G, self.params.P, self.params.S
        I0, state = self.info[I]

        for _ in range(self.hyper_params["TRY_NUM"]):
            t, p = random.choice(G + P + S) if state == 0 else random.choice(P + S)
            
            # TODO: a quick patch for stylized transform
            I_float32 = I.astype("float32", copy=False)
            I_new = t(I_float32, p).reshape(self.params.input_shape[1:])
            I_new = np.clip(I_new, 0, 255)
            # I_new = np.clip(t(I, p).reshape(self.params.input_shape[1:]), 0, 255)

            if (t, p) in S or self._pixel_budget(I0, I_new):
                # update INFO cache
                if (t, p) in G:
                    state = 1
                    self.info[I_new] = (np.clip(t(I0, p), 0, 255), state)
                else:
                    self.info[I_new] = (I0, state)
                return I_new, (t, p)
        # return I, (None, None)      # fallback
        return I, (t, p) # fallback

    def _pixel_budget(self, I, I_new):
        alpha, beta = self.hyper_params["alpha"], self.hyper_params["beta"]
        diff = np.abs(I - I_new)
        if np.sum(diff != 0) < alpha * np.sum(I > 0):
            return diff.max() <= 255
        return diff.max() <= beta * 255

    # --------------- misc ---------------------------------------------------
    def _coverage_gain(self, gain):
        if not self.guided:                  # random fuzz
            return True
        if gain is None:
            return False
        return (gain[0] if isinstance(gain, tuple) else gain) > 0

    def _append_new_seeds(self, T, B_new, B_label_new):
        B_c, Bs, Bs_label = T
        Bs.append(np.stack(B_new))
        Bs_label.append(np.array(B_label_new))
        B_c.append(0)                        # freshness counter

    def _is_changed(self, I, I_new):
        return np.any(I != I_new)

    def _is_adversarial(self, images, labels, topk=1):
        with torch.no_grad():
            scores = self.criterion.model(images)
            topk_idx = scores.topk(topk,1,True,True)[1]
            wrong = ~(topk_idx.eq(labels.view(-1,1)))
            wrong_total = wrong.sum().item()
            return wrong_total, wrong.nonzero(as_tuple=True)[0]

    def _to_tensor(self, arr, norm=False):
        t = torch.from_numpy(arr).transpose(1,3).float()
        if norm: t = image_normalize(t/self.params.input_scale, self.params.dataset)
        return t.to(self.params.device)

    def _save_image(self, arr, path):
        save_image(torch.from_numpy(arr).transpose(1, 3), path, normalize=True)