import os
import copy
import random
import numpy as np
import time
import itertools

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from style_operator import Stylized
from genai import GenerativeAugmentor, GenerativeAugmentorStub
import image_transforms

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from fuzz_metrics import class_hist_entropy

# -----------------------------------------------------------
# Fuzzer Logger (aligned with NLC utility style)
# -----------------------------------------------------------
class Logger(object):
    def __init__(self, args, engine):
        import time
        self.name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log'
        self.args = args
        self.log_path = os.path.join(args.log_dir, self.name)
        os.makedirs(args.log_dir, exist_ok=True)
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
        print('AE (#Faults so far): %d' % engine.num_ae)
        print('#Outputs so far: %d' % engine.num_outputs)
        self.f.write(
            'Delta time: %fs, Epoch: %d, Current coverage: %f, Delta coverage:%f, '
            'AE: %d, Delta batch: %d, #Outputs: %d\n'
            % (engine.delta_time, engine.epoch, engine.criterion.current,
               engine.criterion.current - engine.initial_coverage,
               engine.num_ae, engine.delta_batch, engine.num_outputs)
        )

    def exit(self):
        self.f.close()


# -----------------------------------------------------------
# Image normalization helpers
# -----------------------------------------------------------
def image_normalize(image, dataset):
    if dataset == 'CIFAR10':
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2471, 0.2435, 0.2616))
    elif dataset == 'ImageNet':
        transform = transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return transform(image)


class Parameters(object):
    def __init__(self, base_args):
        # passthrough config
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
        self.wisdom_csv = getattr(base_args, 'wisdom_csv', None)
        self.genai_only = getattr(base_args, 'genai_only', False)

        # I/O dirs (ensure exist)
        self.log_dir = getattr(base_args, 'log_dir', './fuzz_guide/logs/')
        self.image_dir = getattr(base_args, 'image_dir', './fuzz_guide/images/')
        self.coverage_dir = getattr(base_args, 'coverage_dir', './fuzz_guide/ckpts/')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.coverage_dir, exist_ok=True)

        # data params
        self.batch_size = 50
        self.mutate_batch_size = 1
        self.nc = 3
        self.image_size = 128 if self.dataset == 'ImageNet' else 32
        self.input_shape = (1, self.image_size, self.image_size, 3)
        self.num_class = 100 if self.dataset == 'ImageNet' else 10
        self.num_per_class = 1000 // self.num_class

        # mutation budget
        self.input_scale = 255
        self.noise_data = False
        self.K = 64
        self.batch1 = 64
        self.batch2 = 16

        # NLC-like defaults (they often use alpha≈0.2, beta≈0.4)
        self.alpha = 0.2
        self.beta = 0.4
        self.TRY_NUM = 50
        self.save_every = 100
        self.output_dir = './fuzz_guide/output/'
        os.makedirs(self.output_dir, exist_ok=True)

        # transform families
        translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                            [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))        
        scale = list(itertools.product([getattr(image_transforms, "image_scale")], list(np.arange(0.8, 1, 0.05))))
        rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-30, 30))))

        contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [0.8 + 0.2 * k for k in range(7)]))
        brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(7)]))
        blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(10)]))

        self.stylized = Stylized(self.image_size)

        self.G = translation + scale + rotation
        self.P = contrast + brightness + blur
        self.S = list(itertools.product([self.stylized.transform], [0.4, 0.6, 0.8]))
        
        self.genai = GenerativeAugmentorStub("./onnx/unet.onnx", device=self.device)
        self.D = list(itertools.product(
            [self.genai],
            [{"prompt": p, "strength": s}
             for p in ["oil painting of a {label}",
                       "low-poly {label} sculpture",
                       "macro photo of {label}"]
             for s in (0.4, 0.6, 0.8)]
        ))
        
        if self.genai_only:
            self.G = self.P = self.S = []

        self.save_batch = False


class INFO(dict):
    @staticmethod
    def _k(arr):
        return id(arr)

    def __getitem__(self, arr):
        return super().get(self._k(arr), (arr, 0))

    def __setitem__(self, arr, tpl):
        super().__setitem__(self._k(arr), tpl)

    def __missing__(self, arr):
        return (arr, 0)
    

# -----------------------------------------------------------
# Main Fuzzer (aligned with NLC fuzzing loop)
# -----------------------------------------------------------
class Fuzzer:
    def __init__(self, params, model, criterion, guided: bool = True):
        """
        model: the classification model (for adversarial checks and summaries)
        criterion: coverage criterion object (has .current, .calculate(), .update(), .gain(), .save())
        """
        self.params = params
        self.model = model.eval().to(params.device)
        self.criterion = criterion
        self.guided = guided

        self.time_slot = 60 * 10
        self.time_idx = 0
        self.epoch = 0
        self.info = INFO()
        self.delta_time = 0
        self.delta_batch = 0
        self.num_ae = 0          # #Faults so far (accepted that flip prediction)
        self.num_outputs = 0     # #Outputs so far (total accepted mutations)
        self.initial_coverage = copy.deepcopy(criterion.current)

        # default hyper-parameters
        self.hyper_params = dict(
            alpha=0.4,   # pixel proportion budget (NLC often uses ~0.2; kept here as a knob)
            beta=0.8,    # max abs change
            TRY_NUM=50,
            p_min=0.01, 
            gamma=5, 
            K=64
        )

        self.logger = Logger(params, self)

    # -------------- public --------------------------------------------------
    def run(self, I_input, L_input):
        """
        I_input : list/ndarray raw images in range [0,1] HWC
        L_input : list/ndarray ground truth labels
        """
        T = self._preprocess(I_input, L_input)
        
        B, B_label, _ = self._select_next(T)   # current batch
        self.epoch = 0
        start = time.time()

        while not self._should_stop():
            if self.epoch % 500 == 0:
                self.logger.update(self)

            S, S_label = B, B_label
            Ps = self._power_schedule(S, self.hyper_params["K"])
            
            B_new, B_old, B_label_new = [], [], []

            # ---------------- mutation phase -------------------------------
            for s_i, (I, L) in enumerate(zip(S, S_label)):
                n_trials = Ps(s_i) if self.guided else 1
                for _ in range(n_trials):
                    I_new, _op = self._mutate(I)

                    # random mode: keep any changed mutation
                    if not self.guided:
                        if self._is_changed(I, I_new):
                            B_new.append(I_new);   B_old.append(I);   B_label_new.append(L)
                        continue
                    
                    # guided mode
                    if self._detect_failedTest(I_new):
                        # skipped by design in NLC (kept for completeness)
                        pass
                    elif self._is_changed(I, I_new):
                        torch_img = self._to_tensor(np.stack([I_new]), norm=True)
                        torch_lbl = torch.tensor([L], device=self.params.device)

                        if self.params.criterion in ['LSC', 'DSC', 'MDSC']:
                            cov_dict = self.criterion.calculate(torch_img, torch_lbl)
                        else:
                            cov_dict = self.criterion.calculate(torch_img)
                        gain = self.criterion.gain(cov_dict)
                        if self._coverage_gain(gain):
                            self.criterion.update(cov_dict, gain)
                            B_new.append(I_new)
                            B_old.append(I)
                            B_label_new.append(L)
            # ------------- post-processing -------------------
            if B_new:                                   # accepted new seeds
                self._append_new_seeds(T, B_new, B_label_new)
                self.delta_batch += 1

                # Count #Outputs increment
                self.num_outputs += len(B_new)
                # adversarial statistics (#Faults increment)
                tensor_img  = self._to_tensor(np.stack(B_new), norm=True)
                tensor_lbl  = torch.tensor(B_label_new, device=self.params.device)
                wrong_total, wrong_idx = self._is_adversarial(tensor_img, tensor_lbl)
                self.num_ae += int(wrong_total)

                # periodically save montage
                if self.epoch % self.params.save_every == 0:
                    self._save_image(np.stack(B_new) / self.params.input_scale,
                                     os.path.join(self.params.image_dir, f"{self.epoch:03d}_new.jpg"))
                    self._save_image(np.stack(B_old) / self.params.input_scale,
                                     os.path.join(self.params.image_dir, f"{self.epoch:03d}_old.jpg"))
                    if wrong_total > 0:
                        save_image(tensor_img[wrong_idx],
                                   os.path.join(self.params.image_dir, f"{self.epoch:03d}_ae.jpg"),
                                   normalize=True)

            B, B_label, _ = self._select_next(T)
            self.epoch += 1
            self.delta_time = time.time() - start
        
        # ======== Final metrics ========
        # Gather all generated (accepted) images
        all_generated = []
        for batch in T[1]:
            all_generated.extend(batch)
        
        # Prepare real/fake tensors for IS/FID
        real_imgs_tensor = [torch.from_numpy(img).permute(2,0,1).float() for img in I_input]  # HWC->CHW
        fake_imgs_tensor = [torch.from_numpy(img).permute(2,0,1).float() for img in all_generated]
        metrics = self.compute_fid_is(real_imgs_tensor, fake_imgs_tensor)

        # Model-based summary on generated set (#classes, entropy of predicted dist)
        # if all_generated:
        #     norm = (transforms.Normalize((0.4914,0.4822,0.4465), (0.2471,0.2435,0.2616))
        #             if self.params.dataset=="CIFAR10"
        #             else transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)))
        #     hist = torch.zeros(self.params.num_class, device=self.params.device)
        #     with torch.no_grad():
        #         for i in range(0, len(all_generated), 64):
        #             batch = [torch.from_numpy(img).permute(2,0,1).float() for img in all_generated[i:i+64]]
        #             batch_tensor = torch.stack(batch).to(self.params.device) / self.params.input_scale
        #             preds = self.model(norm(batch_tensor)).argmax(1)
        #             hist += torch.bincount(preds, minlength=len(hist))
        #     prob = hist / hist.sum().clamp_min(1)  # avoid divide-by-zero
        #     entropy = float(-(prob * torch.log(prob.clamp_min(1e-12))).sum())
        #     num_classes = int((hist > 0).sum())
        # else:
        #     entropy = float('nan')
        #     num_classes = 0
        
        
        if all_generated:
            gen_chw = [torch.from_numpy(img).permute(2,0,1).float() for img in all_generated]
            gen_tensor = torch.stack(gen_chw, dim=0).to(self.params.device) / self.params.input_scale  # [0,1]
            num_classes, entropy = class_hist_entropy(self.model, gen_tensor, dataset=self.params.dataset, device=self.params.device)
        else:
            num_classes, entropy = 0, float('nan')
        
        metrics['num_classes'] = num_classes
        metrics['entropy'] = entropy
        metrics['faults'] = int(self.num_ae)
        metrics['outputs'] = int(self.num_outputs)
        metrics['fault_ratio'] = float(self.num_ae / max(1, self.num_outputs))

        return metrics
    
    def compute_fid_is(self, real_imgs, fake_imgs):
        """
        real_imgs, fake_imgs: list[Tensor CHW] or Tensor BCHW
        Values may be [0,1] float or [0,255] uint8; we convert to uint8 to avoid
        ambiguity and let torchmetrics handle resizing / scaling internally.
        """
        # stack helpers
        def _to_bchw(x):
            if torch.is_tensor(x):
                t = x
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                return t
            else:
                return torch.stack([xi if torch.is_tensor(xi) else torch.from_numpy(xi) for xi in x], dim=0)

        real = _to_bchw(real_imgs).float()
        fake = _to_bchw(fake_imgs).float()

        # If images are already [0,1], convert to uint8; if [0,255], first clamp
        def to_uint8(t):
            if t.max() <= 1.0:
                t = (t * 255.0)
            return t.round().clamp(0, 255).to(torch.uint8)

        real_uint8 = to_uint8(real)
        fake_uint8 = to_uint8(fake)

        # torchmetrics devices
        device = torch.device(self.params.device if torch.cuda.is_available() or str(self.params.device).startswith('cuda') else 'cpu')

        # Inception Score: DO NOT pass feature=... (that’s for FID)
        inception = InceptionScore(splits=10).to(device)
        fid = FrechetInceptionDistance(feature=2048).to(device)

        with torch.no_grad():
            # IS: only fake
            for i in range(0, fake_uint8.size(0), 64):
                inception.update(fake_uint8[i:i+64].to(device))
            IS_mean = inception.compute()[0].item()

            # FID: real + fake
            for i in range(0, real_uint8.size(0), 64):
                fid.update(real_uint8[i:i+64].to(device), real=True)
            for i in range(0, fake_uint8.size(0), 64):
                fid.update(fake_uint8[i:i+64].to(device), real=False)
            FID_val = fid.compute().item()

        return {"IS": IS_mean, "FID": FID_val}

    def exit(self):
        self.logger.update(self)
        # guard if criterion has save
        if hasattr(self.criterion, "save"):
            self.criterion.save(os.path.join(self.params.coverage_dir, "coverage_final.pt"))
        self.logger.exit()

    # -------------- internal helpers ---------------------------------------
    def _should_stop(self):
        # tune to your budget
        return (self.epoch > 10000) or (self.delta_time > 60*60)
        # return self.epoch > 100

    def _detect_failedTest(self, I_new):
        # NLC disables oracle; keep False
        return False
    
    def _preprocess(self, imgs, labels):
        # shuffle & scale to [0,255] uint8 space (as in the original code)
        order = np.random.permutation(len(imgs))
        imgs = [imgs[i] * self.params.input_scale for i in order]
        labels = [labels[i] for i in order]
        Bs = self._to_batch(imgs)
        Bs_label = self._to_batch(labels)
        # B_c (priorities), Bs (batches of images), Bs_label (batches of labels)
        return [0]*len(Bs), Bs, Bs_label        

    def _to_batch(self, seq):
        batches, cur = [], []
        for x in seq:
            if cur and len(cur) % self.params.mutate_batch_size == 0:
                batches.append(np.stack(cur)); cur = []
            cur.append(x)
        if cur: batches.append(np.stack(cur))
        return batches

    # ---------- per-epoch primitives ---------------------------------------
    def _select_next(self, T):
        B_c, Bs, Bs_label = T
        # simple uniform pick (priority schedule below can be extended)
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
            p = beta * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0))
            potentials.append(max(p, 0))
        denom = np.sum(potentials)
        if denom <= 0:
            # fallback: one try each
            return lambda idx: 1
        potentials = np.array(potentials) / denom
        return lambda idx: int(np.ceil(potentials[idx] * K))

    # ---------- mutation & acceptance --------------------------------------
    def _mutate(self, I):
        G, P, S, D = self.params.G, self.params.P, self.params.S, self.params.D
        I0, state = self.info[I]

        for _ in range(self.hyper_params["TRY_NUM"]):
            if self.params.genai_only:
                t, p = random.choice(D)
            elif state == 0 and not self.params.genai_only:
                t, p = random.choice(G + P + S)
            else:
                t, p = random.choice(P + S)

            # A quick patch for stylized transform
            I_float32 = I.astype("float32", copy=False)
            I_new = t(I_float32, p).reshape(self.params.input_shape[1:])
            I_new = np.clip(I_new, 0, 255)

            if (t, p) in S or self._pixel_budget(I0, I_new):
                # update INFO cache
                if (t, p) in G:
                    state = 1
                    self.info[I_new] = (np.clip(t(I0, p), 0, 255), state)
                else:
                    self.info[I_new] = (I0, state)
                return I_new, (t, p)

        # fallback (no mutation satisfied)
        return I, (t, p)

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
        # gain could be tuple in some criteria
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
            scores = self.model(images)
            topk_idx = scores.topk(topk, 1, True, True)[1]
            wrong = ~(topk_idx.eq(labels.view(-1, 1)))
            wrong_total = wrong.sum().item()
            return wrong_total, wrong.nonzero(as_tuple=True)[0]

    def _to_tensor(self, arr, norm=False):
        t = torch.from_numpy(arr).transpose(1, 3).float()
        if norm:
            t = image_normalize(t / self.params.input_scale, self.params.dataset)
        return t.to(self.params.device)

    def _save_image(self, arr, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(torch.from_numpy(arr).transpose(1, 3), path, normalize=True)
