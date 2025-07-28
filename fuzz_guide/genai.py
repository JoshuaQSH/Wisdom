from onnxruntime import InferenceSession, SessionOptions
import numpy as np, cv2, random


class GenerativeAugmentor:
    def __init__(self, onnx_path, device='cpu'):
        so = SessionOptions()
        so.graph_optimization_level = 3
        self.ort = InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider" if device!='cpu' else "CPUExecutionProvider"])
        self.device = device

    def __call__(self, img_np, cfg):
        """
        img_np : (H,W,C) uint8 numpy
        cfg    : dict { 'prompt':..., 'strength':0‒1 }
        """
        # 1) normalise to [-1,1] float32 expected by SD U-Net
        x = (img_np.astype("float32")/127.5 - 1).transpose(2,0,1)[None]
        # 2) run diffusion – here just one denoise step for brevity
        out = self.ort.run(None, {"latent":x, **cfg})[0]
        # 3) de-normalise back to uint8 HWC
        out = ((out[0].transpose(1,2,0)+1)*127.5).clip(0,255).astype("uint8")
        return out

class GenerativeAugmentorStub:
    """Stub that mimics the API but returns a noised-and-sharpened image."""
    def __init__(self, *_args, **_kw): 
        pass
    
    def __call__(self, img_np, cfg):
        # img_np : HWC uint8
        noise = np.random.normal(0, 10, img_np.shape).astype("float32")
        out = cv2.add(img_np.astype("float32"), noise)              # add noise
        out = cv2.GaussianBlur(out, (0,0), sigmaX=1.2)              # blur
        out = cv2.addWeighted(out, 1.5, -0.5*out + 128, 0, 0)       # sharpen
        return np.clip(out, 0, 255).astype("uint8")