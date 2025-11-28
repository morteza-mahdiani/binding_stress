# binding_probe.py
import math, io
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

@dataclass
class ProbeOutputs:
    pred_text: str
    answer_pos: int
    color_heatmap: Optional[np.ndarray]   # (H, W) in [0,1]
    shape_heatmap: Optional[np.ndarray]   # (H, W) in [0,1]
    color_logit: Optional[float]
    shape_logit: Optional[float]
    # Causal ablations: logit change after zeroing top-K saliency pixels
    color_drop_after_color_mask: Optional[float]
    color_drop_after_shape_mask: Optional[float]
    shape_drop_after_color_mask: Optional[float]
    shape_drop_after_shape_mask: Optional[float]

# --- Generic utilities -------------------------------------------------------

def _to_numpy(x: torch.Tensor):
    return x.detach().float().cpu().numpy()

def _minmax(x: np.ndarray):
    m, M = x.min(), x.max()
    if M <= m + 1e-12:
        return np.zeros_like(x)
    return (x - m) / (M - m)

def _upsample_map(m: np.ndarray, target_hw: Tuple[int,int]) -> np.ndarray:
    import cv2
    H, W = target_hw
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)

def _topk_mask(sal: np.ndarray, frac: float) -> np.ndarray:
    n = sal.size
    k = max(1, int(n * frac))
    thresh = np.partition(sal.ravel(), -k)[-k]
    return (sal >= thresh).astype(np.uint8)

# --- Attention based (optional, works if model returns attentions to image tokens)

def aggregate_answer_to_image_attention(attentions, answer_pos: int, image_span: slice) -> Optional[torch.Tensor]:
    """
    attentions: list of length L; each [B, heads, T, T]
    returns [#img_tokens] averaged over heads and last 4 layers.
    """
    if attentions is None or len(attentions) == 0:
        return None
    with torch.no_grad():
        L = len(attentions)
        use = attentions[max(0, L-4):]               # last 4 layers
        stacked = torch.stack([a[0] for a in use])    # [layers, heads, T, T]
        att = stacked.mean(dim=(0,1))                # [T, T]
        row = att[answer_pos]                        # [T]
        return row[image_span]                       # [#img_tokens]

# --- Gradient saliency on image (robust: no image token indices needed)

def grad_saliency_on_pixels(model, inputs, answer_pos: int, target_id: int):
    # clone fresh normal tensors
    enc = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.detach().clone()
        else:
            enc[k] = v
    if "pixel_values" not in enc:
        return None
    enc["pixel_values"].requires_grad_(True)

    with torch.enable_grad():
        out = model(**enc, output_attentions=False, use_cache=False)
        target_logit = out.logits[0, answer_pos, target_id]
        model.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=False)
        grad = enc["pixel_values"].grad  # [B, C, H, W]
        return grad.abs().sum(dim=1)[0]

def saliency_to_heatmap(sal: torch.Tensor):
    m = _to_numpy(sal)
    return _minmax(m)

# --- Causal ablation (zero out top-K pixels by saliency)

def zero_out_topk_pixels(pil_img: Image.Image, saliency: np.ndarray, frac: float=0.1):
    """
    Returns a new PIL image where the top-K fraction (by saliency) pixels are zeroed.
    """
    arr = np.array(pil_img).astype(np.float32)
    if arr.ndim == 2:  # grayscale -> Hx1
        arr = np.repeat(arr[..., None], 3, axis=2)
    mask = _topk_mask(saliency, frac).astype(bool)    # HxW
    arr[mask] = 0.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# --- Vocabulary helpers ------------------------------------------------------

def vocab_id_of(tokenizer, text: str) -> int:
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids) == 0:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0]

# --- Main probe --------------------------------------------------------------

@torch.no_grad()
def logits_for_answer_tokens(model, inputs, answer_pos: int, color_word: str, shape_word: str, tokenizer) -> Tuple[float, float]:
    out = model(**inputs, output_attentions=False, use_cache=False)
    logits = out.logits  # [B, T, V]
    color_id = vocab_id_of(tokenizer, color_word)
    shape_id = vocab_id_of(tokenizer, shape_word)
    return float(logits[0, answer_pos, color_id]), float(logits[0, answer_pos, shape_id])

def run_binding_probe(
    model, processor, tokenizer,
    prompt, pil_image,
    color_word, shape_word,
    return_attn: bool = False,
    saliency_frac: float = 0.10,
):
    device = next(model.parameters()).device

    # ---------- PASS A: no-grad, for pred + logits ----------
    with torch.no_grad():
        encA = processor(text=prompt, images=[pil_image], return_tensors="pt")
        encA = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in encA.items()}
        answer_pos = encA["input_ids"].shape[1] - 1
        outA = model(**encA, use_cache=False)
        pred_id = int(outA.logits[0, answer_pos].argmax().item())
        pred_text = tokenizer.decode([pred_id]).strip()

        # logits for target words
        color_id = vocab_id_of(tokenizer, color_word)
        shape_id = vocab_id_of(tokenizer, shape_word)
        color_logit = float(outA.logits[0, answer_pos, color_id])
        shape_logit = float(outA.logits[0, answer_pos, shape_id])

    # ---------- PASS B: grad-enabled, fresh inputs for saliency ----------
    # IMPORTANT: rebuild inputs; do NOT reuse encA tensors (they may be "inference tensors")
    # ---------- PASS B: grad-enabled, fresh inputs for saliency ----------
    with torch.enable_grad():
        encB = processor(text=prompt, images=[pil_image], return_tensors="pt")
        encB = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in encB.items()}
        answer_pos_B = encB["input_ids"].shape[1] - 1
        # IMPORTANT: make sure tensors are normal (not inference) and pixel_values has grad
        for k, v in encB.items():
            if isinstance(v, torch.Tensor):
                encB[k] = v.detach().clone()
        if "pixel_values" in encB:
            encB["pixel_values"].requires_grad_(True)

        # ids AFTER we know the words
        color_id = vocab_id_of(tokenizer, color_word)
        shape_id = vocab_id_of(tokenizer, shape_word)

        sal_color = grad_saliency_on_pixels(model, encB, answer_pos_B, target_id=color_id)

        # re-make a fresh enc for the second gradient call (separate graph)
        encC = processor(text=prompt, images=[pil_image], return_tensors="pt")
        encC = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in encC.items()}
        answer_pos_C = encC["input_ids"].shape[1] - 1
        for k, v in encC.items():
            if isinstance(v, torch.Tensor):
                encC[k] = v.detach().clone()
        if "pixel_values" in encC:
            encC["pixel_values"].requires_grad_(True)

        sal_shape = grad_saliency_on_pixels(model, encC, answer_pos_C, target_id=shape_id)


    color_heat = saliency_to_heatmap(sal_color) if sal_color is not None else None
    shape_heat = saliency_to_heatmap(sal_shape) if sal_shape is not None else None

    # causal ablations
    def _logit_after(img, target_id):
        with torch.no_grad():
            e2 = processor(text=prompt, images=[img], return_tensors="pt")
            e2 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in e2.items()}
            ap = e2["input_ids"].shape[1] - 1
            o2 = model(**e2, use_cache=False)
            return float(o2.logits[0, ap, target_id])

    color_drop_cc = color_drop_cs = shape_drop_cc = shape_drop_cs = None
    if color_heat is not None:
        img_cmask = zero_out_topk_pixels(pil_image, color_heat, frac=saliency_frac)
        img_smask = zero_out_topk_pixels(pil_image, shape_heat, frac=saliency_frac) if shape_heat is not None else None
        color_drop_cc = color_logit - _logit_after(img_cmask, color_id)
        color_drop_cs = color_logit - _logit_after(img_smask, color_id) if img_smask else None
        shape_drop_cc = shape_logit - _logit_after(img_cmask, shape_id)
        shape_drop_cs = shape_logit - _logit_after(img_smask, shape_id) if img_smask else None

    return ProbeOutputs(
        pred_text=pred_text,
        answer_pos=answer_pos,
        color_heatmap=color_heat,
        shape_heatmap=shape_heat,
        color_logit=color_logit,
        shape_logit=shape_logit,
        color_drop_after_color_mask=color_drop_cc,
        color_drop_after_shape_mask=color_drop_cs,
        shape_drop_after_color_mask=shape_drop_cc,
        shape_drop_after_shape_mask=shape_drop_cs,
    )