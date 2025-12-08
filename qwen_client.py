# qwen_client.py
import os
from typing import List, Dict, Any, Tuple

import torch
from copy import deepcopy
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from huggingface_hub import snapshot_download

os.environ["HF_HOME"] = f"../scratch/{os.environ['USER']}/hf_cache"
local_dir = f"../scratch/{os.environ['USER']}/models/Qwen2_5_VL_7B_Instruct"

# ---- Config ----
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=local_dir,
    local_dir_use_symlinks=False  # real files (no symlinks to tiny /home)
)
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "auto"  # uses GPU if available

# ---- Lazy singletons (loaded once) ----
_MODEL = None
_PROCESSOR = None

def _ensure_loaded(device_map="auto"):
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map=device_map
    )
    _MODEL.eval()

def get_model():
    return _MODEL

def get_processor():
    return _PROCESSOR

def get_tokenizer():
    return get_processor().tokenizer

# def _load_model():
#     global _processor, _model
#     if _processor is None or _model is None:
#         _processor = AutoProcessor.from_pretrained(local_dir)
#         _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_dir, device_map="auto")        
#         _model.eval()
#     return _processor, _model

def _collect_images_and_template(messages: List[Dict[str, Any]]) -> Tuple[str, list]:
    """
    Converts our vision-chat `messages` into Qwen chat template text + image list.
    We keep a flat image list in the same order as <image> placeholders.
    """
    proc = get_processor()

    # Build Qwen-style messages with explicit placeholders
    tmpl_msgs = []
    flat_images = []

    for m in messages:
        role = m["role"]
        pieces = m["content"]  # list of {type: "text"|"image", ...}
        out_content = []

        for p in pieces:
            if p["type"] == "text":
                out_content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                img_path = p.get("image_url") or p.get("image")
                if not img_path or not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                # Insert placeholder for the image; add actual PIL to flat_images
                out_content.append({"type": "image"})
                flat_images.append(Image.open(img_path).convert("RGB"))
            else:
                # Ignore unknown content types
                continue

        tmpl_msgs.append({"role": role, "content": out_content})

    # Turn into a single prompt string with <image> placeholders
    prompt = proc.apply_chat_template(
        tmpl_msgs,
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt, flat_images

def build_prompt_and_image(messages, probe_single_image: bool = True):
    """
    Returns (prompt, pil_image) for the binding probe.
    If probe_single_image=True, we strip images from shots and keep only the
    final user message's image so the processor sees exactly 1 image.
    We preserve all text (system + shot Q/As) to keep task conditioning.
    """
    if not probe_single_image:
        prompt, images = _collect_images_and_template(messages)
        assert len(images) == 1, "Binding probe assumes 1 query image per episode."
        return prompt, images[0]

    # --- sanitize messages: remove images from all but last user turn ---
    msgs = deepcopy(messages)

    # find the index of the final user message (the query)
    last_user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise RuntimeError("No user message found in messages.")

    # helper to strip image parts from a message's content
    def strip_images(m):
        c = m.get("content")
        if isinstance(c, list):
            newc = []
            for part in c:
                if isinstance(part, dict):
                    # keep text parts only
                    if part.get("type") == "text" or "text" in part:
                        newc.append(part)
                    # drop image parts ('type' == 'image' or keys: image/path/url)
                else:
                    newc.append(part)
            m["content"] = newc
        # if content is a string, leave it
        return m

    # remove images from all user messages except the last one
    for i, m in enumerate(msgs):
        if i == last_user_idx:
            continue
        if m.get("role") == "user":
            strip_images(m)

    # in the last user message, keep only ONE image (the last, if multiple)
    last_user = msgs[last_user_idx]
    c = last_user.get("content")
    if isinstance(c, list):
        img_parts = []
        text_parts = []
        for part in c:
            if isinstance(part, dict) and (
                part.get("type") == "image" or "image" in part or "path" in part or "url" in part
            ):
                img_parts.append(part)
            else:
                text_parts.append(part)
        # keep only the final image (query image)
        keep_img = img_parts[-1:] if img_parts else []
        last_user["content"] = text_parts + keep_img

    # now collect prompt and the single image
    prompt, images = _collect_images_and_template(msgs)
    if len(images) != 1:
        raise AssertionError(f"Expected exactly 1 image after sanitization, got {len(images)}.")
    return prompt, images[0]

def qwen_infer(
    messages: List[Dict[str, Any]],
    *,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **gen_kwargs: Any,
) -> str:
    """
    Runs one inference on Qwen2.5-VL with default-safe generation params,
    while allowing extra Hugging Face `generate` kwargs via **gen_kwargs.
    """
    _ensure_loaded()
    proc = get_processor()
    model = get_model()

    prompt, images = _collect_images_and_template(messages)   # your helper
    inputs = proc(text=prompt, images=images, return_tensors="pt")
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k,v in inputs.items()}
    input_ids_len = inputs["input_ids"].shape[1]
    # Merge defaults with user-supplied overrides
    generate_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        # "top_p": top_p,
        # good defaults for chatty VLMs:
        "do_sample": temperature > 0.0,
        # "use_cache": True,
    }
    generate_args.update(gen_kwargs)  # allow caller to override/add (e.g., repetition_penalty=1.1)

    # Optional: light validation/sanitization
    # Prevent incompatible combos; examples:
    if generate_args.get("temperature", 0.0) == 0.0:
        # zero temperature makes do_sample pointless
        generate_args["do_sample"] = False

    with torch.no_grad():
        # print(generate_args["do_sample"])
        # print(messages)
        out = model.generate(**inputs, **generate_args)
    # text = proc.batch_decode(out, skip_special_tokens=True)[0]
    
    
    # decode only the newly generated part
    generated_ids = out[:, input_ids_len:]
    # Decode only the generated slice
    text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(f"Raw model output: {text}")
    # If you used chat template, strip the prompt echo (optional, depends on template behavior)
    return text.strip()
