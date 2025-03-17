from __future__ import annotations
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import loralib as lora

from PIL import Image
from torchvision.ops import box_convert
import cv2
import supervision as sv

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def replace_linear_layers_with_lora(model, r=4, lora_alpha=2, exceptions=[]):
    for name, module in model.named_children():
        if module in exceptions:
            # Skip the modules in the exceptions list
            continue
        if isinstance(module, nn.Linear):
            if 'qkv' in name:
                #Replace the .qkv layer with lora.MergedLinear
                lora_layer = lora.MergedLinear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha, enable_lora=[True, True, True])
                setattr(model, name, lora_layer)
            elif 'reduction' in name:
                continue
            else:
                # Replace the nn.Linear layer with a LoRA layer
                lora_layer = lora.Linear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha)
                setattr(model, name, lora_layer)
        else:
            # Recursively apply to child modules
            replace_linear_layers_with_lora(module, r, lora_alpha, exceptions)

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # model.eval()
    return model

def load_model_with_lora(model_config_path: str, model_checkpoint_path: str, device: str = "cuda", rank: int = 8, lora_alpha: int = 8):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    replace_linear_layers_with_lora(model.backbone, r=rank, lora_alpha=lora_alpha)
    replace_linear_layers_with_lora(model.transformer.encoder.layers, r=rank, lora_alpha=lora_alpha)
    replace_linear_layers_with_lora(model.transformer.encoder.fusion_layers, r=rank, lora_alpha=lora_alpha)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # model.eval()
    return model

def process_image(image:np.ndarray):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_array = image.transpose(1,2,0).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_transformed, _ = transform(img, None)
    return img_transformed

def preprocess_images(images: np.ndarray):
    transformed = [process_image(image) for image in images]
    return torch.stack(transformed, dim=0)

def predict(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    
    caption = preprocess_caption(caption=caption)
    images = preprocess_images(images)
    images = images.to(device)

    outputs = model(images, captions=[caption]*len(images))
    interm_features = outputs["interm_feat"]

    return interm_features.squeeze(0).transpose(-1,-2)

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    for detection, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = map(int, detection)
        # Draw the bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Adjust the text properties
        font_scale = 0.3  # Smaller text size
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size

        # Position the text below the top-left corner of the bounding box
        text_x = x1
        text_y = y1 + text_h + 3  # A small margin below the top-left corner

        # Draw text background
        cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + 2), (0, 255, 0), -1)
        # Put the text on the image
        cv2.putText(annotated_frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return annotated_frame

def print_trainable_parameters(model):
    r"""Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

def build_vision_model(cfg, device):
    setfile = '../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    ckpt = '../GroundingDINO/weights/groundingdino_swint_ogc.pth'
    if cfg.agent.train_image_model:
        model = load_model_with_lora(setfile, ckpt, device, rank=cfg.agent.lora_rank, lora_alpha=cfg.agent.lora_alpha)
        lora.mark_only_lora_as_trainable(model)
        print_trainable_parameters(model)
    
    else:
        model = load_model(setfile, ckpt, device)
    return model

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer