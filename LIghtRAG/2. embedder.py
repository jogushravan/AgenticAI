# embedder.py
import torch
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_text_embedding(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    return outputs.cpu().numpy()

def get_image_embedding(images):
    inputs = clip_processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.cpu().numpy()
