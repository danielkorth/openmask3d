""" This file defines various embeddings models"""
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import clip
import torch
import torch.nn.functional as F
from typing import Literal

Embedders = Literal["clip", "siglip", "dinov2"]



# Base class for an embedding model
class EmbeddingModel:
    def __init__(self, device):
        self.device = device
        self.embedding_model = None
        self.embedding_preprocess = None
        self.tokenizer = None

        self.load_model()

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def preprocess_image(self, image):
        # Preprocess the image
        return self.embedding_preprocess(image)

    def encode_image(self, batched_image: torch.Tensor):
        """Encodes the preprocessed image using the model."""
        # Encode the image
        image_features = self.embedding_model.encode_image(batched_image.to(self.device))
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return image_features
    
    def encode_text(self, text: str):
        """Encodes the text using the model."""
        # Encode the text
        text_features = self.embedding_model.encode_text(text.to(self.device))
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features
    
    def tokenize(self, text: str):
        """Tokenizes the text using the model's tokenizer."""
        return self.tokenizer(text)

# SigLIP model class
class SigLIPModel(EmbeddingModel):
    def load_model(self):
        name = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
        self.embedding_model, self.embedding_preprocess = create_model_from_pretrained(name, device=self.device)
        self.tokenizer = get_tokenizer(name)
    @property
    def dim(self):
        return 1152

# CLIP model class
class CLIPModel(EmbeddingModel):
    def load_model(self):
        self.embedding_model, self.embedding_preprocess = clip.load('ViT-L/14@336px', device=self.device)
        self.tokenizer = clip.tokenize
    @property
    def dim(self):
        return 768

# DINO V2 model class
class DinoV2Model(EmbeddingModel):
    def load_model(self):
        self.embedding_model, self.embedding_preprocess = create_model_from_pretrained('hf-hub:timm/vit_large_patch14_dinov2.lvd142m', device=self.device)
        
    def encode_text(self, text: str):
        raise NotImplementedError("DinoV2Model does not support text encoding.")

    @property
    def dim(self):
        return None
