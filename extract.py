"""
Use models (CLIP, DINOv2, etc.) to extract latent space features from processed images
`data.py` => images (tensors) => `extract.py` => features (tensors)
processor has additional preprocessing that requires PIL Image
"""
from transformers import (
    CLIPModel,
    CLIPProcessor,
    ViTMSNModel,
    AutoFeatureExtractor,
    AutoModel,
    AutoImageProcessor
)
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str, device='cuda:0'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        if model_name == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)
        elif model_name == "vitmsn":
            self.model = ViTMSNModel.from_pretrained("facebook/vit-msn-base").to(self.device)
            self.processor = AutoFeatureExtractor.from_pretrained("facebook/vit-msn-base")
        elif model_name == "dinov2":
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        else:
            raise ValueError

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Extract features
        with torch.no_grad():
            if self.model_name == "clip":
                features = self.model.get_image_features(**inputs)
            elif self.model_name in ["vitmsn", "dinov2"]:
                features = self.model(**inputs).last_hidden_state[:, 0, :]
        return features
