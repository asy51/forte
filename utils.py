import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import fetch.transforms as FT
from fetch.oai import base_transforms
import monai.transforms as MT
import torchvision.transforms as TT

img_size=320
topilimage = TT.ToPILImage()
tx = MT.Compose([
    FT.load_image,
    base_transforms,
    # MT.Lambda(lambda x: x.rot90(k=1, dims=[-2,-1]).flip(dims=[-1])),
    MT.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True, relative=False),
    # MT.CenterSpatialCrop(roi_size=(img_size, img_size)),
    MT.Resize((img_size, img_size)),
    MT.ToTensor(track_meta=False),
    lambda x: [topilimage(slc).convert("RGB") for slc in x],
])

def load_image_paths(directory: str, suffix: str = None) -> List[str]:
    """
    Load image paths from a directory and its subdirectories with an optional suffix filter.

    Args:
        directory (str): The root directory to search for images.
        suffix (str, optional): File suffix to filter images. Defaults to None.

    Returns:
        List[str]: A list of full paths to the image files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    image_paths = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(directory)
        for filename in filenames
        if not suffix or filename.endswith(suffix)
    ]

    print(f"Found {len(image_paths)} images in '{directory}'" + 
          (f" with suffix '{suffix}'" if suffix else ""))
    
    return image_paths

def extract_features_batch(img_paths: List[str], device: torch.device, 
                           models: List[Tuple[str, Any, Any]]) -> Dict[str, torch.Tensor]:
    """
    Extract features for a batch of images using multiple models.

    Args:
        img_paths (List[str]): List of paths to the images.
        device (torch.device): The device to run the models on.
        models (List[Tuple[str, Any, Any]]): List of (model_name, model, processor) tuples.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of features for each model.

    Raises:
        RuntimeError: If feature extraction fails for any reason.
    """
    try:
        images = []
        for path in img_paths:
            if path.endswith(('jpg', 'png', 'jpeg')):
                images.append(Image.open(path).convert("RGB"))
            else:
                images.extend(tx(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open images: {e}")
    # return images
    all_features = {}

    # Process images once for all models
    _, _, first_processor = models[0]
    inputs = first_processor(images=images, return_tensors="pt", padding=True).to(device)

    for model_name, model, _ in models:
        try:
            with torch.no_grad():
                if model_name == "clip":
                    features = model.get_image_features(**inputs)
                elif model_name in ["vitmsn", "dinov2"]:
                    features = model(**inputs).last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Unsupported model: {model_name}")

            all_features[model_name] = features  # Keep features on GPU
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed for model {model_name}: {e}")

    return all_features

def process_features(directory: str, models: List[Tuple[str, Any, Any]], name: str, 
                     save_embedding_to: str, batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    """
    Process features for all images in a directory using multiple models.

    Args:
        directory (str): Directory containing the images.
        models (List[Tuple[str, Any, Any]]): List of (model_name, model, processor) tuples.
        name (str): Name for the current processing task (used in filenames).
        save_embedding_to (str): Directory to save the extracted features.
        batch_size (int): Number of images to process in each batch.
        device (torch.device): The device to run the models on.

    Returns:
        Dict[str, np.ndarray]: Dictionary of features for each model.

    Raises:
        RuntimeError: If feature processing fails for any reason.
    """
    output_dir = os.path.join("./", save_embedding_to)
    os.makedirs(output_dir, exist_ok=True)

    img_files = load_image_paths(directory)
    all_features = {}
    models_to_process = []

    for model_name, _, _ in models:
        embedding_filename = os.path.join(output_dir, f"{name}_{model_name}_features.npz")
        if os.path.exists(embedding_filename):
            print(f"Loading pre-computed features from {embedding_filename}")
            with np.load(embedding_filename, mmap_mode='r') as data:
                all_features[model_name] = data['features']
        else:
            all_features[model_name] = []
            models_to_process.append(model_name)

    if models_to_process:
        for i in tqdm(range(0, len(img_files), batch_size), desc="Processing batches"):
            try:
                batch = img_files[i:i+batch_size]
                features = extract_features_batch(batch, device, models)

                for model_name, model_features in features.items():
                    if model_name in models_to_process:
                        all_features[model_name].append(model_features.cpu().numpy())
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue

        for model_name in models_to_process:
            try:
                features_combined = np.concatenate(all_features[model_name], axis=0)
                embedding_filename = os.path.join(output_dir, f"{name}_{model_name}_features.npz")
                print(f"Saving features to {embedding_filename}")
                np.savez_compressed(embedding_filename, features=features_combined)
                all_features[model_name] = features_combined
            except Exception as e:
                raise RuntimeError(f"Failed to save features for model {model_name}: {e}")

    return all_features
