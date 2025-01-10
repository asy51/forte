import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor, ViTMSNModel, AutoFeatureExtractor, AutoModel, AutoImageProcessor
import utils, baselines, prdc, prdc_per_point, eval
from tqdm import tqdm
from IPython import embed

def parse_args():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Process in-distribution and out-of-distribution images")
    parser.add_argument("--id_images_directories", nargs='+', help="List of directories containing all in-distribution images")
    parser.add_argument("--id_images_names", nargs='+', help="Names of each in-distribution image category")
    parser.add_argument("--ood_images_directories", nargs='+', help="List of directories containing all out-of-distribution images")
    parser.add_argument("--ood_images_names", nargs='+', help="Names of each out-of-distribution image category")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--device", type=str, default="cuda:0", help="Set device for computation")
    parser.add_argument("--embedding_dir", type=str, default='./embeddings', help="Where to store embeddings")
    parser.add_argument("--run_baselines", action='store_true', help='Whether or not to run baselines.')
    parser.add_argument("--no_run_baselines", action='store_false', dest='run_baselines', help='Explicitly do not run baselines.')
    parser.add_argument("--print_shapes", action='store_true', help='Whether to print shapes of vectors for sanity checks.')
    parser.add_argument("--num_seeds", type=int, default=5, help='Number of random seeds to use for evaluation')
    parser.add_argument("--id_filter", nargs='+', default=[''])
    parser.add_argument("--ood_filter", nargs='+', default=[''])

    return parser.parse_args()

def init_models(device):
    """
    Initialize the models used for feature extraction.
    Args:
        device (str): The device to load the models on.
    Returns:
        list: A list of tuples, each containing (model_name, model, processor).
    """
    models = [
        ("clip", CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device),
         CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")),
        ("vitmsn", ViTMSNModel.from_pretrained("facebook/vit-msn-base").to(device),
         AutoFeatureExtractor.from_pretrained("facebook/vit-msn-base")),
        ("dinov2", AutoModel.from_pretrained('facebook/dinov2-base').to(device),
         AutoImageProcessor.from_pretrained('facebook/dinov2-base'))
    ]
    return models

def get_prdc_features(X_id, X_ood, nearest_k=5):
    """
    Compute PRDC (Precision, Recall, Density, Coverage) features.
    Args:
        X_id (np.array): Features of in-distribution images.
        X_ood (np.array): Features of out-of-distribution images.
        nearest_k (int): Number of nearest neighbors to consider.
    Returns:
        np.array: Array of PRDC features.
    """
    prdc_metrics = prdc_per_point.compute_prdc_per_point(X_id, X_ood, nearest_k, realism=False)
    return np.column_stack((prdc_metrics['recall'], prdc_metrics['density'],    
                            prdc_metrics['precision'], prdc_metrics['coverage']))

def process_image_features(directories, names, models, args, is_id=True):
    """
    Process image features for a set of directories.

    Args:
        directories (list): List of image directories.
        names (list): List of category names for the images.
        models (list): List of models to use for feature extraction.
        args (argparse.Namespace): Parsed command-line arguments.
        is_id (bool): Whether processing in-distribution or out-of-distribution images.

    Returns:
        dict: Dictionary of processed features for each model.
    """
    # features = {name: dict() for name, _, _ in models}
    image_type = "in-distribution" if is_id else "out-of-distribution"

    assert len(directories) == 1
    for directory, name in tqdm(zip(directories, names), total=len(directories), desc=f"Processing {image_type} images"):
        model_features = utils.process_features(directory, models, name, args.embedding_dir, args.batch_size, args.device)
        print(f"Processed {image_type} images in {directory} categorized as {name}")
        
        # for model_name, model_feature in model_features.items():
        #     if args.print_shapes:
        #         print(f"Shape for {model_name} in {directory} is {model_feature.shape}")
        return model_features
    # for model_name in features:
    #     features[model_name].update(model_feature)
    #     # features[model_name] = np.concatenate(features[model_name], axis=0)
    # return features

def run_baseline_evaluations(id_features, ood_features, models, seed):
    """
    Run baseline evaluations and statistical tests.

    Args:
        id_features (dict): Dictionary of in-distribution image features.
        ood_features (dict): Dictionary of out-of-distribution image features.
        models (list): List of models used for feature extraction.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Dictionaries of baseline results and statistical test results.
    """
    baseline_results = {}
    stats_results = {}

    for model_name, _, _ in tqdm(models, desc="Running baseline evaluations"):
        id_train, id_held_out = train_test_split(id_features[model_name], test_size=0.33, random_state=seed)
        ood_test = ood_features[model_name]

        print(f"Running baseline evaluations for {model_name} embeddings...")
        baseline_results[model_name] = eval.run_evaluations(X_train_id=id_train, 
                                                             X_test_id=id_held_out, 
                                                             X_test_ood=ood_test)

        print(f"Performing statistical tests for {model_name} embeddings...")
        stats_results[model_name] = baselines.perform_stats_tests(id_train, ood_test, model_name)

    return baseline_results, stats_results

def compute_prdc_features(id_train, id_held_out, ood_features, seed):
    """np
    Compute PRDC features for in-distribution and out-of-distribution images.

    Args:
        id_train (dict): Dictionary of in-distribution training image features.
        id_held_out (dict): Dictionary of in-distribution held-out image features.
        ood_features (dict): Dictionary of out-of-distribution image features.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Arrays of PRDC features for ID train, ID held-out, and OOD images.
    """
    X_id_train_prdc, X_id_held_out_prdc, X_ood_prdc = [], [], []

    for model_name in tqdm(id_train, desc="Computing PRDC features"):
        id_train_part1, id_train_part2 = train_test_split(id_train[model_name], test_size=0.5, random_state=seed)
        
        X_id_train_prdc.append(get_prdc_features(id_train_part1, id_train_part2))

        # Fuse ID held-out and OOD features
        fused_features = np.concatenate([id_held_out[model_name], ood_features[model_name]])

        # Compute PRDC features for the fused dataset using id_train_part1 as reference
        prdc_results = get_prdc_features(id_train_part1, fused_features)

        # Extract PRDC results for ID held-out and OOD features
        num_id_held_out = id_held_out[model_name].shape[0]
        X_id_held_out_prdc.append(prdc_results[:num_id_held_out])
        X_ood_prdc.append(prdc_results[num_id_held_out:])

    return (np.concatenate(X_id_train_prdc, axis=1),
            np.concatenate(X_id_held_out_prdc, axis=1),
            np.concatenate(X_ood_prdc, axis=1))

def run_evaluation(id_features, ood_features, models, args, seed, id_filter, ood_filter):
    """
    Run a single evaluation with a given random seed.

    Args:
        id_features (dict): Dictionary of in-distribution image features.
        ood_features (dict): Dictionary of out-of-distribution image features.
        models (list): List of models used for feature extraction.
        args (argparse.Namespace): Parsed command-line arguments.
        seed (int): Random seed for this evaluation.

    Returns:
        dict: Evaluation results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"id_features with len {len(id_features['clip'].keys())}")
    if id_filter:
        id_features_filtered = dict()
        for model_name in id_features.keys():
            id_features_filtered[model_name] = np.stack([v for k,v in id_features[model_name].items() if any(desired_key in k for desired_key in id_filter)])
            print(f"id_features[{model_name}] filtered with shape {id_features_filtered[model_name].shape}")
        id_features = id_features_filtered
    else:
        id_features = {model_name: np.stack([v for v in id_features[model_name].values()])}
    print(f"ood_features with len {len(ood_features['clip'].keys())}")
    if ood_filter:
        ood_features_filtered = dict()
        for model_name in ood_features.keys():
            ood_features_filtered[model_name] = np.stack([v for k,v in ood_features[model_name].items() if any(desired_key in k for desired_key in ood_filter)])
            print(f"ood_features[{model_name}] filtered with shape {ood_features_filtered[model_name].shape}")
        ood_features = ood_features_filtered
    else:
        ood_features = {model_name: np.stack([v for v in ood_features[model_name].values()])}
    id_train, id_held_out = {}, {}
    for model_name, id_features_set in id_features.items():
        id_train[model_name], id_held_out[model_name] = train_test_split(id_features_set, test_size=0.33, random_state=seed)

    X_id_train_prdc, X_id_held_out_prdc, X_ood_prdc = compute_prdc_features(id_train, id_held_out, ood_features, seed)

    if args.print_shapes:
        print(f"X ID Train shape {X_id_train_prdc.shape}")
        print(f"X ID Held Out shape {X_id_held_out_prdc.shape}")
        print(f"X OOD PRDC shape {X_ood_prdc.shape}")

    return eval.run_evaluations(X_train_id=X_id_train_prdc, 
                                 X_test_id=X_id_held_out_prdc, 
                                 X_test_ood=X_ood_prdc)

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Running program with arguments {args} on {device}")
    
    models = init_models(device)

    id_features = process_image_features(args.id_images_directories, args.id_images_names, models, args)
    ood_features = process_image_features(args.ood_images_directories, args.ood_images_names, models, args, is_id=False)
    all_results = []
    for seed in tqdm(range(args.num_seeds), desc="Running evaluations across seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if args.run_baselines:
            print("Should not have done this")
            baseline_results, stats_results = run_baseline_evaluations(id_features, ood_features, models, seed)
            
            for model, result in baseline_results.items():
                print(f"Baseline results for {model} (seed {seed}): {result}")
            for model, stats in stats_results.items():
                print(f"Statistical Test Results for {model} (seed {seed}): {stats}")
        print("Running Evaluations")
        results = run_evaluation(id_features, ood_features, models, args, seed, id_filter=args.id_filter, ood_filter=args.ood_filter)
        all_results.append(results)

    # Calculate mean and standard deviation of results across seeds
    mean_results = {method: {metric: np.mean([r[method][metric] for r in all_results]) 
                             for metric in all_results[0][method]} 
                    for method in all_results[0]}
    std_results = {method: {metric: np.std([r[method][metric] for r in all_results]) 
                            for metric in all_results[0][method]} 
                   for method in all_results[0]}

    print("\nFinal evaluation results:")
    for method in mean_results:
        print(f"\n{method}:")
        for metric in mean_results[method]:
            print(f"  {metric}: {mean_results[method][metric]:.4f} ± {std_results[method][metric]:.4f}")

if __name__ == '__main__':
    main()


