from datasets import load_dataset, get_dataset_config_names, concatenate_datasets, interleave_datasets
import re

def load_multiple_datasets(dataset_urls, split="train", mixing_strategy="interleave", weights=None):
    """
    Load and combine multiple datasets.

    Args:
        dataset_urls: List of dataset URLs/names or comma-separated string
        split: Dataset split to load
        mixing_strategy: How to combine datasets ("interleave" or "concatenate")
        weights: Optional list of weights for interleaving (must sum to 1.0)

    Returns:
        Combined dataset
    """
    # Parse dataset URLs
    if isinstance(dataset_urls, str):
        # Split by comma and strip whitespace
        dataset_urls = [url.strip() for url in dataset_urls.split(',') if url.strip()]

    if not dataset_urls:
        raise ValueError("No datasets specified")

    # Load all datasets
    datasets = []
    for url in dataset_urls:
        print(f"Loading dataset: {url}")
        ds = load_hf_dataset(url, split=split)
        datasets.append(ds)
        print(f"  → Loaded {len(ds)} examples")

    # Combine datasets
    if len(datasets) == 1:
        return datasets[0]

    if mixing_strategy == "concatenate":
        print(f"Concatenating {len(datasets)} datasets...")
        combined = concatenate_datasets(datasets)
    elif mixing_strategy == "interleave":
        print(f"Interleaving {len(datasets)} datasets...")
        # If no weights provided, use equal weights
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        combined = interleave_datasets(datasets, probabilities=weights, seed=42)
    else:
        raise ValueError(f"Unknown mixing strategy: {mixing_strategy}")

    print(f"Combined dataset size: {len(combined)} examples")
    return combined


def load_hf_dataset(dataset_url: str, split="train"):
    """
    Accepts various formats:
      - Short name: "wikitext"
      - With config: "wikitext/wikitext-2-raw-v1"
      - Full URL: "https://huggingface.co/datasets/wikitext"
      - Full URL with config: "https://huggingface.co/datasets/wikitext/wikitext-2-raw-v1"
      - Viewer URL: "https://huggingface.co/datasets/wikitext?subset=wikitext-2-raw-v1"

    For datasets with multiple configs, you MUST specify the config either:
      - In the path: "wikitext/wikitext-2-raw-v1"
      - In the URL: "https://huggingface.co/datasets/wikitext?subset=wikitext-2-raw-v1"
    """
    dataset_name = dataset_url
    config_name = None

    # Parse HuggingFace URLs
    # Format 1: https://huggingface.co/datasets/DATASET?subset=CONFIG&split=SPLIT
    # Format 2: https://huggingface.co/datasets/DATASET/tree/refs%2Fconvert%2Fparquet/CONFIG
    # Format 3: https://huggingface.co/datasets/DATASET (no config)
    if dataset_url.startswith('http'):
        # Extract from URL
        url_match = re.match(r'https?://huggingface\.co/datasets/([^/?]+(?:/[^/?]+)?)', dataset_url)
        if url_match:
            dataset_name = url_match.group(1)

            # Check for ?subset=CONFIG parameter
            subset_match = re.search(r'[?&]subset=([^&]+)', dataset_url)
            if subset_match:
                config_name = subset_match.group(1)

            # Check for /tree/refs%2Fconvert%2Fparquet/CONFIG format
            elif '/tree/' in dataset_url:
                tree_match = re.search(r'/tree/[^/]+/(.+?)(?:/|$)', dataset_url)
                if tree_match:
                    config_name = tree_match.group(1)

    # Parse path format: "dataset/config" or "owner/dataset/config"
    if '/' in dataset_name and config_name is None:
        parts = dataset_name.split('/')
        if len(parts) == 2:
            # Could be "owner/dataset" or "dataset/config"
            # We'll try to detect by checking if it has configs
            try:
                # First try as "owner/dataset"
                configs = get_dataset_config_names(dataset_name)
                # If it has configs, user needs to specify one
                if len(configs) > 1:
                    print(f"⚠️  Dataset '{dataset_name}' has multiple configs: {configs}")
                    print(f"   Please specify one, e.g.: {dataset_name}/{configs[0]}")
                    raise ValueError(
                        f"Config name is missing. Available configs: {configs}\n"
                        f"Example: '{dataset_name}/{configs[0]}'"
                    )
            except:
                # If that fails, try splitting as dataset/config
                dataset_name, config_name = parts[0], parts[1]
        elif len(parts) == 3:
            # Format: "owner/dataset/config"
            dataset_name = f"{parts[0]}/{parts[1]}"
            config_name = parts[2]
        elif len(parts) > 3:
            # Format: "owner/dataset/config/subconfig/..."
            dataset_name = f"{parts[0]}/{parts[1]}"
            config_name = '/'.join(parts[2:])

    print(f"Loading dataset: '{dataset_name}'" + (f" with config: '{config_name}'" if config_name else ""))

    # Try to load the dataset
    try:
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        return ds
    except ValueError as e:
        error_msg = str(e)

        # If it's a missing config error, try to provide helpful message
        if "Config name is missing" in error_msg or "Available configs:" in error_msg:
            try:
                configs = get_dataset_config_names(dataset_name)
                raise ValueError(
                    f"Dataset '{dataset_name}' requires a config.\n"
                    f"Available configs: {configs}\n\n"
                    f"Please use one of these formats:\n"
                    f"  - '{dataset_name}/{configs[0]}'\n"
                    f"  - 'https://huggingface.co/datasets/{dataset_name}?subset={configs[0]}'"
                ) from e
            except:
                pass

        # Re-raise the original error
        raise
