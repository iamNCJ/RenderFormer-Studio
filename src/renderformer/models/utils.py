"""
Utility functions for the render transformer model.
"""

from typing import List, Tuple


def map_layer_indices(use_layers: List[int]) -> Tuple[List[int], List[int]]:
    """
    Maps a list of layer indices to unique layers and provides back-mapping indices.
    
    Args:
        use_layers: List of layer indices that may contain duplicates
        
    Returns:
        Tuple containing:
        - unique_layers: List of unique layer indices in order of first appearance
        - use_indices: List mapping each element in use_layers to its index in unique_layers
        
    Example:
        >>> use_layers = [0, 0, 1, 1, 3, 3]
        >>> unique_layers, use_indices = map_layer_indices(use_layers)
        >>> print(unique_layers)  # [0, 1, 3]
        >>> print(use_indices)    # [0, 0, 1, 1, 2, 2]
    """
    # Create mapping from layer index to unique index
    layer_to_unique = {}
    unique_layers = []
    
    for layer_idx in use_layers:
        if layer_idx not in layer_to_unique:
            layer_to_unique[layer_idx] = len(unique_layers)
            unique_layers.append(layer_idx)
    
    # Create back-mapping indices
    use_indices = [layer_to_unique[layer_idx] for layer_idx in use_layers]
    
    return unique_layers, use_indices


def get_unique_layers(use_layers: List[int]) -> List[int]:
    """
    Get unique layer indices from a list, preserving order of first appearance.
    
    Args:
        use_layers: List of layer indices that may contain duplicates
        
    Returns:
        List of unique layer indices in order of first appearance
        
    Example:
        >>> use_layers = [0, 0, 1, 1, 3, 3]
        >>> unique_layers = get_unique_layers(use_layers)
        >>> print(unique_layers)  # [0, 1, 3]
    """
    seen = set()
    unique_layers = []
    
    for layer_idx in use_layers:
        if layer_idx not in seen:
            seen.add(layer_idx)
            unique_layers.append(layer_idx)
    
    return unique_layers


def get_layer_mapping_indices(use_layers: List[int]) -> List[int]:
    """
    Get mapping indices that link back to unique layers.
    
    Args:
        use_layers: List of layer indices that may contain duplicates
        
    Returns:
        List mapping each element in use_layers to its index in the unique layers list
        
    Example:
        >>> use_layers = [0, 0, 1, 1, 3, 3]
        >>> mapping_indices = get_layer_mapping_indices(use_layers)
        >>> print(mapping_indices)  # [0, 0, 1, 1, 2, 2]
    """
    layer_to_unique = {}
    unique_layers = []
    
    for layer_idx in use_layers:
        if layer_idx not in layer_to_unique:
            layer_to_unique[layer_idx] = len(unique_layers)
            unique_layers.append(layer_idx)
    
    return [layer_to_unique[layer_idx] for layer_idx in use_layers]


if __name__ == "__main__":
    # use_layers = [0, 0, 1, 1, 3, 3]
    use_layers = [1, 1, 2, 2, 3, 3]
    unique_layers, use_indices = map_layer_indices(use_layers)
    print(unique_layers)
    print(use_indices)
    print(get_unique_layers(use_layers))
    print(get_layer_mapping_indices(use_layers))
