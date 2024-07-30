import numpy as np

def ndcg_at_k(original_vec: np.ndarray, recovered_vec: np.ndarray, relevant_items: set, kept_items: set, k: int):
    """
    Calculate the normalized discounted cumulative gain (NDCG) at K.

    Parameters:
    - original_vec (np.ndarray): Original vector of relevance scores.
    - recovered_vec (np.ndarray): Predicted vector of relevance scores.
    - relevant_items (set): Set of indices that are relevant.
    - kept_items (set): Set of indices that should not be predicted.
    - k (int): The number of top predictions to consider.

    Returns:
    - float: The NDCG value at K.
    """
    filtered_vec = recovered_vec.copy()
    filtered_vec[list(kept_items)] = -np.inf #do not predict non-masked items

    top_k_items = np.argsort(-filtered_vec)[:k]

    dcg = 0
    for i, item in enumerate(top_k_items):
        dcg += original_vec[item] / np.log2(i + 2) #get true relevance score of our topk preds

    original_just_masked = original_vec.copy()
    original_just_masked[list(kept_items)] = 0 # don't include kept scores when computing ideal DCG

    top_k_ideal = np.argsort(-original_just_masked)[:k]

    idcg = 0
    for i, item in enumerate(top_k_ideal):
        idcg += original_just_masked[item] / np.log2(i+2)

    ndcg = dcg/idcg

    return ndcg

def recall_at_k(recovered_vec: np.ndarray, relevant_items: set, kept_items:set, k: int):
    """
    Calculate recall at K for a set of predictions.

    Parameters:
    - recovered_vec (np.ndarray): The vector from which predictions are derived.
    - relevant_items (set): The set of items that are relevant to the user.
    - kept_items (set): The set of items that should not be considered for predictions.
    - k (int): The number of top predictions to consider.

    Returns:
    - float: The recall value at K.
    """
    # Exclude non-masked (kept) items from predictions
    filtered_vec = np.copy(recovered_vec)
    filtered_vec[list(kept_items)] = -np.inf

    # Get top K predictions
    top_k_items = set(np.argsort(-filtered_vec)[:k])

    # Exclude non-masked (kept) items from recall metric
    relevant_masked = relevant_items - kept_items

    # Calculate recall only for relevant items
    relevant_predictions = top_k_items.intersection(relevant_masked)
    recall = len(relevant_predictions) / len(relevant_masked) if relevant_masked else 0.0

    return recall

def precision_at_k(recovered_vec: np.ndarray, relevant_items: set, kept_items: set, k: int):
    """
    Calculate precision at K for a set of predictions.

    Parameters:
    - recovered_vec (np.ndarray): The vector from which predictions are derived.
    - relevant_items (set): The set of items that are relevant.
    - kept_items (set): The set of items that should not be considered for predictions.
    - k (int): The number of top predictions to consider.

    Returns:
    - float: The precision value at K.
    """
    # Exclude non-masked (kept) items from predictions
    filtered_vec = np.copy(recovered_vec)
    filtered_vec[list(kept_items)] = -np.inf

    # Get top K predictions
    top_k_items = set(np.argsort(-filtered_vec)[:k])

    # Calculate precision only for relevant items
    relevant_predictions = top_k_items.intersection(relevant_items)
    precision = len(relevant_predictions) / k if k > 0 else 0.0

    return precision
