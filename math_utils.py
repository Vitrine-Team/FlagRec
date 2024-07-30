import numpy as np

def random_projection(matrix: np.ndarray, n_components: int, projection: np.ndarray = None) -> np.ndarray:
    """
    Performs a Gaussian random projection to reduce the dimensionality of the data.

    Args:
    matrix (ndarray): A numpy array where each row represents a sample and each column represents a feature.
    n_components (int): The number of dimensions to project the data into.

    Returns:
    ndarray: The dimensionally reduced data.
    """
    if not projection:
        projection = np.random.randn(matrix.shape[1], n_components)
    #Scaling factor according to Johnson Lindenstrauss Lemma
    scaling_factor = 1 / np.sqrt(n_components)
    return (matrix @ projection) * scaling_factor, projection

def cosine_sim(matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity matrix from a given matrix where each row represents a vector.

    Args:
    matrix (ndarray): A numpy array where each row represents a vector.

    Returns:
    ndarray: A square matrix where element (i, j) represents the cosine similarity between row i and row j.
    """
    # Compute the dot product between each pair of rows
    dot_product = np.dot(matrix, matrix.T)

    # Compute the norm (magnitude) of each row
    norm = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero by replacing zeros with a very small number
    norm[norm == 0] = 1e-10

    # Compute the outer product of the norms
    norm_product = np.outer(norm, norm)

    # Compute the cosine similarity
    similarity = dot_product / norm_product

    return similarity
