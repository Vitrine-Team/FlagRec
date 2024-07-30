import csv
from collections import defaultdict
import numpy as np
from typing import Tuple, Dict

def csv_to_pivot(
    filename: str, creator_col: str = 'vitrineShopId', brand_col: str = 'supplierShopId',
    score_col: str = 'recommendationScore') -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Converts a CSV file to a pivot table represented as a NumPy array.

    Args:
        filename (str): The path to the CSV file.
        creator_col (str): Column name for the creator IDs.
        brand_col (str): Column name for the brand IDs.
        score_col (str): Column name for the recommendation scores.

    Returns:
        tuple: A tuple containing the pivot table as a NumPy array,
               and two dictionaries mapping creators and brands to array indices.
    """
    # Creating a nested dictionary to hold the pivot data
    pivot_dict = defaultdict(lambda: defaultdict(float))
    # Map creator and brand ids to respective rows and columns in table
    creator_map = {}
    brand_map = {}

    try:
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    creator = row[creator_col]
                    brand = row[brand_col]
                    score = row[score_col]
                    pivot_dict[creator][brand] = float(score) if score else 0.0
                except ValueError:
                    continue  # or handle/log error

                if creator not in creator_map:
                    creator_map[creator] = len(creator_map)
                if brand not in brand_map:
                    brand_map[brand] = len(brand_map)

        # Convert the nested dictionary to a NumPy array
        num_creators, num_brands = len(creator_map), len(brand_map)
        # Unknown scores are treated as zeros
        pivot_array = np.zeros((num_creators, num_brands), dtype=float)

        for creator, inner_dict in pivot_dict.items():
            row_idx = creator_map[creator]
            for brand, score in inner_dict.items():
                col_idx = brand_map[brand]
                pivot_array[row_idx][col_idx] = score

        return pivot_array, creator_map, brand_map
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def pivot_to_csv(
    pivot_array: np.ndarray, creator_map: Dict[str, int],
    brand_map: Dict[str, int], output_filename: str) -> None:
    """
    Converts a pivot table (NumPy array) and its associated mappings back to a CSV file.

    Args:
        pivot_array (np.ndarray): The NumPy array containing the pivot table data.
        creator_map (Dict[str, int]): A dictionary mapping creator IDs to row indices.
        brand_map (Dict[str, int]): A dictionary mapping brand IDs to column indices.
        output_filename (str): The filename for the output CSV file.
    """
    # Invert the maps to get indices to IDs
    creator_index_to_id = {v: k for k, v in creator_map.items()}
    brand_index_to_id = {v: k for k, v in brand_map.items()}

    # Open the CSV file for writing
    with open(output_filename, mode='w', newline='') as csvfile:
        fieldnames = ['supplierShopId', 'vitrineShopId', 'recommendationScore']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Iterate through the array and write each non-zero value
        for i in range(pivot_array.shape[0]):
            for j in range(pivot_array.shape[1]):
                score = pivot_array[i, j]
                if score != 0:  # Optionally write only non-zero scores
                    writer.writerow({
                        'vitrineShopId': creator_index_to_id[i],
                        'supplierShopId': brand_index_to_id[j],
                        'recommendationScore': score
                    })
