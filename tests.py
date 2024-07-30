import numpy as np
import pytest
import uuid
from Recommender import Recommender

@pytest.fixture
def setup_recommender():
    # Setup for a Recommender instance with minimal setup data
    # SampleData is just a few lines of our actual data
    return Recommender(data_path = 'data/SampleData.csv')

def test_initialization(setup_recommender):
    """
    Test that the Recommender initializes correctly with the expected attributes.
    """
    rec = setup_recommender
    assert isinstance(rec.k_hop_recs, np.ndarray)
    assert isinstance(rec.k_hop_recs, np.ndarray)
    assert isinstance(rec.creator_brand, np.ndarray)
    assert isinstance(rec.creator_index, dict)
    assert isinstance(rec.brand_index, dict)
    assert isinstance(rec.index_creator[0], str)
    assert isinstance(rec.index_brand[0], str)
    assert (setup_recommender.ESAE_recs.shape == setup_recommender.k_hop_recs.shape)

def test_binarize_matrix(setup_recommender):
    """
    Test the binarize matrix function.
    """
    matrix = np.array([[0, 0.5, 0], [0, 0, 3]])
    expected = np.array([[0, 1, 0], [0, 0, 1]])
    binarized = setup_recommender._binarize_matrix(matrix)
    np.testing.assert_array_equal(binarized, expected)

def test_stratify_matrix(setup_recommender):
    """
    Test the stratify matrix function.
    """
    matrix = np.array([0.05, 0, 0.2, 0.45])
    thresholds = (0.1, 0.4)
    values = (0.0, 0.5, 1, 1.5)
    expected = np.array([0.5, 0, 1, 1.5])
    stratified = setup_recommender._stratify_matrix(matrix, thresholds, values)
    np.testing.assert_array_equal(stratified, expected)


def test_add_data_new_users_and_shops(setup_recommender):
    """
    Test adding data with new users and shops.
    """
    initial_len_users = len(setup_recommender.creator_index)
    initial_len_shops = len(setup_recommender.brand_index)
    num_users, num_shops = setup_recommender.ESAE_recs.shape
    # Generate unique user and shop IDs
    unique_user_id = f"user_{uuid.uuid4()}"
    unique_shop_id = f"shop_{uuid.uuid4()}"

    # Add unique data
    new_data = [(unique_user_id, unique_shop_id, 0.8)]
    setup_recommender.add_data(new_data)
    #Check that our indexes have been properly updated
    assert unique_user_id in setup_recommender.creator_index
    assert unique_shop_id in setup_recommender.brand_index
    assert len(setup_recommender.creator_index) == initial_len_users + 1
    assert len(setup_recommender.brand_index) == initial_len_shops + 1
    assert setup_recommender.recompute_needed
    assert setup_recommender.save_needed
    setup_recommender.update()
    assert setup_recommender.ESAE_recs.shape[0] == num_users + 1
    assert setup_recommender.item_item.shape[1] == num_shops + 1
    new_recommender = Recommender(data_path = 'data/SampleData.csv')
    new_len_users = len(new_recommender.creator_index)
    assert new_len_users == initial_len_users + 1
