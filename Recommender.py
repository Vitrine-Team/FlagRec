import numpy as np
from math_utils import cosine_sim, random_projection
from utils import csv_to_pivot, pivot_to_csv
from metrics import ndcg_at_k, precision_at_k, recall_at_k
from typing import Tuple, Dict, List

class Recommender:
    def __init__(self, data_path: str = 'data/Creator Supplier Score Matrix.csv'):
        """
        Initializes the Recommender class with two learned matrices: k_hop and ESAE.
        """

        self.k_hop_recs = None  # Initialize the k-hop recs
        self.ESAE_recs = None # Initialize Embarrassingly Shallow Autoencoder recommendations
        self.item_item = None   # Initialize the Embarrassingly Shallow Autoencoder matrix

        self.alpha, self.beta = 0.05, 0.2
        self.k = 8 # hyperparam k for k hop recs
        self.lambda_reg = 150 # hyperparam needed for ESAE model
        self.cutoff = 0.001 # cutoff where we treat a score as zero

        self.max_features = 2500 # threshold when we start adopting dimensionality reduction
        self.downproj = None # downprojection matrix needed only when feature dim > max_features

        #file for reading and writing our data
        self.data_path = data_path
        #read in locally stored data
        # creator-brand adjacency score matrix, dict of user_id to row in matrix, dict of shop_id to col in matrix
        self.creator_brand, self.creator_index, self.brand_index = csv_to_pivot(filename = self.data_path)

        #flag for if new data has been entered since we've last saved data to our local csv
        self.save_needed = False

        #also useful to map indices back to ids
        self.index_creator = {value: key for key, value in self.creator_index.items()}
        self.index_brand = {value: key for key, value in self.brand_index.items()}

        self._learn_rec_engine()
        #flag for if new data has been entered since we've last learned the rec engine
        self.recompute_needed = False

    def _binarize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Binarize matrix so that non-zeros are 1 and zeros remain zeros.

        Parameters:
        - matrix (np.ndarray): Input dense matrix.

        Returns:
        - np.ndarray: Binarized dense matrix
        """
        binary_matrix = (matrix > 0).astype(int)  # Converts True/False to 1/0
        return binary_matrix

    def _stratify_matrix(
        self, matrix: np.ndarray,
        thresholds: Tuple[int, int]=(0.1, 0.40),
        values: Tuple[int, int]=(0, 0.05, 1, 1.5) ) -> np.ndarray:
        """
        Stratify the values in a dense matrix based on the given thresholds and corresponding values.

        Parameters:
        - matrix (np.ndarray): Input dense matrix.
        - thresholds (tuple of float): Threshold values for stratification.
        - values (tuple of float): Corresponding values for stratified levels.

        Returns:
        - np.ndarray: Stratified dense matrix.
        """
        stratified_matrix = matrix.copy()

        conditions = [stratified_matrix == 0,
            (0 < stratified_matrix) & (stratified_matrix < thresholds[0]),
            (stratified_matrix >= thresholds[0]) & (stratified_matrix < thresholds[1]),
            stratified_matrix >= thresholds[1]
        ]
        stratified_matrix = np.select(conditions, values)

        return stratified_matrix

    def _row_scale(self, matrix):
        """
        Scales each row of the matrix by its maximum value.

        Args:
        - matrix (np.ndarray): The input matrix.

        Returns:
        - np.ndarray: The scaled matrix where each row has been divided by its maximum value.
        """

        # Get the maximum value of each row. `keepdims=True` makes sure the division broadcasts correctly.
        row_max = np.max(matrix, axis=1, keepdims=True)

        # Avoid division by zero by replacing zero maxima with 1 (or any non-zero number).
        row_max[row_max == 0] = 1

        # Scale each row by its maximum value.
        scaled_matrix = matrix / row_max

        return scaled_matrix

    def _learn_k_hop(self, data: np.ndarray) -> None:
        """
        Cache cosine sim k-hop matrix as output predictios.
        Saves outputs to class variables.

        Args:
        data (np.ndarray): User - item implicit interaction matrix.
        """
        #thresholding scores to more accurately describe the meaning of our observed scores
        interactions = self._stratify_matrix(matrix = data)

        if data.shape[1] > self.max_features:
            features, _ = random_projection(matrix = interactions, n_components = self.max_features)
        else:
            features = interactions

        # Calculate the user-user similarity matrix
        user_similarity = cosine_sim(matrix = features)

        # set self-similarity to 0
        dia_indices = np.diag_indices(user_similarity.shape[0])
        user_similarity[dia_indices] = 0

        # calculate the k-th order similarity with matrix power.
        k_hop_sim = user_similarity ** self.k


        # save the recommendations
        k_hop_recs = k_hop_sim @ interactions
        self.k_hop_recs = self._row_scale(k_hop_recs)


    def _learn_ESAE(self, data: np.ndarray) -> None:
        """
        Learn the ESAE matrix and cache predictions from the given data.
        Saves outputs to class variables.

        Args:
        data (np.ndarray): The data to learn from.
        """
        interactions = self._stratify_matrix(matrix = data)

        # preserve number of items, and random project users because ESAE is an item-item approach
        if data.shape[1] > self.max_features:
            features, self.downproj = random_projection(interactions.T, self.max_features).T
        else:
            features = interactions

        G = features.T @ features #compute item-item gram matrix
        dia_indices = np.diag_indices(G.shape[0])
        G[dia_indices] += self.lambda_reg #regularization term added

        # Compute the inverse of the regularized Gram matrix
        P = np.linalg.inv(G)    # Compute the inverse of the regularized Gram matrix
        # Compute the weight matrix B
        B = P / (-np.diag(P)[:, None])
        # Set self-similarity to 0
        B[dia_indices] = 0

        self.item_item = B
        ESAE_recs = interactions @ B
        self.ESAE_recs = self._row_scale(ESAE_recs)

    def _learn_rec_engine(self) -> None:
        """
        Learns the k_hop recommendations, ESAE recommendations, and ESAE item-item matrix.
        """
        self._learn_k_hop(data = self.creator_brand)
        self._learn_ESAE(data = self.creator_brand)
        #Note that the rec engine is up to date with our data
        self.recompute_needed = False


    def _combine_scores(self, known_scores: np.ndarray, pred_scores: np.ndarray) -> np.ndarray:
        """
        Combine known and predicted scores for final rec scores.
        Final prediction at indices with initialized scores -> (1 - alpha) * known + alpha * pred
        Final prediction at indices without initialized scores -> beta * pred

        Args:
        known_scores (np.ndarray): User-creator scores calculated outside of rec engine
        pred_scores (np.ndarray): User-creator scores produced by rec engine

        Returns:
        np.ndarray: Final prediction array of creator-brand scores.
        """
        # Get the indices of the non-zero elements in known_scores
        non_zero_indices = (known_scores != 0)

        # Get the max of known_scores at the non-zero indices
        known_max = known_scores[non_zero_indices].max()

        # Get the max of predicted_scores at the non-zero indices
        pred_max = pred_scores[non_zero_indices].max()

        # Scale predicted_scores to match the maximums of known_scores at the non-zero indices
        scaled_pred_scores = np.copy(pred_scores)

        if pred_max > 0:  # Prevent division by zero
            scaling_factor = known_max / pred_max
            scaled_pred_scores *= scaling_factor

        # Combine scores at the non-zero indices using the weighted average
        combined_scores = np.zeros_like(pred_scores)
        combined_scores = (self.alpha * scaled_pred_scores) + ((1 - self.alpha) * known_scores)

        # At the zero indices in known_scores, scale down predicted_scores by discount factor beta
        combined_scores[~non_zero_indices] = self.beta * scaled_pred_scores[~non_zero_indices]

        return combined_scores

    def _rec_scores(self, user_id: str):
        """
        Calculate and return brand scores for a given user (just the recommendations).

        Args:
        user (np.ndarray): The user for whom to calculate brand scores.

        Returns:
        np.ndarray: An array of creator-brand scores.
        """
        assert user_id in self.creator_index, 'User ID not found'

        ind = self.creator_index[user_id]

        #Ensemble the cached rec scores
        ave_recs = (self.k_hop_recs[ind] + self.ESAE_recs[ind]) / 2

        return ave_recs



    def add_data(self, data: List[Tuple[str, str, float]]) -> None:
        """
        Adds data to the class.

        Args:
            data (List[Tuple[str, str, float]]): A list of tuples, where each tuple contains
                a vitrine ID, a supplier shop ID, and a score.
        """
        # Note that a retrain of the rec engine and a resave of our data is needed.
        self.recompute_needed = True
        self.save_needed = True

        for user_id, shop_id, score in data:

            assert 0 <= score <= 1, 'Creator-brand score not in valid rand [0, 1]'

            # if user/creator is new, add them to our index and a blank row to our adjacency matrix
            if user_id not in self.creator_index:
                ind = len(self.creator_index) # assign new index
                self.creator_index[user_id] = ind
                self.index_creator[ind] = user_id
                #add blank row to adjacency matrix
                zeros_row = np.zeros((1, self.creator_brand.shape[1]))
                self.creator_brand = np.concatenate((self.creator_brand, zeros_row), axis = 0)

            # if shop/brand is new, add them to our index and a blank column to our adjacency matrix
            if shop_id not in self.brand_index:
                ind = len(self.brand_index)
                self.brand_index[shop_id] = ind
                self.index_brand[ind] = shop_id
                #add blank column to adjacency matrix
                zeros_col = np.zeros((self.creator_brand.shape[0], 1))
                self.creator_brand = np.concatenate((self.creator_brand, zeros_col), axis = 1)

            user_ind, brand_ind = self.creator_index[user_id], self.brand_index[shop_id]
            self.creator_brand[user_ind, brand_ind] = score


    def save_data(self) -> None:
        """
        Save updated data to the self.data_path file
        """
        #call to util function
        pivot_to_csv(self.creator_brand, self.creator_index, self.brand_index, self.data_path)
        # note that we have saved all data in our structs
        self.save_needed = False

    def update(self) -> None:
        """
        Function for chron job to call every so often - if we have new data since last update -
        save our data to data file and recompute the rec engine.
        """
        if self.save_needed:
            self.save_data()
        if self.recompute_needed:
            self._learn_rec_engine()


    def get_scores_from_features(self, data: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        When the user is not yet represented in the rec engine we can still get recommendations from ESAE.
        Just use ESAE because ESAE inference for single user is lightweight while k-hop inference for novel user
        is a bit more expensive.

        Args:
        data (List[Tuple[str, float]]): for a user - known brand affinity scores.

        Returns:
        Dict[str, float]: A dictionary of brand IDs to their respective scores, excluding scores
                          approximately equal to zero based on a defined cutoff.
        """
        # cutoff index for last brand since we've updated the item_item matrix.
        max_index = self.item_item.shape[0]
        brand_scores = np.zeros(max_index)

        # build user-brand feature vector in alignment with our item-item matrix
        for brand_id, score in data:
            if brand_id in self.brand_index and self.brand_index[brand_id] <= max_index:
                cur_ind = self.brand_index[brand_id]
                brand_scores[cur_ind] = score

        # ESAE algorithm works better with binary-like input data.
        stratified_scores = self._stratify_matrix(matrix = brand_scores)

        if self.ESAE_recs.shape[1] > self.max_features:
            features, _ = random_projection(
                            matrix = stratified_scores.T,
                            n_components = self.max_features,
                            projection = self.downproj).T
        else:
            features = stratified_scores

        # get ESAE predicted scores by multiplying cur user scores with our learned item_item matrix
        pred_scores = features @ self.item_item

        final_scores = self._combine_scores(known_scores = brand_scores, pred_scores = pred_scores)

        output_brand_scores = {}

        for i, score in enumerate(final_scores):
            # don't report ≈ 0 scores
            if score < self.cutoff:
                continue
            brand_id = self.index_brand[i]
            output_brand_scores[brand_id] = score

        # put back the scores for brands not in our adjacency
        for brand_id, score in data:

            if brand_id not in output_brand_scores:
                output_brand_scores[brand_id] = score

        return output_brand_scores

    def top_k_known_and_recs(self, user_id: str, k = 8) -> Tuple[List[str],List[str]]:
        """
        For testing purposes - get known top k brands and recommended top k brands.

        Args:
        user_id (string)
        """
        user_ind = self.creator_index[user_id]

        top_brand_inds = np.argsort(-self.creator_brand[user_ind])[:k]

        just_preds = self.k_hop_recs[user_ind] + self.ESAE_recs[user_ind] - (100 * self.creator_brand[user_ind])

        top_pred_inds = np.argsort(-just_preds)[:k]

        known = [self.index_brand[ind] for ind in top_brand_inds]
        predicted = [self.index_brand[ind] for ind in top_pred_inds]
        return known, predicted

    def get_top_k(self, scores: Dict[str, float], k: int = 8 ) -> List[str]:
        """
        Get top k brands from a brand-score dictionary for a user/creator.

        Args:
        scores (di)
        """
        # sort items scores in descending order
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        #get the top k items
        top_k_keys = [item[0] for item in sorted_items[:k]]
        return top_k_keys

    def sanity_check(self):
        """
        Sanity check on simple hypothetical user profiles to illustrate basic functionality (or lack thereof).
        Please consult with sanity check when making changes to hyperparams!
        """
        test_feats = [
            [('nakano-knives-usa.myshopify.com', 0.5)],
            [('beccobags.myshopify.com', 0.23)],
            [('sandiegocoffee.myshopify.com', 0.01), ('sundream-coffee.myshopify.com', 0.21), ('wildland-coffee.myshopify.com', 0.43)],
            [('probioticmaker.myshopify.com', 1), ('lifestacksnutrition.myshopify.com', 0.05)],
            [('cozys.myshopify.com', 0.1), ('the-lola-blanket.myshopify.com', 0.25), ('ecuadane.myshopify.com', 0.11)]
        ]
        output_string = "Sanity Check Results:\n"
        for index, feats in enumerate(test_feats):
            input_brands = [pair[0] for pair in feats]
            scores = self.get_scores_from_features(data=feats)
            output_brands = self.get_top_k(scores=scores)
            output_string += f"Test Profile {index + 1}:\n"
            output_string += f"  Input Brands: {', '.join(input_brands)}\n"
            output_string += f"  Output Predictions: {output_brands}\n\n"
        return output_string

    def score_report(self):
        """
        Evaluates the recommendation engine's performance by calculating NDCG, precision,
        and recall metrics at k=5 and k=20. The function masks non-zero values in the
        `creator_brand` data with a 25% chance before retraining the recommendation engine
        and generating predictions. The report summarizes the average metrics for these
        predictions against the original unmasked data, highlighting the system's effectiveness.

        Returns:
        - str: A report containing the average metrics (NDCG, precision, recall) at specified
               k values, indicating the recommendation engine's performance.
        """
        unmasked = self.creator_brand.copy()

        # Mask non-zero values with some probability (e.g., 20% chance to mask)
        prob_mask = 0.25
        random_mask = np.random.rand(*unmasked.shape) < prob_mask
        masked = np.where(random_mask & (unmasked != 0), 0, unmasked)

        self.creator_brand = masked
        self._learn_rec_engine() # Learn recommendation engine on masked data
        reconstructed = (self.ESAE_recs + self.k_hop_recs) / 2

        # Prepare to calculate metrics at k = 5 and k = 20
        ks = [5, 20]
        recalls = {k: [] for k in ks}
        precisions = {k: [] for k in ks}
        ncdgs = {k: [] for k in ks}

        for i in range(reconstructed.shape[0]):
            row = reconstructed[i]
            relevant_set = set(np.where(unmasked[i] != 0)[0])  # Indices where original items were kept
            kept_items = set(np.where(masked[i] != 0)[0]) # Indices where items were kept after masking

            #if no items were kept, learning task is degenerate, ignore
            if len(kept_items) == 0:
                continue

            #if no items masked, ignore
            if len(relevant_set) == len(kept_items):
                continue

            for k in ks:
                # Prepare values for metrics
                original_vec = unmasked[i]
                recovered_vec = row

                #call metrics
                ndcg = ndcg_at_k(original_vec, recovered_vec, relevant_set, kept_items, k)
                precision = precision_at_k(recovered_vec, relevant_set, kept_items, k)
                recall = recall_at_k(recovered_vec, relevant_set, kept_items, k)

                recalls[k].append(recall)
                precisions[k].append(precision)
                ncdgs[k].append(ndcg)

        # Calculate mean for each metric
        mean_recalls = {k: np.mean(recalls[k]) for k in ks}
        mean_precisions = {k: np.mean(precisions[k]) for k in ks}
        mean_ncdgs = {k: np.mean(ncdgs[k]) for k in ks}

        # Format an output string that reports metrics @ 5, 20
        report = "Score Report:\n"
        for k in ks:
            report += f"Metrics @ {k}:\n"
            report += f"  NDCG: {mean_ncdgs[k]:.4f}\n"
            report += f"  Precision: {mean_precisions[k]:.4f}\n"
            report += f"  Recall: {mean_recalls[k]:.4f}\n\n"

        #reset system
        self.creator_brand = unmasked
        self._learn_rec_engine()

        return report

    def reconstruction_mse_loss(self):
        """
        Calculate the Mean Squared Error (MSE) between the predicted scores and the actual interaction scores.

        Returns:
        float: The MSE reconstruction loss.
        """
        if self.k_hop_recs is None or self.ESAE_recs is None:
            raise ValueError("Recommender models have not been trained yet.")

        # Assuming the final prediction is some combination of k_hop_recs and ESAE_recs
        # For simplicity, we might average them, or use any other method deemed appropriate
        final_predictions = (self.k_hop_recs + self.ESAE_recs) / 2

        # Calculate MSE only over the non-zero entries of the creator_brand matrix
        mask = self.creator_brand > 0
        mse = np.mean((self.creator_brand[mask] - final_predictions[mask]) ** 2)
        return round(mse,4)

    def get_scores(self, user_id: str) -> Dict[str, float]:
        """
        Calculate and return brand scores for a given user (including original scores).

        Args:
        user_id (string): Id for user for whom to calculate brand scores.

        Returns:
        Dict[str, float]: Map of brand ids to final predicted score.
        """
        assert user_id in self.creator_index, 'User ID not found'

        # Get established user_brand vector
        known_scores = self.creator_brand[self.creator_index[user_id]]

        # Get rec engine outputs
        pred_scores = self._rec_scores(user_id)

        final_scores = self._combine_scores(known_scores = known_scores, pred_scores = pred_scores)

        brand_scores = {} #Map brand ids to their predictions
        for i, score in enumerate(final_scores):
            # don't report ≈ 0 scores
            if score < self.cutoff:
                continue
            brand_id = self.index_brand[i]
            brand_scores[brand_id] = score

        return brand_scores
