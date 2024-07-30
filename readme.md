# Recommender System

This repository contains the `Recommender` class implemented in Python, which is designed to provide personalized recommendations by ensembling 2 methods - k-hop cosine similarity and Embarrassingly Shallow Autoencoders (ESAE) - https://arxiv.org/abs/1905.03375. The system operates on a creator-brand interaction matrix and offers functionalities to calculate recommendations, add new data to the dataset, update the models to fit the new data, and metrics and sanity checks to ensure the rec system is functioning properly.

## Features

- **Data Handling**: Load and manipulate creator-supplier score matrices from CSV files.
- **Recommendation Models**: Implements k-hop and ESAE recommendation algorithms.
- **Metrics Calculation**: Includes functions to calculate NDCG, precision, and recall at different ranks.
- **Dynamic Updates**: Ability to add new user-brand interactions and update recommendation models accordingly.
- **Performance Evaluation**: Evaluate the recommender system's performance with metrics such as NDCG, precision, and recall, including methods to handle data sparsity and masking.

## Usage

1. **Initialization**:
   Create an instance of the `Recommender` class by specifying the path to your creator-supplier score matrix CSV file. Currently defaults to 'Creator Supplier Score.csv' which contains the current 15,000 user-brand edges.
   ```python
    engine = Recommender(data_path='path_to_your_data.csv')
    ```
2. **Model Training**
    Train the recommendation models using the data provided.
    ```python
     engine._learn_rec_engine()
     ```

3. **Adding Data**
    Add new interaction data as a list of tuples to the recommender system.
    ```python
    engine.add_data([(vitrine_id, supplier_id, score)]) #adds one edge to the dataset
    ```

4. **Generating Recommendations**
    Two ways to get scores from the rec engine. Generally, use get_score which will return a dictionary of supplied ids and respective scores (only for scores above a certain threshold).
    ```python
    engine.get_scores(user_id)
    ```
    We can also generate scores from a 'feature vector' for a user, which can be useful for testing purposed.
    ```python
    user_preferences = [('probioticmaker.myshopify.com', 1), ('lifestacksnutrition.myshopify.com', 0.05)]
    engine.get_scores_from_features(data = user_preferences)
    ```
5. **Updating the System**
    Two main updates need to be made when data is added.  The data needs to be saved, and the rec engines need to be updated. Calling update(self) will do both when new data has been added and should be called periodically by the server.
    ```python
    engine.update()
    ```

6. **Testing and validation**
    The repo supports a few ways to test the system. When making updates to the code base, a good place to start is with the pytests which will validate basic class functionality.
    ```shell
    pytest tests.py
    ```

    The other main testing function which test the behavior of the rec system are sanity_check(), and score_report().  The sanity check leverages get_scores_from_features() to show the behavior over a few mock user inputs, while the score report will give a metric-driven (precision, recall, ndcg) assessment of the model.  Both of these methods can help validate how the model's performance changes as new data is added to the system.
    ```python
    print(engine.score_report()) #recommend printing the output string to get the desired formatting
    print(engine.sanity_check())
    ```


## Design Decisions
- **Minimal Imports**: Only non-standard python libraries used are numpy and pytest.  This approach avoids the need to keep large packages like scipy, pandas, and sklearn on the server and ensure ease of setup.
- **Choice of rec systems**: Recommendations are generated with Embarrassingly Shallow Autoencoders, and k-hop cosine similarity, which were the highest performing item-item and user-user methods from masking experiments.  These approaches had the best precision, recall, and normalized cumulative discounted gain, metrics which are our best proxy for the recommendation task.
- **Inference**: The most updated recommendations are saved when self.update() or self._learn_rec_engine() are called.  This means get_scores(user) is little more than a lookup, allowing the class to respond to requests quickly.
- **Scaling the system**: Both systems rely on matrix methods. The computation is pretty negligible when the adjacency matrix is hundreds by hundreds but will not be feasible for a lightweight server when the scale gets to tens of thousands by tens of thousands. In order to avoid this failure mode the class enables down-projection when the features get past a certain threshold (default 2500) which allows the system to scale into the tens of thousands. Theoretically the system could fail when the down-projection operation no longer fits into memory (n > 50000) - at which point other solutions should be explored (downsampling, neural methods, more compute with better memory chunking enabled, etc.)
- **Output Scores**: The rec system combines the known scores with the inferred scores for its final prediction. However, I would recommend just taking the rec system scores and existing scores and sampling independently. I also recommend generating a top-k ranking and doing some fixed sampling procedure based on the top-k ordering rather than viewing the scores as probabilities and doing probabilistic sampling.
