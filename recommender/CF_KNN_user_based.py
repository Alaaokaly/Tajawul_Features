import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient
import pandas as pd

class UserBasedCF:
    def __init__(self, db_client: Neo4jClient, k_neighbors=5):
        self.db_client = db_client
        self.k_neighbors = k_neighbors
        self.user_similarity = None
        self.user_indices = None
        self.user_item_matrix = None
        self.item_columns = None
        self.item_names_df = None

    def fit(self):
        """
        Train the model by computing user-user similarity.
        Fetches interactions data from the database and processes it.
        """
        fetcher = InteractionsFetcher(self.db_client)
        interactions = fetcher.fetch_interactions()
        self.item_names_df = interactions[['item', 'name', 'item_type']].drop_duplicates(subset=['item', 'item_type']).set_index(['item', 'item_type'])

        if interactions.empty:
            print("Warning: Interaction data is empty. Cannot fit the model.")
            return

        # Fill NaN weights with a default value (e.g., 1.0) before pivoting
        interactions['weight'] = interactions['weight'].fillna(1.0)

        # Create user-item matrix (users as rows, items as columns)
        try:
            self.user_item_matrix = interactions.pivot_table(index="user", columns=["item", "item_type"],
                                                              values="weight", fill_value=0)
        except ValueError as e:
            print(f"Error during pivot_table: {e}. Ensure 'user', 'item', 'item_type', and 'weight' columns exist.")
            return

        if self.user_item_matrix.empty:
            print("Warning: User-item matrix is empty after pivoting. Cannot compute user similarity.")
            return

        # Store user indices
        self.user_indices = self.user_item_matrix.index

        # Store item columns for recommendation output
        self.item_columns = self.user_item_matrix.columns

        # Convert to sparse matrix for efficiency
        sparse_matrix = csr_matrix(self.user_item_matrix.values)

        # Compute user-user similarity using cosine similarity
        self.user_similarity = cosine_similarity(sparse_matrix)

    def recommend(self, user_id, top_n=5, item_type=None):
        """
        Generate item recommendations for a given user based on similar users (KNN).
        :param user_id: ID of the target user.
        :param top_n: Number of recommendations to return.
        :param item_type: The type of item to recommend ('Trip', 'Event', 'Destination').
                          If None, recommends across all item types.
        :return: A pandas DataFrame with 'rank', 'user', 'item', 'item_type', 'name', and 'score'
                 for the top N recommendations.
        """
        if self.user_similarity is None or self.user_indices is None or self.user_item_matrix is None or self.item_columns is None:
            print("Warning: Model has not been fitted yet or fitting resulted in empty data. Cannot make recommendations.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'item_type', 'name', 'score'])

        if user_id not in self.user_indices:
            print(f"Warning: User ID '{user_id}' not found in the training data.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'item_type', 'name', 'score'])

        user_idx = np.where(self.user_indices == user_id)[0][0]
        similarity_scores = self.user_similarity[user_idx]
        similar_user_indices = np.argsort(similarity_scores)[::-1][1:self.k_neighbors + 1]
        similar_user_weights = similarity_scores[similar_user_indices]
        item_scores = pd.Series(0, index=self.item_columns)

        for i, neighbor_idx in enumerate(similar_user_indices):
            neighbor_id = self.user_indices[neighbor_idx]
            neighbor_ratings = self.user_item_matrix.loc[neighbor_id]
            item_scores += neighbor_ratings * similar_user_weights[i]

        user_interactions = self.user_item_matrix.loc[user_id]
        interacted_items = user_interactions[user_interactions > 0].index.tolist()

        ranked_items_with_scores = item_scores.drop([(item, it) for item, it in interacted_items if (item, it) in item_scores.index])

        if item_type:
            ranked_items_with_scores = ranked_items_with_scores[ranked_items_with_scores.index.get_level_values('item_type') == item_type]

        top_ranked_items = ranked_items_with_scores.nlargest(top_n)

        recommendations = []
        for rank, (item_id, item_type_rec) in enumerate(top_ranked_items.index, start=1):
            try:
                item_info = self.item_names_df.loc[(item_id, item_type_rec)]
                item_name = item_info['name']
                score = top_ranked_items[(item_id, item_type_rec)]
                recommendations.append({
                    'rank': rank,
                    'user': user_id,
                    'item': item_id,
                    'item_type': item_type_rec,
                    'name': item_name,
                    'score': score
                })
            except KeyError:
                print(f"Warning: Name not found for item ID '{item_id}' and type '{item_type_rec}'.")
                recommendations.append({
                    'rank': rank,
                    'user': user_id,
                    'item': item_id,
                    'item_type': item_type_rec,
                    'name': f"Unknown {item_type_rec} ({item_id})",
                    'score': top_ranked_items[(item_id, item_type_rec)]
                })

        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df[['rank', 'user', 'item', 'item_type', 'name', 'score']]

if __name__ == "__main__":
    db_client = Neo4jClient()
    model = UserBasedCF(db_client, k_neighbors=5)
    model.fit()
    user_id_to_recommend = "4bf0b634-076d-4d9e-9679-b83fdcaabf81"

    # Get recommendations for 'Trip'
    top_trips = model.recommend(user_id_to_recommend, top_n=3, item_type='Trip')
    print(f"Top Trip recommendations for user {user_id_to_recommend}:")
    print(top_trips)

    # Get recommendations for 'Event'
    top_events = model.recommend(user_id_to_recommend, top_n=3, item_type='Event')
    print(f"Top Event recommendations for user {user_id_to_recommend}:")
    print(top_events)

    # Get recommendations for 'Destination'
    top_destinations = model.recommend(user_id_to_recommend, top_n=3, item_type='Destination')
    print(f"Top Destination recommendations for user {user_id_to_recommend}:")
    print(top_destinations)

    db_client.close()