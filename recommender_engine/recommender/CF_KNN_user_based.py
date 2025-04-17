import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient
import pandas as pd
from scipy.sparse import coo_matrix
from typing import Optional, Any 

import random 

class UserBasedCF:
    def __init__(self, db_client: Neo4jClient, k_neighbors=5, min_sim=0.1, min_overlap=0):
        self.db_client = db_client
        self.k_neighbors = k_neighbors
        self.min_sim = min_sim
        self.min_overlap = min_overlap
        self.user_similarity = None
        self.user_indices = None
        self.user_item_matrix = None
        self.item_columns = None
        self.item_names_df = None
        self.user_id_to_index = {}

    def fit(self):
        """
        Train the model by computing user-user similarity.
        Fetches interactions data from the database and processes it.
        """
        fetcher = InteractionsFetcher(self.db_client)
        interactions = fetcher.fetch_interactions()
        self.item_names_df = interactions[['item', 'name', 'type']].drop_duplicates(subset=['item', 'type']).set_index(['item', 'type'])

        if interactions.empty:
            print("Warning: Interaction data is empty. Cannot fit the model.")
            return

        

        # Create user-item matrix (users as rows, items as columns)
        try:
            self.user_item_matrix = interactions.pivot_table(index="user", columns=["item", "type"], values="avg", fill_value=0, observed=True)

        except ValueError as e:
            print(f"Error during pivot_table: {e}. Ensure 'user', 'item', 'type', and 'weight' columns exist.")
            return

        if self.user_item_matrix.empty:
            print("Warning: User-item matrix is empty after pivoting. Cannot compute user similarity.")
            return

        # Store user indices
        self.user_indices = self.user_item_matrix.index
        self.user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_indices)}


        self.item_columns = self.user_item_matrix.columns
        self.sparse_user_item_matrix = csr_matrix(self.user_item_matrix.values)
        
        user_overlap_matrix = self.sparse_user_item_matrix.astype(bool) @ self.sparse_user_item_matrix.transpose().astype(bool)
        user_overlap_matrix = user_overlap_matrix.tocsr()
      
        
        raw_similarity = cosine_similarity(self.sparse_user_item_matrix, dense_output=False)
        raw_similarity = raw_similarity.tocsr()
        
        similarity_after_min_sim = raw_similarity.multiply(raw_similarity > self.min_sim)
        overlap_mask = user_overlap_matrix > self.min_overlap
        self.user_similarity = similarity_after_min_sim.multiply(overlap_mask)
     

    def recommend(self, user_id, top_n=5, item_type=None):
      
        if self.user_similarity is None or self.user_indices is None or self.user_item_matrix is None or self.item_columns is None:
            print("Warning: Model has not been fitted yet or fitting resulted in empty data. Cannot make recommendations.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'type', 'name', 'score'])

        if user_id not in self.user_indices:
            print(f"Warning: User ID '{user_id}' not found in the training data.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'type', 'name', 'score'])

        user_idx = np.where(self.user_indices == user_id)[0][0]
        similarity_scores = self.user_similarity.getrow(user_idx).toarray().flatten()
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
            ranked_items_with_scores = ranked_items_with_scores[ranked_items_with_scores.index.get_level_values('type') == item_type]

        top_ranked_items = ranked_items_with_scores.nlargest(top_n)

        recommendations = []
        for rank, (item_id, item_type_rec) in enumerate(top_ranked_items.index, start=1):
            try:
                item_info = self.item_names_df.loc[(item_id, item_type_rec)]
                item_name = item_info['name']
                score = top_ranked_items[(item_id, item_type_rec)]
                recommendations.append({
                    'rank': rank,
                    'item': item_id,
                    'type': item_type_rec,
                    'name': item_name,
                    'score': score
                })
            except KeyError:
                print(f"Warning: Name not found for item ID '{item_id}' and type '{item_type_rec}'.")
                recommendations.append({
                    'rank': rank,
                    'item': item_id,
                    'type': item_type_rec,
                    'name': f"Unknown {item_type_rec} ({item_id})",
                    'score': top_ranked_items[(item_id, item_type_rec)]
                })

 
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df[['rank',  'item', 'type', 'name', 'score']]
    def recommend_with_epsilon_greedy(self, user_id, top_candidates_df: pd.DataFrame, item_type: Optional[str] = None, epsilon: float = 0.2, top_n: int = 10):
        
         if self.user_item_matrix is None or self.item_names_df is None:
             print("Model must be fitted first.")
             return pd.DataFrame()
     
         # Get all items of the given type
         all_items_of_type = self.item_names_df[self.item_names_df.index.get_level_values('type') == item_type]
         
         # Get items user already interacted with
         user_vector = self.user_item_matrix.loc[user_id]
         interacted_items = user_vector[user_vector > 0].index
     
         # Filter unseen items
         unseen_items = [
             (item_id, t) for (item_id, t) in all_items_of_type.index
             if (item_id, t) not in interacted_items
         ]
     
         # Choose explore/exploit split
         explore_count = int(top_n * epsilon)
         exploit_count = top_n - explore_count
     
         exploit_items = top_candidates_df.head(exploit_count).to_dict("records")
     
         explore_samples = random.sample(unseen_items, min(explore_count, len(unseen_items)))
         
         explore_items = []
         for rank, (item_id, item_type_val) in enumerate(explore_samples, start=1):
             try:
                 name = self.item_names_df.loc[(item_id, item_type_val)]['name']
             except KeyError:
                 name = f"Unknown {item_type_val} ({item_id})"
             explore_items.append({
                 'rank': None,
                 'item': item_id,
                 'type': item_type_val,
                 'name': name,
                 'score': None  # score is unknown
             })
     
         # Merge & shuffle
         all_items = exploit_items + explore_items
         random.shuffle(all_items)
     
         # Re-rank and return
         for i, item in enumerate(all_items, 1):
             item['rank'] = i
         return pd.DataFrame(all_items)

  

if __name__ == "__main__":
    db_client = Neo4jClient()
    model = UserBasedCF(db_client, k_neighbors=25, min_sim=0.001, min_overlap=0)
    model.fit()

    user_id_to_recommend = [
        '633af53b-f78c-474c-9324-2a734bd86d24',
        '65ab857a-6ff4-493f-aa8d-ddde6463cc20',
        '72effc5b-589a-4076-9be5-f7c3d8533f70',
        '8aaafb9e-0f60-47d1-9b98-1b171564fbf9',
        'b9c32bc3-4b7f-46fd-af3b-ca48060b89a1',
        '3738e035-45a5-4b8b-86a2-32ff64a76f03',
        '82f642dc-fda0-46ed-b080-f4b1866899a6',
        'b99c49fc-f7b1-4cd4-8d22-cc5b8575f07f',
        '3989ed58-1cce-45e2-9b5b-e4827165e324'
    ]

    for user_id in user_id_to_recommend:
        for item_type in ['Trip', 'Event', 'Destination']:
            top_items = model.recommend(user_id, top_n=10, item_type=item_type)
            print(f"Top {item_type} recommendations for user {user_id}:")
            print(top_items[['score','name']])

            epsilon_recs = model.recommend_with_epsilon_greedy(
                user_id,
                top_candidates_df=top_items,
                item_type=item_type,
                epsilon=0.2
            )
            print(f"Epsilon-Greedy {item_type} recommendations for user {user_id}:")
            print(epsilon_recs[['score','name']])

    db_client.close()