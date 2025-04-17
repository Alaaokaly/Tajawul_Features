import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient
import pandas as pd
from scipy.sparse import coo_matrix
from typing import Optional, Any 
import random 

class ItemBasedCF:
    def __init__(self, db_client: Neo4jClient, k_neighbors=5, min_sim=0.1, min_overlap=0):
        self.db_client = db_client
        self.k_neighbors = k_neighbors
        self.min_sim = min_sim
        self.min_overlap = min_overlap
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_columns = None
        self.user_indices = None
        self.item_names_df = None

    def fit(self):
        fetcher = InteractionsFetcher(self.db_client)
        interactions = fetcher.fetch_interactions()
        self.item_names_df = interactions[['item', 'name', 'type']].drop_duplicates(subset=['item', 'type']).set_index(['item', 'type'])

        if interactions.empty:
            print("Warning: Interaction data is empty. Cannot fit the model.")
            return

        try:
            self.user_item_matrix = interactions.pivot_table(index="user", columns=["item", "type"], values="avg", fill_value=0, observed=True)
        except ValueError as e:
            print(f"Error during pivot_table: {e}")
            return

        if self.user_item_matrix.empty:
            print("Warning: User-item matrix is empty after pivoting.")
            return

        self.user_indices = self.user_item_matrix.index
        self.item_columns = self.user_item_matrix.columns

        matrix = self.user_item_matrix.T  # Transpose to make items as rows
        item_sparse = csr_matrix(matrix.values)

       
        similarity = cosine_similarity(item_sparse, dense_output=False)
        similarity = similarity.multiply(similarity > self.min_sim)

 
        binarized = item_sparse.copy()
        binarized.data = np.ones_like(binarized.data)
        overlap = binarized.dot(binarized.T)
        self.item_similarity = similarity.multiply(overlap > self.min_overlap)

    def recommend(self, user_id, top_n=5, item_type=None):
        if self.item_similarity is None or self.user_item_matrix is None:
            print("Model not fitted properly.")
            return pd.DataFrame(columns=['rank', 'item', 'type', 'name', 'score'])

        if user_id not in self.user_indices:
            print(f"User ID '{user_id}' not found.")
            return pd.DataFrame(columns=['rank', 'item', 'type', 'name', 'score'])

        user_vector = self.user_item_matrix.loc[user_id]
        scores = pd.Series(0, index=self.item_columns, dtype=float)

        for item_idx, rating in user_vector[user_vector > 0].items():
            item_index = self.item_columns.get_loc(item_idx)
            similar_items_scores = self.item_similarity[item_index].toarray().flatten()
            scores += rating * pd.Series(similar_items_scores, index=self.item_columns)

        # Filter already interacted
        interacted_items = user_vector[user_vector > 0].index
        scores = scores.drop(interacted_items, errors='ignore')

        if item_type:
            scores = scores[scores.index.get_level_values('type') == item_type]

        top_items = scores.nlargest(top_n)
        recommendations = []

        for rank, (item_id, item_type_rec) in enumerate(top_items.index, start=1):
            try:
                item_info = self.item_names_df.loc[(item_id, item_type_rec)]
                name = item_info['name']
            except KeyError:
                name = f"Unknown {item_type_rec} ({item_id})"
            recommendations.append({
                'rank': rank,
                'item': item_id,
                'type': item_type_rec,
                'name': name,
                'score': top_items[(item_id, item_type_rec)]
            })

        return pd.DataFrame(recommendations)[['rank', 'item', 'type', 'name', 'score']]
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
    model = ItemBasedCF(db_client, k_neighbors=25, min_sim=0.001, min_overlap=0)
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
