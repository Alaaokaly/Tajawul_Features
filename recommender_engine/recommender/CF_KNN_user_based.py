import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient
import pandas as pd
from scipy.sparse import coo_matrix
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

        

        # Create user-item matrix (users as rows, items as columns)
        try:
            self.user_item_matrix = interactions.pivot_table(index="user", columns=["item", "item_type"], values="avg", fill_value=0, observed=True)

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
        coo = coo_matrix(( interactions['avg'].astype(float),
                          (interactions['user'].cat.codes.copy(), 
                          interactions['item'].cat.codes.copy())))
        overlap_matrix = coo.astype(bool).astype(int).dot(coo.transpose().astype(bool).astype(int))
        self.user_similarity = cosine_similarity(coo, dense_output=False)
        self.user_similarity =  self.user_similarity.multiply( self.user_similarity > self.min_sim)
        self.user_similarity  = self.user_similarity.multiply(overlap_matrix > self.min_overlap)
     

    def recommend(self, user_id, top_n=5, item_type=None):
      
        if self.user_similarity is None or self.user_indices is None or self.user_item_matrix is None or self.item_columns is None:
            print("Warning: Model has not been fitted yet or fitting resulted in empty data. Cannot make recommendations.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'item_type', 'name', 'score'])

        if user_id not in self.user_indices:
            print(f"Warning: User ID '{user_id}' not found in the training data.")
            return pd.DataFrame(columns=['rank', 'user', 'item', 'item_type', 'name', 'score'])

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
                    'item': item_id,
                    'item_type': item_type_rec,
                    'name': item_name,
                    'score': score
                })
            except KeyError:
                print(f"Warning: Name not found for item ID '{item_id}' and type '{item_type_rec}'.")
                recommendations.append({
                    'rank': rank,
                    'item': item_id,
                    'item_type': item_type_rec,
                    'name': f"Unknown {item_type_rec} ({item_id})",
                    'score': top_ranked_items[(item_id, item_type_rec)]
                })

 
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df[['rank',  'item', 'item_type', 'name', 'score']]

if __name__ == "__main__":
    db_client = Neo4jClient()
    model = UserBasedCF(db_client, k_neighbors=5, min_sim=0.001, min_overlap=0)
    model.fit()
    user_id_to_recommend =[
                              '633af53b-f78c-474c-9324-2a734bd86d24',
                               '65ab857a-6ff4-493f-aa8d-ddde6463cc20',
                             

                               '72effc5b-589a-4076-9be5-f7c3d8533f70',
                               '8aaafb9e-0f60-47d1-9b98-1b171564fbf9',
                           
                            
                             
                               
                               '841f7b4f-215d-472b-91f2-7241b64']                              
    # Get recommendations for 'Trip'
    for id in user_id_to_recommend:
        top_trips = model.recommend(id, top_n=3, item_type='Trip')
        print(f"Top Trip recommendations for user {id}:")
        print(top_trips)
    
        # Get recommendations for 'Event'
        top_events = model.recommend(id, top_n=3, item_type='Event')
        print(f"Top Event recommendations for user {id}:")
        print(top_events)
    
        # Get recommendations for 'Destination'
        top_destinations = model.recommend(id, top_n=3, item_type='Destination')
        print(f"Top Destination recommendations for user {id}:")
        print(top_destinations)

    db_client.close()