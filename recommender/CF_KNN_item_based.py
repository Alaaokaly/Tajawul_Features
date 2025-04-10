import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from recommender.fetch_data import fetch_interactions
"""
add the pseudo matrix based on the mini threshold of overlap and interactions
"""
class ItemBasedCFRecommender:
    def __init__(self, k=10):
        self.k = k  # Number of similar items 
        self.min_sim = None 
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_indices = None
        self.min_overlap = None

    def fit(self, interactions):
        """
        Train the model by computing item-item similarity.
        :param interactions: Pandas DataFrame with (user, item, rating)
        """
    
        self.user_item_matrix = interactions.pivot_table(index="user", columns=["item", "item_type"], values="weight", fill_value=0)
        self.item_indices = self.user_item_matrix.columns
        sparse_matrix = csr_matrix(self.user_item_matrix.T)
        overlap_matrix = sparse_matrix.astype(bool).astype(int).dot(sparse_matrix.transpose().astype(bool).astype(int))
        self.item_similarity = cosine_similarity(sparse_matrix)
        # self.item_similarity = self.item_similarity.multiply(overlap_matrix > self.min_overlap)
        # self.item_similarity = self.item_similarity.multiply(self.item_similarity > self.min_sim)
        return self.item_similarity

    
    def recommend(self, user_id, top_n=5):
        """
        Generate item recommendations for a given user.
        :param user_id: ID of the target user.
        :param top_n: Number of recommendations to return.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id].values
        scores = user_ratings @ self.item_similarity  

        # Ignore already interacted items
        scores[user_ratings > 0] = 0

        
        top_items = np.argsort(scores)[::-1][:top_n]
        return [self.item_indices[i] for i in top_items]

def fetch_recommendations(recs):
    query = """ MATCH (n:) where n.id """
data = fetch_interactions()

model = ItemBasedCFRecommender()
model.fit(data)

print(model.recommend(user_id="99ae6489-05d2-49df-bb62-490a2a3f707b" , top_n=3))

if __name__ == "__main__":
    data = fetch_interactions()
    model = ItemBasedCFRecommender()
    model.fit()
    recs = model.recommend(user_id="99ae6489-05d2-49df-bb62-490a2a3f707b")
    recs = fetch_recommendations(recs)
    print("the recommendations:" , RecursionError)