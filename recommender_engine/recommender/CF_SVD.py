 

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient




def preprocess_data(df):
    """Convert interactions into a user-item matrix."""
    
    
    pivot_table = df.pivot_table(index="user", columns=["item", "item_type",'name'], values="weight", fill_value=0)
    return pivot_table

def apply_svd(data, k=50):
    matrix = data.to_numpy(dtype=np.float64)  
    k = min(k, min(matrix.shape) - 1)
    U, sigma, Vt = svds(matrix, k=k)
    sigma = np.diag(sigma)
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return pd.DataFrame(predicted_ratings, index=data.index, columns=data.columns)


def get_top_n_recommendations(user_id, predictions, item_type, df, n=5):
    """Get top-N recommended trips or destinations for a user."""
    item_ids = df[df["item_type"] == item_type]["item"].unique()
    sorted_items = predictions.loc[user_id].sort_values(ascending=False)
    return sorted_items[sorted_items.index.get_level_values("item").isin(item_ids)].head(n)



def find_similar_users(user_id, matrix, top_n=10):
    """Find top-N most similar users using cosine similarity."""
    similarity_matrix = cosine_similarity(matrix)
    user_idx = list(matrix.index).index(user_id)
    similarity_scores = list(enumerate(similarity_matrix[user_idx]))
    sorted_users = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_users = [matrix.index[i] for i, _ in sorted_users[1:top_n+1]]  # Exclude self
    return similar_users

db_client = Neo4jClient()
fetch = InteractionsFetcher(db_client)
df = fetch.fetch_interactions()
user_item_matrix = preprocess_data(df)
predicted_ratings = apply_svd(user_item_matrix)



user_id = "4bf0b634-076d-4d9e-9679-b83fdcaabf81"
print("Top 10 similar users:", find_similar_users(user_id, user_item_matrix, top_n=10))
print("Recommended Trips:", get_top_n_recommendations(user_id, predicted_ratings, "Trip", df, n=5))
print("Recommended Destinations:", get_top_n_recommendations(user_id, predicted_ratings, "Destination", df, n=5))
print("Recommended Trips:", get_top_n_recommendations(user_id, predicted_ratings, "Event", df, n=5))
