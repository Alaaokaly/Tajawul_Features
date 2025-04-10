import numpy as np
import pandas as pd
 
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CB_recommendtaions import ContentBasedRecommender



class HybridRecommender:
    def __init__(self, cb_model, cf_model, threshold_interactions=5):
      
        self.cb_model = cb_model
        self.cf_model = cf_model
        self.threshold_interactions = threshold_interactions

    def recommend(self, user_id, top_n=5):
       
       # no pseudo matrix yet 
       # No switching Yet 
       # doen't exclude repeated recs in both 
       # doesn't consider type of recs 
       # no location based context recommedntaion 


        cb_recs = self.cb_model.recommend(top_n= top_n)
        interaction_count = self.cf_model.user_item_matrix.loc[user_id].sum() if user_id in self.cf_model.user_indices else 0

        if interaction_count < self.threshold_interactions:
            return cb_recs  
        cf_recs = self.cf_model.recommend(user_id, top_n=top_n)
        weight_cf = min(1, interaction_count / (2 * self.threshold_interactions))  ## Increasing CF weight over time ## the 2 can change 
        weight_cb = 1 - weight_cf

        hybrid_scores = {}
        for i, rec in enumerate(cb_recs):
            hybrid_scores[rec] = hybrid_scores.get(rec, 0) + weight_cb * (top_n - i)

        for i, rec in enumerate(cf_recs):
            hybrid_scores[rec] = hybrid_scores.get(rec, 0) + weight_cf * (top_n - i)
        final_recs = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_n]
        return final_recs


if __name__ == "__main__":
    user_id="99ae6489-05d2-49df-bb62-490a2a3f707b"
    db_client = Neo4jClient()
    
    content_fetcher = ContentBasedFetcher(db_client)
    cbf = ContentBasedRecommender(
        content_fetcher,
        new_user=True,
        user_id=user_id,
        limit=10
    )
    cf_knn_model = UserBasedCF(db_client, k_neighbors=5)
    cf_knn_model.fit()

    hybrid_model = HybridRecommender(cbf, cf_knn_model)
    recs = hybrid_model.recommend(user_id=user_id)
    print("Hybrid Recommendations:", recs)
    
    user_id =  "4bf0b634-076d-4d9e-9679-b83fdcaabf81"

    
    recommendations = hybrid_model.recommend(user_id = user_id, top_n=5)
    print("Hybrid Recommendations:", recommendations)
