import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 


from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CB_recommendtaions import ContentBasedRecommender

"""No switching Yet 
   doen't exclude repeated recs in both 
   doesn't consider type of recs 
   no location based context recommedntaion """

class HybridRecommender:
    def __init__(self, db_client: Neo4jClient, 
                  cf_model: UserBasedCF, threshold_interactions: int = 5):
      
        self.db_client = db_client
        self.cf_model = cf_model
        self.threshold_interactions = threshold_interactions
        
        

    def get_interactions_count(self, user_id):
        if user_id in self.cf_model.user_id_to_index:
            user_idx = self.cf_model.user_id_to_index[user_id]
            if user_idx < self.cf_model.sparse_user_item_matrix.shape[0]:
                return self.cf_model.sparse_user_item_matrix[user_idx].nnz 
        return 0 
    
    def format_output(self, df, user_id,
                       expected_cols=['rank', 'user', 'item', 'item_type', 'name', 'score']):
        

        return None
    
    def _normalize_scores(self,model, df, score_column='score'):
        if df.empty or score_column not in df.columns:
            print(f'Dataset of scores is empty or score column does not exist in model {model}')
        df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0.0)
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        if max_score == min_score:
           df[f'norm_{score_column}'] = 1.0 if max_score > 0 else 0.0
        else:
           scaler = MinMaxScaler()
           df[f'norm_{score_column}'] = scaler.fit_transform(df[[score_column]])

        return df



    def recommend(self, user_id, top_n=10, item_type=None, use_mmr_for_cb=True):
        interactions_count = self.get_interactions_count(user_id)
        print(f"Interaction count from CF matrix: {interactions_count}")

        # --- Stage 1 : Cold Start  (0) -> Use Content-Based ---

        if interactions_count == 0:     
           cbf = ContentBasedRecommender(
           content_fetcher=content_fetcher,
           new_user=True,
           user_id=user_id,
           limit=top_n*5 # why !!?
             )
           recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
           if item_type in recs_df.columns:
               recs_df=  recs_df[recs_df['type'] == item_type].copy()
           else : print(f"Warning: Cannot filter CB recommendations by this type. Column doesn't exist.")
           recs_df['score'] = pd.to_numeric(recs_df['score'], errors='coerce')
           final_recs_df = recs_df.nlargest(top_n, 'score')     
           return final_recs_df
 
 
         # --- Stage 2: Low Interaction  (1 to threshold-1 interactions) ---


        elif interactions_count <self.threshold_interactions :
            cbf = ContentBasedRecommender(
                content_fetcher=self.content_fetcher,
                new_user=False, # Fetcher should get history to build profile
                user_id=user_id,
                limit=top_n * 5
            )
            recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
            if item_type in recs_df.columns:
               recs_df=  recs_df[recs_df['type'] == item_type].copy()
            else : print(f"Warning: Cannot filter CB recommendations by this type. Column doesn't exist.")
            final_recs_df = recs_df.nlargest(top_n, 'score')      
            return final_recs_df
        

        # --- Stage 3: Active User (threshold or more interactions) -> Weighted Hybrid ---


        else :
            weight_cf = min(0.8, interactions_count / (2.0 * self.threshold_interactions)) # can be optimized 
            weight_cb = 1.0 - weight_cf
            cf_recs_df = self.cf_model.recommend(user_id, top_n=(top_n * 3), item_type=None) # Get all types
            cbf = ContentBasedRecommender(
                content_fetcher=content_fetcher,
                new_user=False,
                user_id=user_id,
                limit=top_n * 5
            )
            cb_recs_df = cbf.recommend(top_n=(top_n * 3), use_mmr=use_mmr_for_cb)
            cf_recs_df = self._normalize_scores( 'cf', cf_recs_df, 'score')
            cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')

            merge_cols = ['item', 'item_type'] 

            if not cf_recs_df.empty and not cb_recs_df.empty:
                merged_recs = pd.merge(
                cf_recs_df[merge_cols + ['norm_score']],
                cb_recs_df[merge_cols + ['norm_score']],
                on=merge_cols,
                how='outer',
                suffixes=('_cf', '_cb') )
            elif not cf_recs_df.empty:
                 merged_recs = cf_recs_df[merge_cols + ['norm_score']].rename(columns={'norm_score':'norm_score_cf'})
                 merged_recs['norm_score_cb'] = 0.0 # Add missing column
            elif not cb_recs_df.empty:
                 merged_recs = cb_recs_df[merge_cols + ['norm_score']].rename(columns={'norm_score':'norm_score_cb'})
                 merged_recs['norm_score_cf'] = 0.0 # Add missing column
            else:
                 merged_recs = pd.DataFrame(columns=merge_cols + ['norm_score_cf', 'norm_score_cb'])
            
            merged_recs.fillna({'norm_score_cf': 0, 'norm_score_cb': 0}, inplace=True)     


            if 'norm_score_cf' in merged_recs.columns and 'norm_score_cb' in merged_recs.columns:
                 merged_recs['hybrid_score'] = (weight_cf * merged_recs['norm_score_cf'] +
                                              weight_cb * merged_recs['norm_score_cb'])
            elif 'norm_score_cf' in merged_recs.columns: # Only CF results
                 merged_recs['hybrid_score'] = merged_recs['norm_score_cf']
            elif 'norm_score_cb' in merged_recs.columns: # Only CB results
                 merged_recs['hybrid_score'] = merged_recs['norm_score_cb']
            else:
                 merged_recs['hybrid_score'] = 0.0

            # 5. Filter by item_type (if requested)
            if item_type and 'item_type' in merged_recs.columns:
                print(f"Filtering Hybrid recommendations by type: {item_type}")
                merged_recs = merged_recs[merged_recs['item_type'] == item_type].copy()
            elif item_type:
                 print("Warning [Stage 3]: Cannot filter merged recommendations by type. Column missing.")


            # 6. Sort and Select Top N
            final_recs_df = merged_recs.sort_values('hybrid_score', ascending=False).head(top_n)

            # 7. Format Output
            return final_recs_df

        



if __name__ == "__main__":
   
    user_id_zero_interactions = "user_with_only_onboarding_tags" 
    user_id_few_interactions = "4e7a97b8-189a-4677-9b7e-81a1d92f489e" 
    user_id_active = "4bf0b634-076d-4d9e-9679-b83fdcaabf81" 

    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)

    print("\n--- Fitting CF Model ---")
    cf_knn_model = UserBasedCF(db_client, k_neighbors=10, min_overlap=2, min_sim=0.05)
    try:
        cf_knn_model.fit()
        print("CF Model Fitted.")
    except Exception as e:
        print(f"Error fitting CF model: {e}")
        
        cf_knn_model = None 

    if cf_knn_model:
       
        hybrid_model = HybridRecommender(db_client,cf_knn_model) # Use threshold=5 for example

        # Test Different User Stages
        print(f"\n--- Testing Stage 1 User: {user_id_zero_interactions} ---")
        recs_zero = hybrid_model.recommend(user_id=user_id_zero_interactions, top_n=5, item_type='Destination')
        print(f"Hybrid Recommendations (Stage 1 - CB Only, Destination):")
        print(recs_zero)

        print(f"\n--- Testing Stage 2 User: {user_id_few_interactions} ---")
        recs_few = hybrid_model.recommend(user_id=user_id_few_interactions, top_n=5, item_type='Event')
        print(f"Hybrid Recommendations (Stage 2 - CB Informed, Event):")
        print(recs_few)

        print(f"\n--- Testing Stage 3 User: {user_id_active} ---")
        recs_active = hybrid_model.recommend(user_id=user_id_active, top_n=5, item_type='Trip')
        print(f"Hybrid Recommendations (Stage 3 - Weighted, Trip):")
        print(recs_active)
    else:
        print("\nSkipping Hybrid Recommendations because CF model fitting failed.")


    db_client.close()
    print("\nNeo4j driver closed.")
