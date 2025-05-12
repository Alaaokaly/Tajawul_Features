import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import random
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF  
from CB_recommendations import ContentBasedRecommender

# to be done 

"""
   -- 
   -- 
   -- no location based context recommedntaion 
   --
   -- If The Engine Gives no Recs : 
                             - fill with the most popular 
                             - items with highest averages
                             -  baseline recommendations """

class HybridRecommender:
    def __init__(self, db_client: Neo4jClient, 
                 user_cf_model: UserBasedCF, 
                 item_cf_model: ItemBasedCF,
                 threshold_interactions: int = 1 # threshold should be equal to the min_overlap in CF
                ):
        self.db_client = db_client
        self.user_cf_model = user_cf_model
        self.item_cf_model = item_cf_model
        self.threshold_interactions = threshold_interactions
        self.content_fetcher = ContentBasedFetcher(db_client)
        
      
    def get_interactions_count(self, user_id):
        if user_id in self.user_cf_model.user_id_to_index:
            user_idx = self.user_cf_model.user_id_to_index[user_id]
            if user_idx < self.user_cf_model.sparse_user_item_matrix.shape[0]:
                return self.user_cf_model.sparse_user_item_matrix[user_idx].nnz 
        return 0 
    
    def _normalize_scores(self, model, df, score_column='score'):
        # Check if dataframe is empty or if the score column doesn't exist
        if df.empty or score_column not in df.columns:
            print(f'Dataset of scores is empty or score column does not exist in model {model}')
            return pd.DataFrame()  # Return an empty dataframe if conditions aren't met
        
        # Ensure the 'score' column is numeric, filling non-numeric values with 0
        df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0.0)
        
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        
        # If all scores are the same, normalize to 1.0 if positive or 0.0 if not
        if max_score == min_score:
            df[f'norm_{score_column}'] = 1.0 if max_score > 0 else 0.0
        else:
            scaler = MinMaxScaler()
            try:
                df[f'norm_{score_column}'] = scaler.fit_transform(df[[score_column]])
            except ValueError:
                print(f"Error in normalization. Returning unmodified dataframe for model {model}.")
                return df  # If normalization fails, return the dataframe as-is
    
        return df

    def _select_cf_model(self, item_type):
        """
        Select the appropriate CF model based on item type:
        - ItemBasedCF for Destinations and Events
        - UserBasedCF for Trips and Posts 
        """
        if item_type in ['Destination', 'Event']:
            return self.item_cf_model
        else:  # 'Trip' or None or any other type
            return self.user_cf_model

    def recommend(self, user_id, top_n=10, item_type=None, use_mmr_for_cb=True, epsilon=0.1):
        interactions_count = self.get_interactions_count(user_id)
        print(f"Interaction count from CF matrix: {interactions_count}")
        
        # Select appropriate CF model based on item type
        cf_model = self._select_cf_model(item_type)
        
        # --- Stage 1 : Cold Start  (0) -> Use Content-Based ---
        if interactions_count == 0:
            cbf = ContentBasedRecommender(
                content_fetcher=self.content_fetcher,
                new_user=True,
                user_id=user_id,
                limit=top_n * 5
            )
            recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
            if 'type' in recs_df.columns and item_type is not None:
                recs_df = recs_df[recs_df['type'] == item_type].copy()
            else:
                print(f"Warning: Cannot filter CB recommendations by this type. Column doesn't exist.")
            recs_df['score'] = pd.to_numeric(recs_df['score'], errors='coerce')
            recs_df = self._normalize_scores('cb', recs_df, 'score')
            
            if recs_df.empty:
                return pd.DataFrame()  # Return empty if no valid recommendations
    
            final_df = recs_df.nlargest(top_n, 'norm_score')
            return final_df
    
        # --- Stage 2: Low Interaction  (1 to threshold-1 interactions) ---
        elif interactions_count < self.threshold_interactions:
            # For low interaction users, blend Content-Based with some CF recommendations
            # but with a heavier weight on Content-Based
            
            # Get Content-Based recommendations
            cbf = ContentBasedRecommender(
                content_fetcher=self.content_fetcher,
                new_user=False,
                user_id=user_id,
                limit=top_n * 5
            )
            cb_recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
            if 'type' in cb_recs_df.columns and item_type is not None:
                cb_recs_df = cb_recs_df[cb_recs_df['type'] == item_type].copy()
            
            # Try to get some CF recommendations (may be sparse)
            try:
                cf_recs_df = cf_model.recommend(user_id, top_n=(top_n * 2), item_type=item_type)
                if not cf_recs_df.empty:
                    # Normalize and blend with a low weight for CF
                    weight_cf = 0.2  # Low weight for CF due to sparse interactions
                    weight_cb = 0.8  # High weight for Content-Based
                    
                    cf_recs_df = self._normalize_scores('cf', cf_recs_df, 'score')
                    cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')
                    
                    # Blend recommendations
                    return self._blend_recommendations(cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n)
            except Exception as e:
                print(f"Error getting CF recommendations for low-interaction user: {e}")
            
            # If CF failed or returned empty, just use CB
            if cb_recs_df.empty:
                return pd.DataFrame()
                
            final_df = cb_recs_df.nlargest(top_n, 'score')
            return final_df
    
        # --- Stage 3: Active User (threshold or more interactions) -> Weighted Hybrid ---
        else:
            weight_cf = min(0.8, interactions_count / (2.0 * self.threshold_interactions))
            weight_cb = 1.0 - weight_cf
    
            cf_recs_df = cf_model.recommend(user_id, top_n=(top_n * 3), item_type=item_type)
            cbf = ContentBasedRecommender(
                content_fetcher=self.content_fetcher,
                new_user=False,
                user_id=user_id,
                limit=top_n * 5
            )
            cb_recs_df = cbf.recommend(top_n=(top_n * 3), use_mmr=use_mmr_for_cb)
            
            # Filter content-based recommendations by item type if specified
            if 'type' in cb_recs_df.columns and item_type is not None:
                cb_recs_df = cb_recs_df[cb_recs_df['type'] == item_type].copy()
    
            # Normalize scores
            cf_recs_df = self._normalize_scores('cf', cf_recs_df, 'score')
            cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')
            
            # Blend recommendations
            return self._blend_recommendations(cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n)

    def _blend_recommendations(self, cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n):
       
    
        # Check if 'item' exists in cb_recs_df, if not, try to map from 'name' or another identifier
        if 'item' not in cb_recs_df.columns and 'name' in cb_recs_df.columns:
            # Create an item column based on name (or you could use another ID if available)
            cb_recs_df = cb_recs_df.copy()
            cb_recs_df['item'] = cb_recs_df['name']
        
        # Make sure 'type' exists, otherwise add a default
        if 'type' not in cb_recs_df.columns:
            cb_recs_df['type'] = 'Unknown'
        
        # Prepare for merge operation
        merge_cols = ['item', 'type']
        
        # Handle empty dataframes
        if cf_recs_df.empty and cb_recs_df.empty:
            return pd.DataFrame()
        elif cf_recs_df.empty:
            merged_recs = cb_recs_df[merge_cols + ['norm_score']].rename(columns={'norm_score': 'norm_score_cb'})
            merged_recs['norm_score_cf'] = 0.0
        elif cb_recs_df.empty:
            merged_recs = cf_recs_df[merge_cols + ['norm_score']].rename(columns={'norm_score': 'norm_score_cf'})
            merged_recs['norm_score_cb'] = 0.0
        else:
            # Both dataframes have data, perform the merge
            merged_recs = pd.merge(
                cf_recs_df[merge_cols + ['norm_score']],
                cb_recs_df[merge_cols + ['norm_score']],
                on=merge_cols,
                how='outer',
                suffixes=('_cf', '_cb')
            )
            
            # Fill NA values with 0
            merged_recs['norm_score_cf'] = merged_recs.get('norm_score_cf', 0).fillna(0)
            merged_recs['norm_score_cb'] = merged_recs.get('norm_score_cb', 0).fillna(0)
            
            # Calculate weighted hybrid score
            merged_recs['hybrid_score'] = (weight_cf * merged_recs['norm_score_cf']) + (weight_cb * merged_recs['norm_score_cb'])
            
            # Get top N recommendations
            top_hybrid_recs = merged_recs.nlargest(top_n, 'hybrid_score')
            
            # Add additional information like name
            if not cf_recs_df.empty and 'name' in cf_recs_df.columns:
                name_map = dict(zip(cf_recs_df[['item', 'type']].apply(tuple, axis=1), cf_recs_df['name']))
                top_hybrid_recs['name'] = top_hybrid_recs[['item', 'type']].apply(lambda x: name_map.get(tuple(x), "Unknown"), axis=1)
            elif not cb_recs_df.empty and 'name' in cb_recs_df.columns:
                name_map = dict(zip(cb_recs_df[['item', 'type']].apply(tuple, axis=1), cb_recs_df['name']))
                top_hybrid_recs['name'] = top_hybrid_recs[['item', 'type']].apply(lambda x: name_map.get(tuple(x), "Unknown"), axis=1)
            
            # Add rank
            top_hybrid_recs['rank'] = range(1, len(top_hybrid_recs) + 1)
            
            # Rename the hybrid score to score for consistency
            top_hybrid_recs = top_hybrid_recs.rename(columns={'hybrid_score': 'score'})
            
            # Select and order the columns
            result_columns = ['rank', 'item', 'type', 'score']
            if 'name' in top_hybrid_recs.columns:
                result_columns.append('name')
                
            return top_hybrid_recs[result_columns]

    def recommend_with_epsilon_greedy(self, user_id, top_n=10, item_type=None, use_mmr_for_cb=True, epsilon=0.2):
        
        recommendations = self.recommend(user_id, top_n=top_n, item_type=item_type, use_mmr_for_cb=use_mmr_for_cb)
        
        if recommendations.empty:
            return pd.DataFrame()
        
     
        cf_model = self._select_cf_model(item_type)
        
        
        try:
            epsilon_greedy_recs = cf_model.recommend_with_epsilon_greedy(
                user_id=user_id,
                top_candidates_df=recommendations,
                item_type=item_type,
                epsilon=epsilon,
                top_n=top_n
            )
            return epsilon_greedy_recs
        except Exception as e:
            print(f"Error applying epsilon-greedy approach: {e}")
            return recommendations  # Fall back to regular recommendations
   

if __name__ == "__main__":
   
    user_id_few_interactions = "b9c32bc3-4b7f-46fd-af3b-ca48060b89a1" 
    user_id_active =  '65ab857a-6ff4-493f-aa8d-ddde6463cc20'

    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)

    print("\n--- Fitting CF Models ---")
    # Initialize UserBasedCF model
    user_cf_model = UserBasedCF(db_client, k_neighbors=10, min_sim=0.05)
    # Initialize ItemBasedCF model
    item_cf_model = ItemBasedCF(db_client, k_neighbors=15, min_overlap=1, min_sim=0.05)
    
    try:
        user_cf_model.fit()
        print("User-based CF Model Fitted.")
        item_cf_model.fit()
        print("Item-based CF Model Fitted.")
        cf_models_fitted = True
    except Exception as e:
        print(f"Error fitting CF models: {e}")
        cf_models_fitted = False

    if cf_models_fitted:
        hybrid_model = HybridRecommender(
            db_client=db_client,
            user_cf_model=user_cf_model,
            item_cf_model=item_cf_model,
            threshold_interactions=5
        )
        user_id_zero_interactions = "72308a60-c755-47df-9bd5-1a75c51886ad"
        # Test Different User Stages with different item types
        print(f"\n--- Testing Stage 1 User: {user_id_zero_interactions} ---")
        
        # Test with Destination (should use item-based CF model for active users)
        recs_destination = hybrid_model.recommend(
            user_id=user_id_zero_interactions, 
            top_n=5, 
            item_type='Destination'
        )
        print(f"Hybrid Recommendations (Stage 1 - Cold Start, Destination):")
        print(recs_destination)
        
        #Test with Event (should also use item-based CF model for active users)
        recs_event = hybrid_model.recommend(
            user_id=user_id_few_interactions, 
            top_n=5, 
            item_type='Event'
        )
        print(f"\nHybrid Recommendations (Stage 2 - Few Interactions, Event):")
        print(recs_event)
        
        #Test with Trip (should use user-based CF model for active users)
        recs_trip = hybrid_model.recommend(
            user_id=user_id_active, 
            top_n=5, 
            item_type='Trip'
        )
        print(f"\nHybrid Recommendations (Stage 3 - Active User, Trip):")
        recs_trip = hybrid_model.recommend(
            user_id=user_id_active, 
            top_n=5, 
            item_type='Post'
        )
        print(f"\nHybrid Recommendations (Stage 3 - Active User, POst):")

        print(recs_trip[['score','name']])

        recs_des = hybrid_model.recommend(
            user_id=user_id_active, 
            top_n=5, 
            item_type='Destination'
        )
    
        top_users = hybrid_model.user_cf_model.recommend_users_to_user(user_id_active,top_n = 2)
      
        print(f"Users ::  {top_users[['user_id','similarity']]}")
              
        print(f"\nHybrid Recommendations (Stage 3 - Active User, Destinations):")

        print(recs_des[['score','name']])
        
        # Test epsilon-greedy recommendation approach
        print("\n--- Testing Epsilon-Greedy Recommendations ---")
        epsilon_recs = hybrid_model.recommend_with_epsilon_greedy(
            user_id=user_id_active,
            top_n=5,
            item_type='Destination',
            epsilon=0.2
        )
        print(f"Epsilon-Greedy Recommendations (Destination):")
        print(epsilon_recs)
        
    else:
        print("\nSkipping Hybrid Recommendations because CF model fitting failed.")

    db_client.close()
    print("\nNeo4j driver closed.")