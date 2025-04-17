

# to be done 

"""
   -- #----Explore/Exploit-----#
   -- doen't exclude repeated recs in both 
   -- no location based context recommedntaion 
   -- time effect on the system : how old woill be the data that we will keep ?
   -- If The Engine Gives no Recs : 
                             - fill with the most popular 
                             - items with highest averages
                             -  baseline recommendations """

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF
from CB_recommendations import ContentBasedRecommender

class HybridRecommender:
    def __init__(self, db_client: Neo4jClient, 
                 user_cf_model: UserBasedCF, 
                 item_cf_model: ItemBasedCF,
                 threshold_interactions: int = 5):
        self.db_client = db_client
        self.user_cf_model = user_cf_model
        self.item_cf_model = item_cf_model
        self.threshold_interactions = threshold_interactions
        self.content_fetcher = ContentBasedFetcher(db_client)
        
    def get_interactions_count(self, user_id):
        """Get the number of interactions for a user."""
        if user_id in self.user_cf_model.user_id_to_index:
            user_idx = self.user_cf_model.user_id_to_index[user_id]
            if user_idx < self.user_cf_model.sparse_user_item_matrix.shape[0]:
                return self.user_cf_model.sparse_user_item_matrix[user_idx].nnz 
        return 0 
    
    def _normalize_scores(self, model_name, df, score_column='score'):
        """Normalize scores to range [0,1]."""
        # Return empty DataFrame if input is empty or missing score column
        if df is None or df.empty or score_column not in df.columns:
            print(f'Dataset of scores is empty or score column does not exist in model {model_name}')
            return pd.DataFrame(columns=[score_column, f'norm_{score_column}'])
        
        # Convert scores to numeric values
        df = df.copy()  # Create a copy to avoid modifying the original
        df[score_column] = pd.to_numeric(df[score_column], errors='coerce').fillna(0.0)
        
        min_score = df[score_column].min()
        max_score = df[score_column].max()
        
        # Handle case where all scores are identical
        if max_score == min_score:
            df[f'norm_{score_column}'] = 1.0 if max_score > 0 else 0.0
        else:
            # Apply min-max scaling
            df[f'norm_{score_column}'] = (df[score_column] - min_score) / (max_score - min_score)
    
        return df

    def _select_cf_model(self, item_type):
        """Select the appropriate CF model based on item type."""
        if item_type in ['Destination', 'Event']:
            return self.item_cf_model
        else:  # 'Trip' or None or any other type
            return self.user_cf_model

    def recommend(self, user_id, top_n=10, item_type=None, use_mmr_for_cb=True, epsilon=0.1):
        """Get hybrid recommendations for a user."""
        # Get interaction count for user
        interactions_count = self.get_interactions_count(user_id)
        print(f"Interaction count from CF matrix: {interactions_count}")
        
        # Select CF model based on item type
        cf_model = self._select_cf_model(item_type)
        
        # --- Stage 1: Cold Start (0 interactions) -> Use Content-Based only ---
        if interactions_count == 0:
            return self._get_cold_start_recommendations(user_id, top_n, item_type, use_mmr_for_cb)
            
        # --- Stage 2: Low Interaction (1 to threshold-1 interactions) ---
        elif interactions_count < self.threshold_interactions:
            return self._get_low_interaction_recommendations(user_id, cf_model, top_n, item_type, use_mmr_for_cb)
    
        # --- Stage 3: Active User (threshold or more interactions) -> Weighted Hybrid ---
        else:
            return self._get_active_user_recommendations(user_id, cf_model, top_n, item_type, use_mmr_for_cb)

    def _get_cold_start_recommendations(self, user_id, top_n, item_type, use_mmr_for_cb):
        """Get recommendations for new users with no interactions."""
        # Create Content-Based recommender
        cbf = ContentBasedRecommender(
            content_fetcher=self.content_fetcher,
            new_user=True,
            user_id=user_id,
            limit=top_n * 5
        )
        
        # Get recommendations
        recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
        
        # Filter by item type if needed
        if not recs_df.empty and 'type' in recs_df.columns and item_type is not None:
            recs_df = recs_df[recs_df['type'] == item_type].copy()
        
        # Normalize scores
        recs_df = self._normalize_scores('cb', recs_df, 'score')
        
        # Return empty DataFrame if no recommendations
        if recs_df.empty:
            return pd.DataFrame()
        
        # Add rank column
        final_df = recs_df.nlargest(top_n, 'norm_score')
        final_df['rank'] = range(1, len(final_df) + 1)
        
        # Select and order columns
        result_columns = ['rank', 'item', 'type', 'score']
        if 'name' in final_df.columns:
            result_columns.append('name')
            
        return final_df[result_columns] if set(result_columns).issubset(final_df.columns) else final_df

    def _get_low_interaction_recommendations(self, user_id, cf_model, top_n, item_type, use_mmr_for_cb):
        """Get recommendations for users with few interactions."""
        # Get Content-Based recommendations
        cbf = ContentBasedRecommender(
            content_fetcher=self.content_fetcher,
            new_user=False,
            user_id=user_id,
            limit=top_n * 5
        )
        cb_recs_df = cbf.recommend(top_n=(top_n * 2), use_mmr=use_mmr_for_cb)
        
        # Filter by item type if needed
        if not cb_recs_df.empty and 'type' in cb_recs_df.columns and item_type is not None:
            cb_recs_df = cb_recs_df[cb_recs_df['type'] == item_type].copy()
        
        # Try to get CF recommendations
        cf_recs_df = pd.DataFrame()
        try:
            cf_recs_df = cf_model.recommend(user_id, top_n=(top_n * 2), item_type=item_type)
        except Exception as e:
            print(f"Error getting CF recommendations for user {user_id}: {e}")
        
        # If we have both types of recommendations, blend them
        if not cf_recs_df.empty and not cb_recs_df.empty:
            # Low weight for CF due to sparse interactions
            weight_cf = 0.2
            weight_cb = 0.8
            
            # Normalize scores
            cf_recs_df = self._normalize_scores('cf', cf_recs_df, 'score')
            cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')
            
            # Blend recommendations
            return self._blend_recommendations(cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n)
        
        # If CF failed, just use CB
        if not cb_recs_df.empty:
            # Normalize scores and add rank
            cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')
            final_df = cb_recs_df.nlargest(top_n, 'score')
            final_df['rank'] = range(1, len(final_df) + 1)
            
            # Select and order columns
            result_columns = ['rank', 'item', 'type', 'score']
            if 'name' in final_df.columns:
                result_columns.append('name')
                
            return final_df[result_columns] if set(result_columns).issubset(final_df.columns) else final_df
        
        # No recommendations
        return pd.DataFrame()

    def _get_active_user_recommendations(self, user_id, cf_model, top_n, item_type, use_mmr_for_cb):
        """Get recommendations for active users."""
        # Calculate weights based on interaction count
        weight_cf = min(0.8, self.get_interactions_count(user_id) / (2.0 * self.threshold_interactions))
        weight_cb = 1.0 - weight_cf
        
        # Get CF recommendations
        cf_recs_df = pd.DataFrame()
        try:
            cf_recs_df = cf_model.recommend(user_id, top_n=(top_n * 3), item_type=item_type)
        except Exception as e:
            print(f"Error getting CF recommendations for user {user_id}: {e}")
        
        # Get Content-Based recommendations
        cbf = ContentBasedRecommender(
            content_fetcher=self.content_fetcher,
            new_user=False,
            user_id=user_id,
            limit=top_n * 5
        )
        cb_recs_df = cbf.recommend(top_n=(top_n * 3), use_mmr=use_mmr_for_cb)
        
        # Filter by item type if needed
        if not cb_recs_df.empty and 'type' in cb_recs_df.columns and item_type is not None:
            cb_recs_df = cb_recs_df[cb_recs_df['type'] == item_type].copy()
        
        # Normalize scores
        cf_recs_df = self._normalize_scores('cf', cf_recs_df, 'score')
        cb_recs_df = self._normalize_scores('cb', cb_recs_df, 'score')
        
        # Blend recommendations
        return self._blend_recommendations(cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n)

    def _blend_recommendations(self, cb_recs_df, cf_recs_df, weight_cb, weight_cf, top_n):
        """Blend Content-Based and Collaborative Filtering recommendations."""
        # Return empty DataFrame if both are empty
        if (cb_recs_df is None or cb_recs_df.empty) and (cf_recs_df is None or cf_recs_df.empty):
            return pd.DataFrame()
        
        # Make sure 'item' column exists in CB recommendations
        if not cb_recs_df.empty and 'item' not in cb_recs_df.columns and 'name' in cb_recs_df.columns:
            cb_recs_df = cb_recs_df.copy()
            cb_recs_df['item'] = cb_recs_df['name']
        
        # Make sure 'type' column exists in CB recommendations
        if not cb_recs_df.empty and 'type' not in cb_recs_df.columns:
            cb_recs_df['type'] = 'Unknown'
        
        # Define merge columns
        merge_cols = ['item', 'type']
        
        # Handle empty dataframes
        if cf_recs_df.empty and not cb_recs_df.empty:
            # Only CB recommendations available
            merged_recs = cb_recs_df.copy()
            if 'norm_score' in merged_recs.columns:
                merged_recs['hybrid_score'] = weight_cb * merged_recs['norm_score']
            else:
                merged_recs['hybrid_score'] = 0.0
        elif not cf_recs_df.empty and cb_recs_df.empty:
            # Only CF recommendations available
            merged_recs = cf_recs_df.copy()
            if 'norm_score' in merged_recs.columns:
                merged_recs['hybrid_score'] = weight_cf * merged_recs['norm_score']
            else:
                merged_recs['hybrid_score'] = 0.0
        else:
            # Both types available, perform outer merge
            # Make sure the required columns exist
            for col in merge_cols:
                if col not in cf_recs_df.columns:
                    cf_recs_df[col] = 'Unknown'
                if col not in cb_recs_df.columns:
                    cb_recs_df[col] = 'Unknown'

            # Check if norm_score columns exist
            if 'norm_score' not in cf_recs_df.columns:
                cf_recs_df['norm_score'] = 0.0
            if 'norm_score' not in cb_recs_df.columns:
                cb_recs_df['norm_score'] = 0.0
                
            # Perform merge
            merged_recs = pd.merge(
                cf_recs_df[merge_cols + ['norm_score']],
                cb_recs_df[merge_cols + ['norm_score']],
                on=merge_cols,
                how='outer',
                suffixes=('_cf', '_cb')
            )
            
            # Fill NA values with 0
            merged_recs['norm_score_cf'] = merged_recs['norm_score_cf'].fillna(0)
            merged_recs['norm_score_cb'] = merged_recs['norm_score_cb'].fillna(0)
            
            # Calculate weighted hybrid score
            merged_recs['hybrid_score'] = (weight_cf * merged_recs['norm_score_cf']) + (weight_cb * merged_recs['norm_score_cb'])
        
        # If merged_recs is empty, return empty DataFrame
        if merged_recs.empty:
            return pd.DataFrame()
            
        # Get top N recommendations
        top_hybrid_recs = merged_recs.nlargest(min(top_n, len(merged_recs)), 'hybrid_score')
        
        # Add name from original dataframes if available
        if 'name' not in top_hybrid_recs.columns:
            name_map = {}
            if not cf_recs_df.empty and 'name' in cf_recs_df.columns and 'item' in cf_recs_df.columns:
                name_map.update(dict(zip(cf_recs_df['item'], cf_recs_df['name'])))
            if not cb_recs_df.empty and 'name' in cb_recs_df.columns and 'item' in cb_recs_df.columns:
                name_map.update(dict(zip(cb_recs_df['item'], cb_recs_df['name'])))
            
            if name_map:
                top_hybrid_recs['name'] = top_hybrid_recs['item'].map(name_map).fillna("Unknown")
        
        # Add rank
        top_hybrid_recs['rank'] = range(1, len(top_hybrid_recs) + 1)
        
        # Rename hybrid_score to score for consistency
        top_hybrid_recs = top_hybrid_recs.rename(columns={'hybrid_score': 'score'})
        
        # Select and order columns
        result_columns = ['rank', 'item', 'type', 'score']
        if 'name' in top_hybrid_recs.columns:
            result_columns.append('name')
            
        return top_hybrid_recs[result_columns] if set(result_columns).issubset(top_hybrid_recs.columns) else top_hybrid_recs

    def recommend_with_epsilon_greedy(self, user_id, top_n=10, item_type=None, use_mmr_for_cb=True, epsilon=0.2):
        """Get recommendations using epsilon-greedy exploration strategy."""
        # Get standard recommendations
        recommendations = self.recommend(user_id, top_n=top_n, item_type=item_type, use_mmr_for_cb=use_mmr_for_cb)
        
        # Return empty DataFrame if no recommendations
        if recommendations is None or recommendations.empty:
            return pd.DataFrame()
        
        # Select CF model based on item type
        cf_model = self._select_cf_model(item_type)
        
        # Try to apply epsilon-greedy strategy
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
            return recommendations  # Fall back to standard recommendations


if __name__ == "__main__":
    user_id_zero_interactions = "1aba2aa2-5be8-4710-a0cd-f3294b7a6d49"
    user_id_few_interactions = "b9c32bc3-4b7f-46fd-af3b-ca48060b89a1" 
    user_id_active = "b9c32bc3-4b7f-46fd-af3b-ca48060b89a1" 

    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)

    print("\n--- Fitting CF Models ---")
    # Initialize CF models
    user_cf_model = UserBasedCF(db_client, k_neighbors=10, min_overlap=2, min_sim=0.05)
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
        
        # Test with Event (should also use item-based CF model for active users)
        recs_event = hybrid_model.recommend(
            user_id=user_id_few_interactions, 
            top_n=5, 
            item_type='Event'
        )
        print(f"\nHybrid Recommendations (Stage 2 - Few Interactions, Event):")
        print(recs_event)
        
        # Test with Trip (should use user-based CF model for active users)
        recs_trip = hybrid_model.recommend(
            user_id=user_id_active, 
            top_n=5, 
            item_type='Trip'
        )
        print(f"\nHybrid Recommendations (Stage 3 - Active User, Trip):")
        
        # Add safety check for NoneType
        if recs_trip is not None and not recs_trip.empty and 'score' in recs_trip.columns and 'name' in recs_trip.columns:
            print(recs_trip[['score', 'name']])
        else:
            print("No Trip recommendations found or missing required columns.")

        recs_des = hybrid_model.recommend(
            user_id=user_id_active, 
            top_n=5, 
            item_type='Destination'
        )
        print(f"\nHybrid Recommendations (Stage 3 - Active User, Destinations):")
        
        # Add safety check for NoneType
        if recs_des is not None and not recs_des.empty and 'score' in recs_des.columns and 'name' in recs_des.columns:
            print(recs_des[['score', 'name']])
        else:
            print("No Destination recommendations found or missing required columns.")
        
        # Test epsilon-greedy recommendation approach
        print("\n--- Testing Epsilon-Greedy Recommendations ---")
        epsilon_recs = hybrid_model.recommend_with_epsilon_greedy(
            user_id=user_id_active,
            top_n=5,
            item_type='Destination',
            epsilon=0.2
        )
        print(f"Epsilon-Greedy Recommendations (Destination):")
        print(epsilon_recs[['score', 'name']])
        
    else:
        print("\nSkipping Hybrid Recommendations because CF model fitting failed.")

    db_client.close()
    print("\nNeo4j driver closed.")