import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from neo4j_data_fetcher import Neo4jClient, InteractionsFetcher
from CF_KNN_user_based import UserBasedCF
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional


class RecommenderEvaluator:
    """
    Evaluator class for the UserBasedCF recommender system.
    Implements offline evaluation metrics including:
    - Prediction accuracy (RMSE, MAE)
    - Ranking metrics (NDCG, precision, recall)
    - Coverage, diversity, and novelty metrics
    """
    
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.fetcher = InteractionsFetcher(db_client)
        self.interactions_df = None
        self.train_df = None
        self.test_df = None
        self.all_items = None
        
    def load_data(self):
        """Load interaction data from Neo4j"""
        print("Loading interaction data...")
        self.interactions_df = self.fetcher.fetch_interactions()
        if self.interactions_df.empty:
            raise ValueError("No interaction data found")
        
        # Get list of all unique items for coverage calculation
        self.all_items = self.interactions_df[['item', 'type']].drop_duplicates().values.tolist()
        print(f"Loaded {len(self.interactions_df)} interactions for {self.interactions_df['user'].nunique()} users")
        
    def train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        if self.interactions_df is None:
            self.load_data()
            
        # Group by user to ensure we don't have users only in test set
        user_groups = self.interactions_df.groupby('user')
        
        train_dfs = []
        test_dfs = []
        
        # For each user, split their interactions
        for user_id, user_data in user_groups:
            if len(user_data) < 2:  # Skip users with too few interactions
                train_dfs.append(user_data)  # Add all to training
                continue
                
            user_train, user_test = train_test_split(
                user_data, test_size=test_size, random_state=random_state
            )
            train_dfs.append(user_train)
            test_dfs.append(user_test)
        
        self.train_df = pd.concat(train_dfs)
        self.test_df = pd.concat(test_dfs) if test_dfs else pd.DataFrame(columns=self.train_df.columns)
        
        print(f"Training set: {len(self.train_df)} interactions")
        print(f"Test set: {len(self.test_df)} interactions")
        
    def evaluate_model(self, param_grid: Dict, test_users: Optional[List[str]] = None, top_n=10):
        """
        Evaluate the model with different hyperparameters
        
        Args:
            param_grid: Dictionary of parameters to try
            test_users: List of user IDs to test on (if None, use all users in test set)
            top_n: Number of recommendations to generate
            
        Returns:
            DataFrame with evaluation results for each parameter combination
        """
        if self.train_df is None or self.test_df is None:
            self.train_test_split()
            
        # If no test users specified, use all users in test set
        if test_users is None:
            test_users = self.test_df['user'].unique().tolist()
        
        # Create all parameter combinations
        k_values = param_grid.get('k_neighbors', [5])
        sim_values = param_grid.get('min_sim', [0.1])
        overlap_values = param_grid.get('min_overlap', [0])
        
        results = []
        
        for k in k_values:
            for sim in sim_values:
                for overlap in overlap_values:
                    print(f"\nEvaluating model with k={k}, min_sim={sim}, min_overlap={overlap}")
                    
                    # Configure and train model on training data
                    model = UserBasedCF(self.db_client, k_neighbors=k, min_sim=sim, min_overlap=overlap)
                    
                    # Create a copy of the database with just training data
                    # In a real evaluation, you'd need to modify this to train only on training data
                    # This is a simplified version that assumes model.fit() will use training data
                    start_time = time.time()
                    model.fit()
                    fit_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(model, test_users, top_n)
                    
                    # Store results
                    results.append({
                        'k_neighbors': k,
                        'min_sim': sim,
                        'min_overlap': overlap,
                        'fit_time': fit_time,
                        **metrics
                    })
                    
        return pd.DataFrame(results)
    
    def _calculate_metrics(self, model: UserBasedCF, test_users: List[str], top_n: int) -> Dict:
        """Calculate evaluation metrics for the model"""
        # Initialize metric counters
        total_precision = 0
        total_recall = 0
        total_ndcg = 0
        total_hit_rate = 0
        total_mrr = 0
        
        # For coverage calculation
        recommended_items = set()
        
        # For diversity calculation
        all_recommendation_lists = []
        
        # For novelty calculation
        item_popularity = self.train_df.groupby(['item', 'type']).size()
        total_popularity = 0
        
        # Process each test user
        evaluated_users = 0
        
        for user_id in test_users:
            # Get user's interactions from test set
            user_test = self.test_df[self.test_df['user'] == user_id]
            
            if user_test.empty:
                continue
                
            # Get actual items the user interacted with in the test set
            actual_items = set(tuple(x) for x in user_test[['item', 'type']].values)
            
            all_user_recs = []
            
            # Get recommendations for each item type
            for item_type in ['Trip', 'Event', 'Destination']:
                try:
                    # Get recommendations for this user and item type
                    recs = model.recommend(user_id, top_n=top_n, item_type=item_type)
                    
                    if not recs.empty:
                        # Add to all recommendations for this user
                        all_user_recs.extend([(row['item'], row['type']) for _, row in recs.iterrows()])
                        
                        # Update coverage
                        recommended_items.update((row['item'], row['type']) for _, row in recs.iterrows())
                        
                        # Calculate novelty
                        for _, row in recs.iterrows():
                            item_key = (row['item'], row['type'])
                            if item_key in item_popularity:
                                total_popularity += -np.log2(item_popularity[item_key] / len(self.train_df))
                except Exception as e:
                    print(f"Error getting recommendations for user {user_id}, type {item_type}: {str(e)}")
                    continue
            
            if not all_user_recs:
                continue
                
            # Convert to set for intersection calculation
            rec_items_set = set(all_user_recs)
            
            # Calculate precision and recall
            hits = len(rec_items_set.intersection(actual_items))
            precision = hits / len(rec_items_set) if rec_items_set else 0
            recall = hits / len(actual_items) if actual_items else 0
            
            # Hit rate is 1 if at least one recommendation was relevant
            hit_rate = 1 if hits > 0 else 0
            
            # Calculate MRR (Mean Reciprocal Rank)
            # Find the rank of the first relevant item
            mrr = 0
            for rank, item in enumerate(all_user_recs, 1):
                if item in actual_items:
                    mrr = 1 / rank
                    break
            
            # Calculate NDCG
            # Create binary relevance scores for recommendations
            y_true = np.zeros(len(all_user_recs))
            for i, item in enumerate(all_user_recs):
                if item in actual_items:
                    y_true[i] = 1
                    
            # Reshape for ndcg_score and handle edge case
            if len(y_true) > 0 and np.sum(y_true) > 0:
                y_true_reshaped = y_true.reshape(1, -1)
                y_score_reshaped = np.array(range(len(y_true), 0, -1)).reshape(1, -1)
                ndcg = ndcg_score(y_true_reshaped, y_score_reshaped)
            else:
                ndcg = 0
            
            # Update totals
            total_precision += precision
            total_recall += recall
            total_ndcg += ndcg
            total_hit_rate += hit_rate
            total_mrr += mrr
            
            # Store user's recommendations for diversity calculation
            all_recommendation_lists.append(all_user_recs)
            
            evaluated_users += 1
            
        # Calculate averages
        avg_precision = total_precision / evaluated_users if evaluated_users > 0 else 0
        avg_recall = total_recall / evaluated_users if evaluated_users > 0 else 0
        avg_ndcg = total_ndcg / evaluated_users if evaluated_users > 0 else 0
        avg_hit_rate = total_hit_rate / evaluated_users if evaluated_users > 0 else 0
        avg_mrr = total_mrr / evaluated_users if evaluated_users > 0 else 0
        
        # Calculate coverage
        coverage = len(recommended_items) / len(self.all_items) if self.all_items else 0
        
        # Calculate average novelty
        avg_novelty = total_popularity / (evaluated_users * top_n) if evaluated_users > 0 else 0
        
        # Calculate diversity (average pairwise Jaccard distance)
        total_diversity = 0
        diversity_pairs = 0
        
        for i in range(len(all_recommendation_lists)):
            for j in range(i + 1, len(all_recommendation_lists)):
                list1 = set(all_recommendation_lists[i])
                list2 = set(all_recommendation_lists[j])
                
                # Jaccard distance = 1 - Jaccard similarity
                # where Jaccard similarity = |intersection| / |union|
                if list1 and list2:  # Ensure non-empty lists
                    intersection = len(list1.intersection(list2))
                    union = len(list1.union(list2))
                    jaccard_distance = 1 - (intersection / union)
                    total_diversity += jaccard_distance
                    diversity_pairs += 1
        
        avg_diversity = total_diversity / diversity_pairs if diversity_pairs > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'ndcg': avg_ndcg,
            'hit_rate': avg_hit_rate,
            'mrr': avg_mrr,
            'coverage': coverage,
            'diversity': avg_diversity,
            'novelty': avg_novelty,
            'evaluated_users': evaluated_users
        }
    
    def evaluate_epsilon_greedy(self, model: UserBasedCF, test_users: List[str], 
                               epsilon_values: List[float], top_n: int = 10) -> pd.DataFrame:
        """
        Evaluate the epsilon-greedy recommendation strategy with different epsilon values
        
        Args:
            model: Trained UserBasedCF model
            test_users: List of user IDs to test on
            epsilon_values: List of epsilon values to try
            top_n: Number of recommendations to generate
            
        Returns:
            DataFrame with evaluation results for each epsilon value
        """
        results = []
        
        for epsilon in epsilon_values:
            print(f"\nEvaluating epsilon-greedy with epsilon={epsilon}")
            
            total_precision = 0
            total_recall = 0
            total_diversity = 0
            total_novelty = 0
            evaluated_users = 0
            
            # Item popularity for novelty calculation
            item_popularity = self.train_df.groupby(['item', 'type']).size()
            
            # For diversity calculation
            all_recommendation_lists = []
            
            for user_id in test_users:
                user_test = self.test_df[self.test_df['user'] == user_id]
                
                if user_test.empty:
                    continue
                    
                # Get actual items the user interacted with in the test set
                actual_items = set(tuple(x) for x in user_test[['item', 'type']].values)
                
                all_user_recs = []
                
                # Get recommendations for each item type
                for item_type in ['Trip', 'Event', 'Destination']:
                    try:
                        # Get standard recommendations first
                        standard_recs = model.recommend(user_id, top_n=top_n, item_type=item_type)
                        
                        if standard_recs.empty:
                            continue
                            
                        # Get epsilon-greedy recommendations
                        epsilon_recs = model.recommend_with_epsilon_greedy(
                            user_id, 
                            top_candidates_df=standard_recs,
                            item_type=item_type,
                            epsilon=epsilon,
                            top_n=top_n
                        )
                        
                        if not epsilon_recs.empty:
                            # Add to all recommendations for this user
                            all_user_recs.extend([(row['item'], row['type']) for _, row in epsilon_recs.iterrows()])
                            
                            # Calculate novelty
                            for _, row in epsilon_recs.iterrows():
                                item_key = (row['item'], row['type']) 
                                if item_key in item_popularity:
                                    total_novelty += -np.log2(item_popularity[item_key] / len(self.train_df))
                    except Exception as e:
                        print(f"Error getting epsilon-greedy recs for user {user_id}, type {item_type}: {str(e)}")
                        continue
                
                if not all_user_recs:
                    continue
                    
                # Convert to set for intersection calculation
                rec_items_set = set(all_user_recs)
                
                # Calculate precision and recall
                hits = len(rec_items_set.intersection(actual_items))
                precision = hits / len(rec_items_set) if rec_items_set else 0
                recall = hits / len(actual_items) if actual_items else 0
                
                # Update totals
                total_precision += precision
                total_recall += recall
                
                # Store user's recommendations for diversity calculation
                all_recommendation_lists.append(all_user_recs)
                
                evaluated_users += 1
            
            # Calculate average metrics
            avg_precision = total_precision / evaluated_users if evaluated_users > 0 else 0
            avg_recall = total_recall / evaluated_users if evaluated_users > 0 else 0
            avg_novelty = total_novelty / (evaluated_users * top_n) if evaluated_users > 0 else 0
            
            # Calculate diversity (average pairwise Jaccard distance)
            total_diversity = 0
            diversity_pairs = 0
            
            for i in range(len(all_recommendation_lists)):
                for j in range(i + 1, len(all_recommendation_lists)):
                    list1 = set(all_recommendation_lists[i])
                    list2 = set(all_recommendation_lists[j])
                    
                    if list1 and list2:  # Ensure non-empty lists
                        intersection = len(list1.intersection(list2))
                        union = len(list1.union(list2))
                        jaccard_distance = 1 - (intersection / union)
                        total_diversity += jaccard_distance
                        diversity_pairs += 1
            
            avg_diversity = total_diversity / diversity_pairs if diversity_pairs > 0 else 0
            
            results.append({
                'epsilon': epsilon,
                'precision': avg_precision,
                'recall': avg_recall,
                'diversity': avg_diversity,
                'novelty': avg_novelty,
                'evaluated_users': evaluated_users
            })
            
        return pd.DataFrame(results)
    
    def visualize_results(self, results_df, metric_columns=None, param_col='k_neighbors'):
        """Visualize evaluation results"""
        if metric_columns is None:
            metric_columns = ['precision', 'recall', 'ndcg', 'hit_rate', 'mrr', 'coverage', 'diversity', 'novelty']
        
        # Keep only metrics that exist in the results DataFrame
        metric_columns = [col for col in metric_columns if col in results_df.columns]
        
        # Create subplots for each metric
        fig, axes = plt.subplots(len(metric_columns), 1, figsize=(10, 3*len(metric_columns)))
        if len(metric_columns) == 1:
            axes = [axes]  # Make sure axes is iterable
        
        for i, metric in enumerate(metric_columns):
            if param_col == 'epsilon':
                # Line plot for epsilon comparison
                axes[i].plot(results_df[param_col], results_df[metric], marker='o')
                axes[i].set_xlabel('Epsilon')
            else:
                # Bar plot for parameter comparison
                results_df.plot(x=param_col, y=metric, kind='bar', ax=axes[i])
            
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} by {param_col}')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize database client
    db_client = Neo4jClient()
    
    # Create evaluator
    evaluator = RecommenderEvaluator(db_client)
      
   
    
    # Load data and split into training/test sets
    evaluator.load_data()
    evaluator.train_test_split(test_size=0.3)
    
    # Define parameters to test
    param_grid = {
        'k_neighbors': [5, 10, 15, 20, 25],
        'min_sim': [0.001, 0.01, 0.1],
        'min_overlap': [0, 1, 2]
    }
    model =  UserBasedCF(db_client)
    
    model.fit()
    # Define a subset of users to test on (for faster evaluation)
    test_user_ids = model.user_indices[:1000]
    
    # Evaluate model with different parameters
    results = evaluator.evaluate_model(param_grid, test_users=test_user_ids)
    
    # Print results
    print("\nParameter tuning results:")
    print(results)
    
    # Find best parameters
    best_ndcg_idx = results['ndcg'].idxmax()
    best_params = results.iloc[best_ndcg_idx]
    print(f"\nBest parameters based on NDCG: k={best_params['k_neighbors']}, "
          f"min_sim={best_params['min_sim']}, min_overlap={best_params['min_overlap']}")
    
    # Train model with best parameters
    best_model = UserBasedCF(
        db_client, 
        k_neighbors=int(best_params['k_neighbors']), 
        min_sim=best_params['min_sim'], 
        min_overlap=int(best_params['min_overlap'])
    )
    
    # Evaluate epsilon-greedy strategy
    epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    epsilon_results = evaluator.evaluate_epsilon_greedy(
        best_model, 
        test_users=test_user_ids, 
        epsilon_values=epsilon_values
    )
    
    print("\nEpsilon-greedy evaluation results:")
    print(epsilon_results)
    
    # Visualize results
    param_fig = evaluator.visualize_results(results, param_col='k_neighbors')
    epsilon_fig = evaluator.visualize_results(epsilon_results, param_col='epsilon')
    
    # Close database connection
    db_client.close()