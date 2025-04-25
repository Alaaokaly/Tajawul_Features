import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from neo4j_data_fetcher import Neo4jClient, InteractionsFetcher
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
import time
import os
import json
from datetime import datetime

class RecommenderEvaluator:

    
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.fetcher = InteractionsFetcher(db_client)
        self.interactions_df = None
        self.train_df = None
        self.test_df = None
        self.all_items = None
        
    def load_data(self):
        print("Loading interaction data...")
        self.interactions_df = self.fetcher.fetch_interactions()
        if self.interactions_df.empty:
            raise ValueError("No interaction data found")
        
        # Get list of all unique items for coverage calculation
        self.all_items = self.interactions_df[['item', 'type']].drop_duplicates().values.tolist()
        print(f"Loaded {len(self.interactions_df)} interactions for {self.interactions_df['user'].nunique()} users")
        
    def train_test_split(self, test_size=0.2, random_state=42):
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
                    
                    model = UserBasedCF(self.db_client, k_neighbors=k, min_sim=sim, min_overlap=overlap)
    
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

        total_precision = 0
        total_recall = 0
        total_ndcg = 0
        total_hit_rate = 0
        total_mrr = 0
    
        recommended_items = set()

        all_recommendation_lists = []
        
        item_popularity = self.train_df.groupby(['item', 'type']).size()
        total_popularity = 0

        evaluated_users = 0
        
        for user_id in test_users:
            user_test = self.test_df[self.test_df['user'] == user_id]
            
            if user_test.empty:
                continue

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

            mrr = 0
            for rank, item in enumerate(all_user_recs, 1):
                if item in actual_items:
                    mrr = 1 / rank
                    break

            y_true = np.zeros(len(all_user_recs))
            for i, item in enumerate(all_user_recs):
                if item in actual_items:
                    y_true[i] = 1

            if len(y_true) > 0 and np.sum(y_true) > 0:
                y_true_reshaped = y_true.reshape(1, -1)
                y_score_reshaped = np.array(range(len(y_true), 0, -1)).reshape(1, -1)
                ndcg = ndcg_score(y_true_reshaped, y_score_reshaped)
            else:
                ndcg = 0

            total_precision += precision
            total_recall += recall
            total_ndcg += ndcg
            total_hit_rate += hit_rate
            total_mrr += mrr
            all_recommendation_lists.append(all_user_recs)
            
            evaluated_users += 1
            

        avg_precision = total_precision / evaluated_users if evaluated_users > 0 else 0
        avg_recall = total_recall / evaluated_users if evaluated_users > 0 else 0
        avg_ndcg = total_ndcg / evaluated_users if evaluated_users > 0 else 0
        avg_hit_rate = total_hit_rate / evaluated_users if evaluated_users > 0 else 0
        avg_mrr = total_mrr / evaluated_users if evaluated_users > 0 else 0

        coverage = len(recommended_items) / len(self.all_items) if self.all_items else 0

        avg_novelty = total_popularity / (evaluated_users * top_n) if evaluated_users > 0 else 0

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






db_client = Neo4jClient()

# Create evaluator
evaluator = RecommenderEvaluator(db_client)

# Create output directory if it doesn't exist
output_dir = "eval_cf_user"
os.makedirs(output_dir, exist_ok=True)

# Log file setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{output_dir}/evaluation_log_{timestamp}.txt"

def log_message(message, print_to_console=True):
    """Write message to log file and optionally print to console"""
    with open(log_filename, 'a') as f:
        f.write(f"{message}\n")
    if print_to_console:
        print(message)

log_message(f"=== UserBasedCF Evaluation Started at {timestamp} ===")

# Load data and split into training/test sets
try:
    log_message("Loading data...")
    evaluator.load_data()
    evaluator.train_test_split(test_size=0.3)
    log_message(f"Data loaded: {len(evaluator.interactions_df)} interactions, {evaluator.interactions_df['user'].nunique()} users")
except Exception as e:
    log_message(f"Error loading data: {str(e)}")
    db_client.close()
    exit(1)

# Define parameters to test
param_grid = {
    'k_neighbors': [10, 15, 20, 25, 35],
    'min_sim': [0.001, 0.01, 0.02, 0.04, 0.06, 0.09, 0.15],
    'min_overlap': [0, 1, 2]
}

# Initial model for getting test users
try:
    model = UserBasedCF(db_client)
    model.fit()
    # Define a subset of users to test on (for faster evaluation)
    test_user_ids = model.user_indices[:1000]
    log_message(f"Using {len(test_user_ids)} test users for evaluation")
except Exception as e:
    log_message(f"Error initializing model: {str(e)}")
    db_client.close()
    exit(1)

# Evaluate model with different parameters
try:
    log_message("Starting parameter grid search evaluation...")
    results = evaluator.evaluate_model(param_grid, test_users=test_user_ids)
    
    # Save results to CSV
    results_csv_path = f"{output_dir}/parameter_tuning_results_{timestamp}.csv"
    results.to_csv(results_csv_path, index=False)
    log_message(f"Parameter tuning results saved to {results_csv_path}")
    
    # Log summary of results
    log_message("\nParameter tuning results summary:")
    summary_stats = results.describe()
    log_message(summary_stats.to_string())
    
    # Find best parameters
    best_ndcg_idx = results['ndcg'].idxmax()
    best_params = results.iloc[best_ndcg_idx]
    log_message(f"\nBest parameters based on NDCG: k={best_params['k_neighbors']}, "
          f"min_sim={best_params['min_sim']}, min_overlap={best_params['min_overlap']}")
    
    # Save best parameters as JSON
    best_params_json = {
        'k_neighbors': int(best_params['k_neighbors']),
        'min_sim': float(best_params['min_sim']),
        'min_overlap': int(best_params['min_overlap']),
        'ndcg': float(best_params['ndcg']),
        'precision': float(best_params['precision']),
        'recall': float(best_params['recall']),
        'hit_rate': float(best_params['hit_rate']),
        'mrr': float(best_params['mrr']),
        'coverage': float(best_params['coverage']),
        'diversity': float(best_params['diversity']),
        'novelty': float(best_params['novelty'])
    }
    
    with open(f"{output_dir}/best_params_{timestamp}.json", 'w') as f:
        json.dump(best_params_json, f, indent=4)
    
except Exception as e:
    log_message(f"Error during parameter tuning: {str(e)}")
    db_client.close()
    exit(1)

# Train model with best parameters
try:
    log_message("\nTraining model with best parameters...")
    best_model = UserBasedCF(
        db_client, 
        k_neighbors=int(best_params['k_neighbors']), 
        min_sim=float(best_params['min_sim']), 
        min_overlap=int(best_params['min_overlap'])
    )
    
    start_time = time.time()
    best_model.fit()
    fit_time = time.time() - start_time
    log_message(f"Model fitted in {fit_time:.2f} seconds")
    
    # Evaluate epsilon-greedy strategy
    log_message("\nEvaluating epsilon-greedy strategy...")
    epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    epsilon_results = evaluator.evaluate_epsilon_greedy(
        best_model, 
        test_users=test_user_ids, 
        epsilon_values=epsilon_values
    )
    
    # Save epsilon-greedy results
    epsilon_csv_path = f"{output_dir}/epsilon_greedy_results_{timestamp}.csv"
    epsilon_results.to_csv(epsilon_csv_path, index=False)
    log_message(f"Epsilon-greedy evaluation results saved to {epsilon_csv_path}")
    
    log_message("\nEpsilon-greedy evaluation results:")
    log_message(epsilon_results.to_string())
    
    # Find best epsilon value
    best_epsilon_idx = epsilon_results['diversity'].idxmax()
    best_epsilon = epsilon_results.iloc[best_epsilon_idx]
    log_message(f"\nBest epsilon for diversity: {best_epsilon['epsilon']}")
    
    best_epsilon_precision_idx = epsilon_results['precision'].idxmax()
    best_epsilon_precision = epsilon_results.iloc[best_epsilon_precision_idx]
    log_message(f"Best epsilon for precision: {best_epsilon_precision['epsilon']}")
    
except Exception as e:
    log_message(f"Error during epsilon-greedy evaluation: {str(e)}")
    db_client.close()
    exit(1)

# Generate and save visualizations
try:
    log_message("\nGenerating visualizations...")
    
    # Parameter tuning visualization
    param_fig = evaluator.visualize_results(results, param_col='k_neighbors')
    param_fig_path = f"{output_dir}/parameter_tuning_vis_{timestamp}.png"
    param_fig.savefig(param_fig_path)
    log_message(f"Parameter tuning visualization saved to {param_fig_path}")
    
    # Epsilon-greedy visualization
    epsilon_fig = evaluator.visualize_results(epsilon_results, param_col='epsilon')
    epsilon_fig_path = f"{output_dir}/epsilon_greedy_vis_{timestamp}.png"
    epsilon_fig.savefig(epsilon_fig_path)
    log_message(f"Epsilon-greedy visualization saved to {epsilon_fig_path}")
    
    # Additional visualizations for specific metrics
    
    # K-neighbors impact on precision and recall
    plt.figure(figsize=(10, 6))
    k_results = results.groupby('k_neighbors')[['precision', 'recall']].mean()
    k_results.plot(kind='line', marker='o')
    plt.title('Impact of k_neighbors on Precision and Recall')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    k_impact_path = f"{output_dir}/k_neighbors_impact_{timestamp}.png"
    plt.savefig(k_impact_path)
    log_message(f"K-neighbors impact visualization saved to {k_impact_path}")
    
    # Min similarity impact on metrics
    plt.figure(figsize=(10, 6))
    sim_results = results.groupby('min_sim')[['precision', 'diversity', 'novelty']].mean()
    sim_results.plot(kind='line', marker='o')
    plt.title('Impact of Minimum Similarity on Key Metrics')
    plt.xlabel('Minimum Similarity Threshold')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    sim_impact_path = f"{output_dir}/min_sim_impact_{timestamp}.png"
    plt.savefig(sim_impact_path)
    log_message(f"Minimum similarity impact visualization saved to {sim_impact_path}")
    
except Exception as e:
    log_message(f"Error generating visualizations: {str(e)}")

# Sample recommendations
try:
    log_message("\nGenerating sample recommendations with best model...")
    sample_users = test_user_ids[:5]  # Take 5 sample users
    
    sample_recs_path = f"{output_dir}/sample_recommendations_{timestamp}.txt"
    with open(sample_recs_path, 'w') as f:
        for user_id in sample_users:
            f.write(f"\nRecommendations for user {user_id}:\n")
            for item_type in ['Trip', 'Event', 'Destination']:
                try:
                    recs = best_model.recommend(user_id, top_n=5, item_type=item_type)
                    f.write(f"\n  Top {item_type} recommendations:\n")
                    f.write(recs.to_string() + "\n")
                    
                    # With epsilon-greedy (best epsilon value)
                    best_eps = best_epsilon['epsilon']
                    eps_recs = best_model.recommend_with_epsilon_greedy(
                        user_id, 
                        top_candidates_df=recs,
                        item_type=item_type,
                        epsilon=best_eps
                    )
                    f.write(f"\n  Epsilon-greedy ({best_eps}) {item_type} recommendations:\n")
                    f.write(eps_recs.to_string() + "\n")
                except Exception as e:
                    f.write(f"  Error getting {item_type} recommendations: {str(e)}\n")
    
    log_message(f"Sample recommendations saved to {sample_recs_path}")
except Exception as e:
    log_message(f"Error generating sample recommendations: {str(e)}")

# Generate comprehensive report
try:
    report_path = f"{output_dir}/evaluation_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(f"# User-Based Collaborative Filtering Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Statistics\n\n")
        f.write(f"- Total interactions: {len(evaluator.interactions_df)}\n")
        f.write(f"- Number of users: {evaluator.interactions_df['user'].nunique()}\n")
        f.write(f"- Number of unique items: {len(evaluator.all_items)}\n")
        f.write(f"- Training set size: {len(evaluator.train_df)}\n")
        f.write(f"- Test set size: {len(evaluator.test_df)}\n\n")
        
        f.write("## Parameter Tuning Results\n\n")
        f.write("### Best Parameters\n\n")
        f.write(f"- k_neighbors: {int(best_params['k_neighbors'])}\n")
        f.write(f"- min_sim: {best_params['min_sim']}\n")
        f.write(f"- min_overlap: {int(best_params['min_overlap'])}\n\n")
        
        f.write("### Best Performance Metrics\n\n")
        f.write(f"- Precision: {best_params['precision']:.4f}\n")
        f.write(f"- Recall: {best_params['recall']:.4f}\n")
        f.write(f"- NDCG: {best_params['ndcg']:.4f}\n")
        f.write(f"- Hit Rate: {best_params['hit_rate']:.4f}\n")
        f.write(f"- MRR: {best_params['mrr']:.4f}\n")
        f.write(f"- Coverage: {best_params['coverage']:.4f}\n")
        f.write(f"- Diversity: {best_params['diversity']:.4f}\n")
        f.write(f"- Novelty: {best_params['novelty']:.4f}\n\n")
        
        f.write("## Epsilon-Greedy Evaluation\n\n")
        f.write("### Best Epsilon Value\n\n")
        f.write(f"- For diversity: {best_epsilon['epsilon']}\n")
        f.write(f"- For precision: {best_epsilon_precision['epsilon']}\n\n")
        
        f.write("### Epsilon Impact Summary\n\n")
        f.write("| Epsilon | Precision | Recall | Diversity | Novelty |\n")
        f.write("|---------|-----------|--------|-----------|--------|\n")
        for _, row in epsilon_results.iterrows():
            f.write(f"| {row['epsilon']:.1f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['diversity']:.4f} | {row['novelty']:.4f} |\n")
        
        f.write("\n## Conclusion\n\n")
        f.write("The evaluation shows that the User-Based Collaborative Filtering approach performs best with ")
        f.write(f"k={int(best_params['k_neighbors'])}, min_sim={best_params['min_sim']}, and min_overlap={int(best_params['min_overlap'])}. ")
        
        if best_epsilon['epsilon'] > 0:
            f.write(f"Adding randomness with epsilon={best_epsilon['epsilon']} improves recommendation diversity ")
            f.write(f"from {epsilon_results.iloc[0]['diversity']:.4f} to {best_epsilon['diversity']:.4f}, ")
            
            precision_diff = best_epsilon['precision'] - epsilon_results.iloc[0]['precision']
            if precision_diff >= 0:
                f.write(f"while also maintaining good precision (change of {precision_diff:.4f}).")
            else:
                f.write(f"with a small precision trade-off of {abs(precision_diff):.4f}.")
        else:
            f.write(f"Adding randomness with epsilon-greedy did not improve the recommendation quality.")
            
    log_message(f"Comprehensive evaluation report saved to {report_path}")
except Exception as e:
    log_message(f"Error generating evaluation report: {str(e)}")

# Close database connection
db_client.close()
log_message(f"\n=== Evaluation Completed at {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")