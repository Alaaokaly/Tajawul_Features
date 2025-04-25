from sklearn.metrics import ndcg_score
from neo4j_data_fetcher import Neo4jClient, InteractionsFetcher
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from datetime import datetime
from hybird import HybridRecommender
from typing import Dict, List, Optional

class HybridRecommenderEvaluator:
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

    def evaluate_hybrid_model(self, hybrid_recommender: HybridRecommender, 
                             test_users: Optional[List[str]] = None, top_n=10):
        """Evaluate the hybrid recommender model."""
        if self.train_df is None or self.test_df is None:
            self.train_test_split()
            
        # If no test users specified, use all users in test set
        if test_users is None:
            test_users = self.test_df['user'].unique().tolist()
        
        print(f"\nEvaluating hybrid recommender model on {len(test_users)} test users")
        
        # Calculate metrics
        metrics = self._calculate_hybrid_metrics(hybrid_recommender, test_users, top_n)
        
        return metrics
    
    def _calculate_hybrid_metrics(self, hybrid_recommender: HybridRecommender, 
                                test_users: List[str], top_n: int) -> Dict:
        """Calculate evaluation metrics for the hybrid recommender."""
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
                    # Get recommendations using the hybrid recommender
                    recs = hybrid_recommender.recommend(
                        user_id=user_id, 
                        top_n=top_n, 
                        item_type=item_type
                    )
                    
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

            # Calculate MRR
            mrr = 0
            for rank, item in enumerate(all_user_recs, 1):
                if item in actual_items:
                    mrr = 1 / rank
                    break

            # Calculate NDCG
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

            # Update totals
            total_precision += precision
            total_recall += recall
            total_ndcg += ndcg
            total_hit_rate += hit_rate
            total_mrr += mrr
            all_recommendation_lists.append(all_user_recs)
            
            evaluated_users += 1
            
        # Calculate average metrics
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
    
    def evaluate_hybrid_with_epsilon_greedy(self, hybrid_recommender: HybridRecommender, 
                                         test_users: List[str], epsilon_values: List[float], 
                                         top_n: int = 10) -> pd.DataFrame:
        """
        Evaluate hybrid recommender with epsilon-greedy exploration.
        
        Returns:
            DataFrame with evaluation results for each epsilon value
        """
        results = []
        
        for epsilon in epsilon_values:
            print(f"\nEvaluating hybrid with epsilon-greedy (epsilon={epsilon})")
            
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
                        # Get epsilon-greedy recommendations using the hybrid recommender
                        epsilon_recs = hybrid_recommender.recommend_with_epsilon_greedy(
                            user_id=user_id, 
                            top_n=top_n, 
                            item_type=item_type,
                            epsilon=epsilon
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
    

from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF
from CB_recommendations import ContentBasedRecommender


def main():
    # Create output directory
    output_dir = "eval_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{output_dir}/hybrid_evaluation_log_{timestamp}.txt"
    
    def log_message(message, print_to_console=True):
        """Write message to log file and optionally print to console"""
        with open(log_filename, 'a') as f:
            f.write(f"{message}\n")
        if print_to_console:
            print(message)
    
    log_message(f"=== Hybrid Recommender Evaluation Started at {timestamp} ===")
    
    # Initialize database client
    db_client = Neo4jClient()
    
    # Create evaluator
    evaluator = HybridRecommenderEvaluator(db_client)
    
    # Load data and split into training/test sets
    try:
        log_message("Loading data...")
        evaluator.load_data()
        evaluator.train_test_split(test_size=0.3)
        log_message(f"Data loaded: {len(evaluator.interactions_df)} interactions, {evaluator.interactions_df['user'].nunique()} users")
    except Exception as e:
        log_message(f"Error loading data: {str(e)}")
        db_client.close()
        return
    
    # Initialize CF models with best parameters
    # (Use best parameters found from previous evaluations)
    log_message("Initializing CF models...")
    user_cf_model = UserBasedCF(db_client, k_neighbors=15, min_overlap=2, min_sim=0.05)
    item_cf_model = ItemBasedCF(db_client, k_neighbors=20, min_overlap=1, min_sim=0.04)
    
    # Fit CF models
    try:
        log_message("Fitting User-based CF model...")
        user_cf_model.fit()
        log_message("Fitting Item-based CF model...")
        item_cf_model.fit()
    except Exception as e:
        log_message(f"Error fitting CF models: {str(e)}")
        db_client.close()
        return
    
    # Initialize hybrid recommender
    log_message("Creating hybrid recommender...")
    hybrid_model = HybridRecommender(
        db_client=db_client,
        user_cf_model=user_cf_model,
        item_cf_model=item_cf_model,
        threshold_interactions=5
    )
    
    # Select test users (a subset for faster evaluation)
    log_message("Selecting test users...")
    test_users = evaluator.test_df['user'].unique()[:1000]
    log_message(f"Using {len(test_users)} test users for evaluation")
    
    # Evaluate user-based CF model
    log_message("\nEvaluating User-based CF model...")
    try:
        user_cf_metrics = evaluator._calculate_hybrid_metrics(user_cf_model, test_users, top_n=10)
        log_message("User-based CF metrics:")
        for metric, value in user_cf_metrics.items():
            log_message(f"  {metric}: {value:.4f}")
    except Exception as e:
        log_message(f"Error evaluating User-based CF model: {str(e)}")
    
    # Evaluate item-based CF model
    log_message("\nEvaluating Item-based CF model...")
    try:
        item_cf_metrics = evaluator._calculate_hybrid_metrics(item_cf_model, test_users, top_n=10)
        log_message("Item-based CF metrics:")
        for metric, value in item_cf_metrics.items():
            log_message(f"  {metric}: {value:.4f}")
    except Exception as e:
        log_message(f"Error evaluating Item-based CF model: {str(e)}")
    
    # Evaluate hybrid model
    log_message("\nEvaluating Hybrid model...")
    try:
        hybrid_metrics = evaluator.evaluate_hybrid_model(hybrid_model, test_users, top_n=10)
        log_message("Hybrid model metrics:")
        for metric, value in hybrid_metrics.items():
            log_message(f"  {metric}: {value:.4f}")
    except Exception as e:
        log_message(f"Error evaluating Hybrid model: {str(e)}")
    
    # Evaluate epsilon-greedy on hybrid model
    log_message("\nEvaluating Hybrid model with epsilon-greedy exploration...")
    try:
        epsilon_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        epsilon_results = evaluator.evaluate_hybrid_with_epsilon_greedy(
            hybrid_model, test_users, epsilon_values, top_n=10
        )
        
        log_message("Epsilon-greedy results:")
        log_message(epsilon_results.to_string())
        
        # Save epsilon-greedy results
        epsilon_csv_path = f"{output_dir}/hybrid_epsilon_results_{timestamp}.csv"
        epsilon_results.to_csv(epsilon_csv_path, index=False)
        log_message(f"Epsilon-greedy results saved to {epsilon_csv_path}")
        
        # Find best epsilon value
        best_epsilon_idx = epsilon_results['diversity'].idxmax()
        best_epsilon = epsilon_results.iloc[best_epsilon_idx]
        log_message(f"\nBest epsilon for diversity: {best_epsilon['epsilon']}")
        
        best_epsilon_precision_idx = epsilon_results['precision'].idxmax()
        best_epsilon_precision = epsilon_results.iloc[best_epsilon_precision_idx]
        log_message(f"Best epsilon for precision: {best_epsilon_precision['epsilon']}")
    except Exception as e:
        log_message(f"Error evaluating Hybrid model with epsilon-greedy: {str(e)}")
    
    # Compare models
    log_message("\nComparing all models:")
    try:
        comparison_data = {
            'User-based CF': user_cf_metrics,
            'Item-based CF': item_cf_metrics,
            'Hybrid': hybrid_metrics,
            'Hybrid+Epsilon': epsilon_results.iloc[epsilon_results['precision'].idxmax()].to_dict()
        }
        
        comparison_df = pd.DataFrame(comparison_data).T
        log_message("\nModel comparison:")
        log_message(comparison_df.to_string())
        
        # Save comparison results
        comparison_csv_path = f"{output_dir}/model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_csv_path)
        log_message(f"Model comparison saved to {comparison_csv_path}")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['precision', 'recall', 'ndcg', 'diversity', 'novelty']
        comparison_df[metrics_to_plot].plot(kind='bar')
        plt.title('Recommender Models Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = f"{output_dir}/model_comparison_{timestamp}.png"
        plt.savefig(plot_path)
        log_message(f"Comparison plot saved to {plot_path}")
    except Exception as e:
        log_message(f"Error comparing models: {str(e)}")
    
    # Generate comprehensive report
    try:
        report_path = f"{output_dir}/hybrid_evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write(f"# Hybrid Recommender Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Statistics\n\n")
            f.write(f"- Total interactions: {len(evaluator.interactions_df)}\n")
            f.write(f"- Number of users: {evaluator.interactions_df['user'].nunique()}\n")
            f.write(f"- Number of unique items: {len(evaluator.all_items)}\n")
            f.write(f"- Training set size: {len(evaluator.train_df)}\n")
            f.write(f"- Test set size: {len(evaluator.test_df)}\n\n")
            
            f.write("## Model Comparison\n\n")
            f.write(comparison_df.to_markdown())
            f.write("\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Analyze which model performed best for each metric
            best_precision = comparison_df['precision'].idxmax()
            best_recall = comparison_df['recall'].idxmax()
            best_ndcg = comparison_df['ndcg'].idxmax()
            best_diversity = comparison_df['diversity'].idxmax()
            best_novelty = comparison_df['novelty'].idxmax()
            
            f.write(f"- Best model for precision: **{best_precision}** ({comparison_df.loc[best_precision, 'precision']:.4f})\n")
            f.write(f"- Best model for recall: **{best_recall}** ({comparison_df.loc[best_recall, 'recall']:.4f})\n")
            f.write(f"- Best model for NDCG: **{best_ndcg}** ({comparison_df.loc[best_ndcg, 'ndcg']:.4f})\n")
            f.write(f"- Best model for diversity: **{best_diversity}** ({comparison_df.loc[best_diversity, 'diversity']:.4f})\n")
            f.write(f"- Best model for novelty: **{best_novelty}** ({comparison_df.loc[best_novelty, 'novelty']:.4f})\n\n")
            
            f.write("## Epsilon-Greedy Exploration Impact\n\n")
            f.write("| Epsilon | Precision | Recall | Diversity | Novelty |\n")
            f.write("|---------|-----------|--------|-----------|--------|\n")
            for _, row in epsilon_results.iterrows():
                f.write(f"| {row['epsilon']:.1f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['diversity']:.4f} | {row['novelty']:.4f} |\n")
            
            f.write("\n## Conclusion\n\n")
            
            if hybrid_metrics['precision'] > max(user_cf_metrics['precision'], item_cf_metrics['precision']):
                f.write("The hybrid recommender outperforms both collaborative filtering approaches in terms of precision, ")
            else:
                f.write("While the hybrid recommender doesn't achieve the highest precision, ")
                
            if hybrid_metrics['diversity'] > max(user_cf_metrics['diversity'], item_cf_metrics['diversity']):
                f.write("and also provides more diverse recommendations. ")
            else:
                f.write("though it may need improvements for recommendation diversity. ")
                
            if best_epsilon['epsilon'] > 0:
                f.write(f"\n\nAdding randomness with epsilon={best_epsilon['epsilon']} improves recommendation diversity ")
                f.write(f"from {epsilon_results.iloc[0]['diversity']:.4f} to {best_epsilon['diversity']:.4f}, ")
                
                precision_diff = best_epsilon['precision'] - epsilon_results.iloc[0]['precision']
                if precision_diff >= 0:
                    f.write(f"while also maintaining good precision (change of {precision_diff:.4f}).")
                else:
                    f.write(f"with a small precision trade-off of {abs(precision_diff):.4f}.")
                    
            f.write("\n\nOverall, the evaluation shows that...")
            if comparison_df.loc['Hybrid', 'precision'] > comparison_df.loc['User-based CF', 'precision'] and \
               comparison_df.loc['Hybrid', 'precision'] > comparison_df.loc['Item-based CF', 'precision']:
                f.write(" the hybrid approach successfully combines the strengths of content-based and collaborative filtering methods, ")
                f.write("providing superior recommendations compared to either method alone.")
            else:
                f.write(" while the hybrid approach shows promise in combining content-based and collaborative filtering, ")
                f.write("further tuning is needed to fully leverage the strengths of both methods.")
                
        log_message(f"Comprehensive evaluation report saved to {report_path}")
    except Exception as e:
        log_message(f"Error generating evaluation report: {str(e)}")
    
    # Close database connection
    db_client.close()
    log_message(f"\n=== Hybrid Evaluation Completed at {datetime.now().strftime('%Y%m%d_%H%M%S')} ===")

if __name__ == "__main__":
    main()