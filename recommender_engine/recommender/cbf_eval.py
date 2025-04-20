import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher, InteractionsFetcher
from CB_recommendations import ContentBasedRecommender

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
import time
import random
from collections import defaultdict


class ContentBasedEvaluator:
    """
    Evaluator class for the ContentBasedRecommender system.
    Implements offline evaluation metrics including:
    - Ranking metrics (precision, recall, NDCG)
    - Beyond accuracy metrics (coverage, diversity, novelty)
    - Reranking strategy comparison (MMR vs greedy tag reranking)
    """
    
    def __init__(self, db_client: Neo4jClient):
        self.db_client = db_client
        self.interaction_fetcher = InteractionsFetcher(db_client)
        self.content_fetcher = ContentBasedFetcher(db_client)
        self.interactions_df = None
        self.train_df = None
        self.test_df = None
        self.all_items = None
        self.item_names_map = {}  # Map of (item_id, type) -> name
        
    def load_data(self):
        """Load interaction data from Neo4j"""
        print("Loading interaction data...")
        self.interactions_df = self.interaction_fetcher.fetch_interactions()
        if self.interactions_df.empty:
            raise ValueError("No interaction data found")
        
        # Get list of all unique items for coverage calculation
        self.all_items = self.interactions_df[['item', 'type']].drop_duplicates().values.tolist()
        
        # Create item names map for reference
        for _, row in self.interactions_df.iterrows():
            self.item_names_map[(row['item'], row['type'])] = row['name']
        
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
    
    def evaluate_single_user(self, user_id: str, new_user: bool, top_n: int = 10, 
                            use_mmr: bool = True, lambda_: float = 0.5) -> Dict:
        """Evaluate content-based recommendations for a single user"""
        # Get user's interactions from test set
        user_test = self.test_df[self.test_df['user'] == user_id]
        
        if user_test.empty:
            return None
            
        # Get actual items the user interacted with in the test set
        actual_items = set(tuple(x) for x in user_test[['item', 'type']].values)
        
        # Initialize the content-based recommender
        try:
            recommender = ContentBasedRecommender(
                content_fetcher=self.content_fetcher,
                new_user=new_user,
                user_id=user_id,
                limit=100  # Higher limit to ensure we have enough candidates
            )
        except Exception as e:
            print(f"Error initializing recommender for user {user_id}: {str(e)}")
            return None
        
        metrics = {}
        
        # Get recommendations for each item type and overall
        for type_filter in ['Trip', 'Event', 'Destination', None]:
            type_name = type_filter if type_filter else "All"
            try:
                # Get recommendations
                rec_df = recommender.recommend(
                    top_n=top_n, 
                    use_mmr=use_mmr,
                    type=type_filter,
                    lambda_=lambda_
                )
                
                if rec_df.empty:
                    metrics[f"{type_name}_precision"] = 0
                    metrics[f"{type_name}_recall"] = 0
                    metrics[f"{type_name}_ndcg"] = 0
                    continue
                
                # Extract recommended items
                recommended_items = []
                for _, row in rec_df.iterrows():
                    # Find the item ID for this name and type
                    for (item_id, item_type), name in self.item_names_map.items():
                        if name == row['name'] and item_type == row['type']:
                            recommended_items.append((item_id, item_type))
                            break
                
                # Calculate precision and recall
                recommended_set = set(recommended_items)
                relevant_and_recommended = recommended_set.intersection(actual_items)
                
                precision = len(relevant_and_recommended) / len(recommended_set) if recommended_set else 0
                recall = len(relevant_and_recommended) / len(actual_items) if actual_items else 0
                
                # Calculate NDCG
                ndcg = self._calculate_ndcg(recommended_items, actual_items)
                
                # Store metrics
                metrics[f"{type_name}_precision"] = precision
                metrics[f"{type_name}_recall"] = recall
                metrics[f"{type_name}_ndcg"] = ndcg
            
            except Exception as e:
                print(f"Error getting recommendations for user {user_id}, type {type_filter}: {str(e)}")
                metrics[f"{type_name}_precision"] = 0
                metrics[f"{type_name}_recall"] = 0
                metrics[f"{type_name}_ndcg"] = 0
        
        return metrics
    
    def _calculate_ndcg(self, recommended_items: List[Tuple], actual_items: Set[Tuple]) -> float:
        """Calculate NDCG (Normalized Discounted Cumulative Gain)"""
        # Create relevance vector (1 if item is relevant, 0 otherwise)
        relevance = [1 if item in actual_items else 0 for item in recommended_items]
        
        if not any(relevance):  # No relevant items
            return 0.0
        
        # Calculate DCG
        dcg = 0
        for i, rel in enumerate(relevance, 1):
            dcg += rel / np.log2(i + 1)
        
        # Calculate ideal DCG (IDCG)
        # Sort relevance in descending order for ideal ranking
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0
        for i, rel in enumerate(ideal_relevance, 1):
            idcg += rel / np.log2(i + 1)
        
        # Return NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_recommender(self, test_users: Optional[List[str]] = None, 
                             top_n: int = 10, use_mmr: bool = True, 
                             lambda_: float = 0.5) -> Dict:
        """
        Evaluate content-based recommender on a set of test users
        
        Args:
            test_users: List of user IDs to test on (if None, use all users in test set)
            top_n: Number of recommendations to generate
            use_mmr: Whether to use MMR reranking (True) or greedy tag reranking (False)
            lambda_: Lambda parameter for reranking algorithms
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.train_df is None or self.test_df is None:
            self.train_test_split()
            
        # If no test users specified, use a sample of users in test set
        if test_users is None:
            test_users = self.test_df['user'].unique().tolist()
            # Limit to 50 users for efficiency if there are more
            if len(test_users) > 50:
                test_users = random.sample(test_users, 50)
        
        # Evaluate for new and existing users
        metrics_new = defaultdict(float)
        metrics_existing = defaultdict(float)
        count_new = 0
        count_existing = 0
        
        start_time = time.time()
        
        for i, user_id in enumerate(test_users):
            print(f"Evaluating user {i+1}/{len(test_users)}: {user_id}")
            
            # Evaluate as new user
            metrics = self.evaluate_single_user(
                user_id=user_id,
                new_user=True,
                top_n=top_n,
                use_mmr=use_mmr,
                lambda_=lambda_
            )
            
            if metrics:
                count_new += 1
                for k, v in metrics.items():
                    metrics_new[k] += v
            
            # Evaluate as existing user
            metrics = self.evaluate_single_user(
                user_id=user_id,
                new_user=False,
                top_n=top_n,
                use_mmr=use_mmr,
                lambda_=lambda_
            )
            
            if metrics:
                count_existing += 1
                for k, v in metrics.items():
                    metrics_existing[k] += v
        
        # Calculate averages
        avg_metrics_new = {
            k: v / count_new if count_new > 0 else 0 
            for k, v in metrics_new.items()
        }
        
        avg_metrics_existing = {
            k: v / count_existing if count_existing > 0 else 0 
            for k, v in metrics_existing.items()
        }
        
        evaluation_time = time.time() - start_time
        
        results = {
            'new_user': avg_metrics_new,
            'existing_user': avg_metrics_existing,
            'new_user_count': count_new,
            'existing_user_count': count_existing,
            'evaluation_time': evaluation_time,
            'mmr': use_mmr,
            'lambda': lambda_,
            'top_n': top_n
        }
        
        return results
    
    def compare_reranking_strategies(self, test_users: Optional[List[str]] = None,
                                    top_n: int = 10) -> pd.DataFrame:
        """
        Compare MMR and greedy tag reranking strategies
        
        Args:
            test_users: List of user IDs to test on
            top_n: Number of recommendations to generate
            
        Returns:
            DataFrame with comparison results
        """
        strategies = [
            {'use_mmr': True, 'lambda_': 0.5, 'name': 'MMR (λ=0.5)'},
            {'use_mmr': True, 'lambda_': 0.7, 'name': 'MMR (λ=0.7)'},
            {'use_mmr': True, 'lambda_': 0.3, 'name': 'MMR (λ=0.3)'},
            {'use_mmr': False, 'lambda_': 0.5, 'name': 'Greedy Tag (λ=0.5)'},
            {'use_mmr': False, 'lambda_': 0.7, 'name': 'Greedy Tag (λ=0.7)'},
            {'use_mmr': False, 'lambda_': 0.3, 'name': 'Greedy Tag (λ=0.3)'}
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\nEvaluating strategy: {strategy['name']}")
            metrics = self.evaluate_recommender(
                test_users=test_users,
                top_n=top_n,
                use_mmr=strategy['use_mmr'],
                lambda_=strategy['lambda_']
            )
            
            # Extract key metrics for both user types
            for user_type in ['new_user', 'existing_user']:
                row = {
                    'strategy': strategy['name'],
                    'user_type': user_type,
                    'precision': metrics[user_type].get('All_precision', 0),
                    'recall': metrics[user_type].get('All_recall', 0),
                    'ndcg': metrics[user_type].get('All_ndcg', 0),
                    'evaluation_time': metrics['evaluation_time']
                }
                results.append(row)
        
        return pd.DataFrame(results)
    
    def diversity_analysis(self, test_users: List[str], top_n: int = 10, 
                          use_mmr: bool = True, lambda_: float = 0.5) -> Dict:
        """
        Analyze diversity of recommendations
        
        Args:
            test_users: List of user IDs to analyze
            top_n: Number of recommendations to generate
            use_mmr: Whether to use MMR reranking
            lambda_: Lambda parameter for reranking
            
        Returns:
            Dictionary with diversity metrics
        """
        # Store all recommendations for all users
        all_recommendations = []
        item_coverage = set()
        tag_coverage = set()
        
        for user_id in test_users:
            try:
                # Initialize the recommender
                recommender = ContentBasedRecommender(
                    content_fetcher=self.content_fetcher,
                    new_user=False,  # Existing user for diversity analysis
                    user_id=user_id,
                    limit=100
                )
                
                # Get recommendations
                rec_df = recommender.recommend(
                    top_n=top_n,
                    use_mmr=use_mmr,
                    type=None,  # All types
                    lambda_=lambda_
                )
                
                if not rec_df.empty:
                    # Store recommendations for this user
                    user_recs = []
                    for _, row in rec_df.iterrows():
                        user_recs.append({
                            'name': row['name'],
                            'type': row['type'],
                            'tags': row['tags'] if isinstance(row['tags'], list) else []
                        })
                        
                        # Update item coverage
                        item_coverage.add((row['name'], row['type']))
                        
                        # Update tag coverage
                        if isinstance(row['tags'], list):
                            for tag in row['tags']:
                                tag_coverage.add(tag)
                    
                    all_recommendations.append(user_recs)
            except Exception as e:
                print(f"Error analyzing diversity for user {user_id}: {str(e)}")
                continue
        
        # Calculate diversity metrics
        if not all_recommendations:
            return {
                'inter_user_diversity': 0,
                'intra_list_diversity': 0,
                'item_coverage_ratio': 0,
                'tag_coverage_ratio': 0
            }
        
        # Inter-user diversity: average Jaccard distance between recommendation lists
        inter_user_diversity = self._calculate_inter_user_diversity(all_recommendations)
        
        # Intra-list diversity: average tag diversity within each recommendation list
        intra_list_diversity = self._calculate_intra_list_diversity(all_recommendations)
        
        # Calculate coverage ratios
        # For item coverage, we need the total number of items in the dataset
        total_items = len(self.all_items)
        item_coverage_ratio = len(item_coverage) / total_items if total_items > 0 else 0
        
        # For tag coverage, we would need a reference to all tags in the dataset
        # As a proxy, we'll use the count of unique tags found in recommendations
        total_tags = len(tag_coverage)
        tag_coverage_ratio = 1.0  # Placeholder (can't calculate without knowing total tags)
        
        return {
            'inter_user_diversity': inter_user_diversity,
            'intra_list_diversity': intra_list_diversity,
            'item_coverage_ratio': item_coverage_ratio,
            'tag_coverage': total_tags
        }
    
    def _calculate_inter_user_diversity(self, all_recommendations: List[List[Dict]]) -> float:
        """Calculate average Jaccard distance between user recommendation lists"""
        if len(all_recommendations) <= 1:
            return 0
        
        total_distance = 0
        comparison_count = 0
        
        # Convert recommendations to sets of (name, type) tuples for comparison
        rec_sets = []
        for user_recs in all_recommendations:
            rec_set = {(item['name'], item['type']) for item in user_recs}
            rec_sets.append(rec_set)
        
        # Calculate pairwise Jaccard distances
        for i in range(len(rec_sets)):
            for j in range(i+1, len(rec_sets)):
                set_i = rec_sets[i]
                set_j = rec_sets[j]
                
                union_size = len(set_i.union(set_j))
                intersection_size = len(set_i.intersection(set_j))
                
                # Jaccard distance = 1 - Jaccard similarity
                if union_size > 0:
                    jaccard_distance = 1 - (intersection_size / union_size)
                    total_distance += jaccard_distance
                    comparison_count += 1
        
        # Calculate average
        return total_distance / comparison_count if comparison_count > 0 else 0
    
    def _calculate_intra_list_diversity(self, all_recommendations: List[List[Dict]]) -> float:
        """Calculate average tag diversity within recommendation lists"""
        total_diversity = 0
        list_count = 0
        
        for user_recs in all_recommendations:
            # Skip empty recommendations
            if not user_recs:
                continue
                
            # Count unique tags in this recommendation list
            unique_tags = set()
            total_tags = 0
            
            for item in user_recs:
                if isinstance(item['tags'], list):
                    unique_tags.update(item['tags'])
                    total_tags += len(item['tags'])
            
            # Calculate diversity as ratio of unique tags to total tags
            if total_tags > 0:
                diversity = len(unique_tags) / total_tags
                total_diversity += diversity
                list_count += 1
        
        # Calculate average
        return total_diversity / list_count if list_count > 0 else 0
    
    def visualize_results(self, comparison_df: pd.DataFrame):
        """Visualize comparison results"""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Group by strategy and user_type
        grouped_df = comparison_df.groupby(['strategy', 'user_type'])
        
        # Plot precision
        for (strategy, user_type), group in grouped_df:
            label = f"{strategy} ({user_type})"
            if user_type == 'new_user':
                axes[0, 0].bar(label, group['precision'].values[0], alpha=0.7)
            else:
                axes[0, 1].bar(label, group['precision'].values[0], alpha=0.7)
        
        axes[0, 0].set_title('Precision (New Users)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 1].set_title('Precision (Existing Users)')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot NDCG
        for (strategy, user_type), group in grouped_df:
            label = f"{strategy} ({user_type})"
            if user_type == 'new_user':
                axes[1, 0].bar(label, group['ndcg'].values[0], alpha=0.7)
            else:
                axes[1, 1].bar(label, group['ndcg'].values[0], alpha=0.7)
        
        axes[1, 0].set_title('NDCG (New Users)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 1].set_title('NDCG (Existing Users)')
        axes[1, 1].set_ylim(0, 1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Rotate x-axis labels for better visibility
        for ax in axes.flatten():
            ax.tick_params(axis='x', rotation=45)
        
        return fig


# Example usage
if __name__ == "__main__":
    # Initialize database client
    db_client = Neo4jClient()
    
    # Create evaluator
    evaluator = ContentBasedEvaluator(db_client)
    
    # Load data and split into training/test sets
    evaluator.load_data()
    evaluator.train_test_split(test_size=0.3)
    
    # Define a subset of users to test on (for faster evaluation)
    test_user_ids = [
        '633af53b-f78c-474c-9324-2a734bd86d24',
        '65ab857a-6ff4-493f-aa8d-ddde6463cc20',
        '72effc5b-589a-4076-9be5-f7c3d8533f70',
        '8aaafb9e-0f60-47d1-9b98-1b171564fbf9',
        'b9c32bc3-4b7f-46fd-af3b-ca48060b89a1',
        '3738e035-45a5-4b8b-86a2-32ff64a76f03'
    ]
    
    # Compare reranking strategies
    comparison_results = evaluator.compare_reranking_strategies(test_users=test_user_ids, top_n=10)
    print("\nReranking Strategy Comparison:")
    print(comparison_results)
    
    # Analyze diversity
    diversity_metrics = evaluator.diversity_analysis(test_users=test_user_ids, top_n=10)
    print("\nDiversity Metrics:")
    for metric, value in diversity_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results
    comparison_figure = evaluator.visualize_results(comparison_results)
    
    # Close database connection
    db_client.close()