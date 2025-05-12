import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import InteractionsFetcher, Neo4jClient
import pandas as pd
from scipy.sparse import coo_matrix
from typing import Optional, Any 
import random 

class ItemBasedCF:
    def __init__(self, db_client: Neo4jClient, k_neighbors=5, min_sim=0.1, min_overlap=0):
        self.db_client = db_client
        self.k_neighbors = k_neighbors
        self.min_sim = min_sim
        self.min_overlap = min_overlap
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_columns = None
        self.user_indices = None
        self.item_names_df = None

    def fit(self):
        fetcher = InteractionsFetcher(self.db_client)
        interactions = fetcher.fetch_interactions()
        
        # Debug information
        print("Interactions DataFrame structure:")
        print(f"Columns: {interactions.columns}")
        print(f"First few rows: {interactions.head()}")
        
        # Make sure name column exists
        if 'name' not in interactions.columns:
            print("Warning: 'name' column not found in interactions data.")
            # Add a placeholder name column if missing
            interactions['name'] = interactions.apply(
                lambda row: f"{row['item_type']} {row['item']}", axis=1
            )
            
        # Create item_names_df safely
        try:
            # First, make sure we have all needed columns
            required_columns = ['item', 'name', 'type']
            if 'type' not in interactions.columns and 'item_type' in interactions.columns:
                # Rename item_type to type if needed
                interactions['type'] = interactions['item_type']
                
            # Check if we have all needed columns now
            missing_columns = [col for col in required_columns if col not in interactions.columns]
            if missing_columns:
                print(f"Warning: Missing columns in interactions data: {missing_columns}")
                for col in missing_columns:
                    if col == 'name':
                        interactions['name'] = interactions.apply(
                            lambda row: f"{row.get('item_type', 'Unknown')} {row['item']}", axis=1
                        )
                    elif col == 'type' and 'item_type' in interactions.columns:
                        interactions['type'] = interactions['item_type']
                    else:
                        interactions[col] = "Unknown"
                        
            # Create item_names_df safely
            self.item_names_df = interactions[['item', 'name', 'type']].drop_duplicates(subset=['item', 'type'])
            # Set the index safely
            self.item_names_df = self.item_names_df.set_index(['item', 'type'])
            print(f"Item names DataFrame created with index: {self.item_names_df.index.names}")
        except Exception as e:
            print(f"Error creating item_names_df: {e}")
            # Create a minimal fallback DataFrame
            data = {'item': [], 'type': [], 'name': []}
            self.item_names_df = pd.DataFrame(data).set_index(['item', 'type'])

        if interactions.empty:
            print("Warning: Interaction data is empty. Cannot fit the model.")
            return

        try:
            # Handle pivot_table safely
            if 'user' not in interactions.columns:
                print("Error: 'user' column not found in interactions data.")
                return
                
            if 'item' not in interactions.columns:
                print("Error: 'item' column not found in interactions data.")
                return
                
            if 'type' not in interactions.columns and 'item_type' in interactions.columns:
                # Use item_type instead of type if needed
                pivot_columns = ["item", "item_type"]
            else:
                pivot_columns = ["item", "type"]
                
            # Check for the value column - 'avg', 'weight', or fallback to 'weight' with value 1
            value_column = 'avg' if 'avg' in interactions.columns else 'weight'
            if value_column not in interactions.columns:
                print(f"Warning: '{value_column}' not found, using default weight of 1")
                interactions[value_column] = 1
                
            print(f"Creating pivot table with columns: {pivot_columns}, value: {value_column}")
            
            # Create the pivot table
            self.user_item_matrix = interactions.pivot_table(
                index="user", 
                columns=pivot_columns, 
                values=value_column, 
                fill_value=0, 
                observed=True
            )
            
            print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        except ValueError as e:
            print(f"Error during pivot_table: {e}")
            return
        except Exception as e:
            print(f"Unexpected error during pivot_table: {e}")
            return

        if self.user_item_matrix.empty:
            print("Warning: User-item matrix is empty after pivoting.")
            return

        self.user_indices = self.user_item_matrix.index
        self.item_columns = self.user_item_matrix.columns

        matrix = self.user_item_matrix.T  # Transpose to make items as rows
        item_sparse = csr_matrix(matrix.values)

        # Calculate similarity
        print("Calculating item similarity...")
        similarity = cosine_similarity(item_sparse, dense_output=False)
        similarity = similarity.multiply(similarity > self.min_sim)

        # Calculate overlap
        binarized = item_sparse.copy()
        binarized.data = np.ones_like(binarized.data)
        overlap = binarized.dot(binarized.T)
        self.item_similarity = similarity.multiply(overlap > self.min_overlap)
        print("Item similarity matrix created.")

    def recommend(self, user_id, top_n=5, item_type=None):
        if self.item_similarity is None or self.user_item_matrix is None:
            print(f"Model not fitted properly for user: {user_id}")
            return pd.DataFrame(columns=['rank', 'item', 'type', 'name', 'score'])

        if user_id not in self.user_indices:
            print(f"User ID '{user_id}' not found.")
            return pd.DataFrame(columns=['rank', 'item', 'type', 'name', 'score'])

        try:
            user_vector = self.user_item_matrix.loc[user_id]
            scores = pd.Series(0, index=self.item_columns, dtype=float)

            for item_idx, rating in user_vector[user_vector > 0].items():
                try:
                    item_index = self.item_columns.get_loc(item_idx)
                    similar_items_scores = self.item_similarity[item_index].toarray().flatten()
                    scores += rating * pd.Series(similar_items_scores, index=self.item_columns)
                except Exception as e:
                    print(f"Error processing item {item_idx}: {e}")
                    continue

            # Filter already interacted
            interacted_items = user_vector[user_vector > 0].index
            scores = scores.drop(interacted_items, errors='ignore')

            if item_type:
                # Handle different multi-index structures
                if isinstance(self.item_columns, pd.MultiIndex) and len(self.item_columns.names) > 1:
                    type_level = 'type' if 'type' in self.item_columns.names else 'item_type'
                    if type_level in self.item_columns.names:
                        scores = scores[scores.index.get_level_values(type_level) == item_type]
                    else:
                        print(f"Warning: Cannot filter by type. Available levels: {self.item_columns.names}")
                else:
                    print(f"Warning: Item columns is not a multi-index: {type(self.item_columns)}")

            top_items = scores.nlargest(top_n)
            recommendations = []

            for rank, item_tuple in enumerate(top_items.index, start=1):
                try:
                    # Handle both tuple and non-tuple indices
                    if isinstance(item_tuple, tuple):
                        item_id, item_type_rec = item_tuple
                    else:
                        # For non-multiindex case
                        item_id = item_tuple
                        item_type_rec = item_type or 'unknown'
                    
                    # Try to get the name, with fallback
                    try:
                        item_info = self.item_names_df.loc[(item_id, item_type_rec)]
                        name = item_info['name']
                    except KeyError:
                        name = f"Unknown {item_type_rec} ({item_id})"
                    except Exception as e:
                        print(f"Error retrieving name for {(item_id, item_type_rec)}: {e}")
                        name = f"Unknown {item_type_rec} ({item_id})"
                        
                    recommendations.append({
                        'rank': rank,
                        'item': item_id,
                        'type': item_type_rec,
                        'name': name,
                        'score': top_items[item_tuple] if isinstance(item_tuple, tuple) else top_items[item_id]
                    })
                except Exception as e:
                    print(f"Error processing recommendation for {item_tuple}: {e}")
                    continue

            return pd.DataFrame(recommendations)[['rank', 'item', 'type', 'name', 'score']]
            
        except Exception as e:
            print(f"Error in recommend method: {e}")
            return pd.DataFrame(columns=['rank', 'item', 'type', 'name', 'score'])

    def recommend_with_epsilon_greedy(self, user_id, top_candidates_df: pd.DataFrame, item_type: Optional[str] = None, epsilon: float = 0.2, top_n: int = 10):
        if self.user_item_matrix is None or self.item_names_df is None:
            print("Model must be fitted first.")
            return pd.DataFrame()
        
        try:
            # Get all items of the given type
            if isinstance(self.item_names_df.index, pd.MultiIndex):
                type_level = 1  # Assuming type is the second level in the index
                all_items_of_type = self.item_names_df[
                    self.item_names_df.index.get_level_values(type_level) == item_type
                ]
            else:
                print("Warning: item_names_df does not have a multi-index. Cannot filter by type.")
                all_items_of_type = self.item_names_df
                
            # Get items user already interacted with
            user_vector = self.user_item_matrix.loc[user_id]
            interacted_items = user_vector[user_vector > 0].index
            
            # Filter unseen items
            unseen_items = []
            if isinstance(self.item_columns, pd.MultiIndex):
                # For multi-index columns
                for idx in all_items_of_type.index:
                    if idx not in interacted_items:
                        unseen_items.append(idx)
            else:
                # Fallback for non-multi-index
                for idx in all_items_of_type.index:
                    unseen_items.append(idx)
            
            # Choose explore/exploit split
            explore_count = int(top_n * epsilon)
            exploit_count = top_n - explore_count
            
            # Make sure we don't exceed available exploits
            exploit_count = min(exploit_count, len(top_candidates_df))
            exploit_items = top_candidates_df.head(exploit_count).to_dict("records")
            
            # Make sure we don't exceed available explores
            explore_count = min(explore_count, len(unseen_items))
            if explore_count > 0 and unseen_items:
                explore_samples = random.sample(unseen_items, explore_count)
                
                explore_items = []
                for rank, item_idx in enumerate(explore_samples, start=1):
                    try:
                        # Handle both tuple and non-tuple indices
                        if isinstance(item_idx, tuple):
                            item_id, item_type_val = item_idx
                        else:
                            item_id = item_idx
                            item_type_val = item_type or 'unknown'
                        
                        # Try to get the name from item_names_df
                        try:
                            name = self.item_names_df.loc[item_idx]['name']
                        except KeyError:
                            name = f"Unknown {item_type_val} ({item_id})"
                        except Exception as e:
                            print(f"Error retrieving name for {item_idx}: {e}")
                            name = f"Unknown {item_type_val} ({item_id})"
                            
                        explore_items.append({
                            'rank': None,
                            'item': item_id,
                            'type': item_type_val,
                            'name': name,
                            'score': None  # score is unknown
                        })
                    except Exception as e:
                        print(f"Error processing explore item {item_idx}: {e}")
                        continue
            else:
                explore_items = []
                
            # Merge & shuffle
            all_items = exploit_items + explore_items
            random.shuffle(all_items)
            
            # Re-rank and return
            for i, item in enumerate(all_items, 1):
                item['rank'] = i
                
            return pd.DataFrame(all_items)
            
        except Exception as e:
            print(f"Error in recommend_with_epsilon_greedy: {e}")
            return pd.DataFrame()
if __name__ == "__main__":
    db_client = Neo4jClient()
    model = ItemBasedCF(db_client, k_neighbors=25, min_sim=0.001, min_overlap=0)
    model.fit()
    
    
    user_id_to_recommend = [
        '633af53b-f78c-474c-9324-2a734bd86d24',
        '65ab857a-6ff4-493f-aa8d-ddde6463cc20',
        '72effc5b-589a-4076-9be5-f7c3d8533f70',
        '8aaafb9e-0f60-47d1-9b98-1b171564fbf9',
        'b9c32bc3-4b7f-46fd-af3b-ca48060b89a1',
        '3738e035-45a5-4b8b-86a2-32ff64a76f03',
        '82f642dc-fda0-46ed-b080-f4b1866899a6',
        'b99c49fc-f7b1-4cd4-8d22-cc5b8575f07f',
        '3989ed58-1cce-45e2-9b5b-e4827165e324'
    ]

    for user_id in user_id_to_recommend:
        for item_type in ['Trip', 'Event', 'Destination', 'Post']:
            top_items = model.recommend(user_id, top_n=10, item_type=item_type)
            print(f"Top {item_type} recommendations for user {user_id}:")
            print(top_items[['score','name']])

            epsilon_recs = model.recommend_with_epsilon_greedy(
                user_id,
                top_candidates_df=top_items,
                item_type=item_type,
                epsilon=0.2
            )
            print(f"Epsilon-Greedy {item_type} recommendations for user {user_id}:")
            print(epsilon_recs[['score','name']])

    db_client.close()
