import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher # Assuming these are correctly defined
from typing import Optional, List, Any, Set, Dict, Union
import numpy as np 

class ContentBasedRecommender:
    def __init__(self,
                 content_fetcher: ContentBasedFetcher,
                 new_user: bool,
                 user_id: str,
                 limit: Optional[int] = None):
        self.new_user = new_user
        self.user_id = user_id
        self.limit = limit
        self.content_fetcher = content_fetcher
        self.similarity_matrix: Optional[np.ndarray] = None # Item-Item similarity matrix (TF-IDF based)
        self.results: List[Dict[str, Any]] = [] # Candidate items list [{item_id:x, name:y, ...}, ...]
        self.user_styles: List[Any] = [] # User profile features (e.g., list of tags/genres from onboarding or history)
        self.scores: Optional[pd.Series] = None # Series holding scores for items in df_results index
        # self.user_id_to_index = {} # Not used within this class logic

        print(f"CB Init: User {self.user_id}, New User: {self.new_user}")
        try:
            # Fetch data based on user status
            if self.new_user:
                # Fetches candidate items and onboarding styles/tags for the user
                self.results, self.user_styles = self.content_fetcher.fetch_new_user_data(
                    new_user=self.new_user, user_id=self.user_id, limit=self.limit
                )
                print(f"CB Init: Fetched {len(self.results)} candidate items and {len(self.user_styles)} user styles (new user).")
            else:
                # Fetches candidate items and derives user styles from interaction history
                self.results, self.user_styles = self.content_fetcher.fetch_existing_user_data(
                    user_id=self.user_id, limit=self.limit
                )
                print(f"CB Init: Fetched {len(self.results)} candidate items and {len(self.user_styles)} user styles (existing user).")

            # Compute scores if candidate items were found
            if self.results and isinstance(self.results, list):
                 # Check if results have content needed for TF-IDF
                 if any(item.get('description') or item.get('tags') for item in self.results):
                      self._compute_content_scores() # Renamed for clarity
                 else:
                      print("Warning: Fetched items lack description/tags for TF-IDF scoring.")
                      self._assign_default_scores() # Assign default scores if no content
            else:
                 print("Warning: No candidate items fetched or results format invalid.")
                 self._assign_default_scores()

        except Exception as e:
            print(f"Error during CB initialization or data fetching for user {self.user_id}: {e}")
            self.results = []
            self._assign_default_scores()

    def _assign_default_scores(self):
        """Assigns default scores if main scoring fails or no data."""
        if isinstance(self.results, list) and self.results:
             try:
                 df_results = pd.DataFrame(self.results)
                 self.scores = pd.Series(0.0, index=df_results.index) # Default score 0
                 df_results['score'] = self.scores.values
                 self.results = df_results.to_dict('records')
             except Exception as e:
                  print(f"Error creating DataFrame for default scores: {e}")
                  # Ensure results remains a list, even if scoring fails
                  self.results = [] if not isinstance(self.results, list) else self.results
                  self.scores = pd.Series(dtype=float)
        else:
            self.results = []
            self.scores = pd.Series(dtype=float)

    def _make_hashable_set(self, input_list: List[Any]) -> Set[Union[str, int, float, tuple]]:
        """Converts a list (potentially with nested lists/unhashables) into a set of hashable items."""
        hashable_set = set()
        if not isinstance(input_list, list):
            return hashable_set # Return empty set if input is not a list

        for item in input_list:
            if isinstance(item, list):
                # Recursively handle nested lists or simply iterate one level
                for sub_item in item:
                    if isinstance(sub_item, (str, int, float)): # Add common hashable types
                        hashable_set.add(sub_item)
                    # Add other hashable types if needed (e.g., tuple)
                    # Ignore unhashable types like dicts or other lists within sub-lists
            elif isinstance(item, (str, int, float)):
                hashable_set.add(item)
            # Ignore other unhashable types
        return hashable_set

    def _compute_content_scores(self):
        """Computes item scores based on TF-IDF content similarity and tag matching."""
        if not isinstance(self.results, list) or not self.results:
             print("Warning: Cannot compute scores, self.results is invalid.")
             self._assign_default_scores()
             return

        try:
            df_results = pd.DataFrame(self.results)
            # Ensure essential columns exist, provide defaults
            if 'description' not in df_results.columns: df_results['description'] = ''
            if 'tags' not in df_results.columns: df_results['tags'] = [[] for _ in range(len(df_results))] # Default empty list

            # Handle NaN/None before string operations
            df_results['description'].fillna('', inplace=True)
            # Ensure tags are lists before processing
            df_results['tags'] = df_results['tags'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [str(x)]))

            # Create combined text for TF-IDF
            df_results['combined_text'] = df_results['description'] + " " + df_results['tags'].apply(
                 lambda tags: " ".join(map(str, tags)) # Convert all tags to string just in case
            )
        except Exception as e:
            print(f"Error creating DataFrame or combined_text in _compute_content_scores: {e}")
            self._assign_default_scores()
            return

        # --- Compute TF-IDF and Item-Item Similarity ---
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) # Added common params
            results_matrix = vectorizer.fit_transform(df_results["combined_text"])
            # Store Item-Item similarity if needed (e.g., for MMR)
            self.similarity_matrix = cosine_similarity(results_matrix)
        except Exception as e:
             print(f"Error computing TF-IDF or item similarity: {e}")
             self.similarity_matrix = None # Ensure it's None if failed
             # Cannot proceed without results_matrix if scoring relies on profile vector below
             self._assign_default_scores() # Assign defaults as we can't reliably score
             return

        # --- Calculate Item Scores relative to User Profile/Styles ---
        # Initialize scores
        final_scores = pd.Series(0.0, index=df_results.index)

        # 1. Optional: Score based on similarity to user's interaction profile vector
        # (Requires fetcher to provide interacted items' content, not just styles)
        # Example logic if `self.interacted_items_content` was populated:
        # interacted_texts = [...]
        # user_profile_vector = vectorizer.transform([" ".join(interacted_texts)])
        # sim_to_profile = cosine_similarity(user_profile_vector, results_matrix)
        # base_scores = pd.Series(sim_to_profile.flatten(), index=df_results.index)
        # final_scores += base_scores # Add this component if calculated

        # 2. Boost score based on matching user_styles (tags/genres)
        # Ensure user_styles is processed correctly
        try:
            valid_user_styles_set = self._make_hashable_set(self.user_styles) # Use helper
            if valid_user_styles_set:
                print(f"CB Scoring: Valid user styles for matching: {valid_user_styles_set}")
                def calculate_theme_match(item_tags_list):
                    item_tags_set = self._make_hashable_set(item_tags_list) # Use helper
                    return len(item_tags_set & valid_user_styles_set)

                theme_matches = df_results['tags'].apply(calculate_theme_match)
                boost_factor = 2.0 # Tunable boost factor
                theme_boost = theme_matches * boost_factor
                print(f"CB Scoring: Applying theme boost (max boost value: {theme_boost.max()})")
                final_scores += theme_boost
            else:
                print("CB Scoring: No valid user styles found for boosting.")

        except Exception as e:
             print(f"Error calculating tag matching boost: {e}. Skipping boost.")
             # Continue without the boost if it fails

        # Ensure scores are numeric and assign
        self.scores = pd.to_numeric(final_scores, errors='coerce').fillna(0.0)

        # Add the final score back to the list of dictionaries format
        df_results['score'] = self.scores.values
        self.results = df_results.to_dict('records')

    # MMR_rerank method (requires careful checking of indices against potentially modified results)
    def MMR_rerank(self, top_n, lambda_=0.7):
        if not isinstance(self.results, list) or not self.results or self.similarity_matrix is None:
             print("MMR Warning: Invalid inputs (results list or similarity matrix missing).")
             return []

        # Filter items with positive score only for candidate pool
        # IMPORTANT: Ensure score is numeric BEFORE filtering
        scored_results = []
        for item in self.results:
            score = item.get('score', 0)
            if isinstance(score, (int, float)) and score > 0:
                 scored_results.append(item)
            elif isinstance(score, (int, float)) and score <= 0:
                 pass # Skip items with zero or negative score
            else:
                 print(f"MMR Warning: Invalid score type '{type(score)}' for item '{item.get('name', 'N/A')}'. Skipping.")

        if not scored_results:
             print("MMR Warning: No items with positive scores found for reranking.")
             return []

        # Map names to index *within the filtered scored_results list*
        try:
             name_to_filtered_idx = {item['name']: i for i, item in enumerate(scored_results)}
             candidate_names = [item['name'] for item in scored_results]
        except KeyError:
             print("MMR Error: 'name' key missing from scored results.")
             return []

        selected_names = []
        while len(selected_names) < top_n and candidate_names:
            mmr_scores = []
            for name in candidate_names:
                try:
                    candidate_filtered_idx = name_to_filtered_idx[name]
                    relevance = scored_results[candidate_filtered_idx].get('score', 0)

                    # Calculate max similarity to already selected items
                    max_sim_to_selected = 0.0
                    if selected_names:
                        for selected_name in selected_names:
                            if selected_name in name_to_filtered_idx:
                                selected_filtered_idx = name_to_filtered_idx[selected_name]
                                # Indices here are for the FILTERED list, which match the similarity matrix dimensions IF df_results order was preserved
                                if candidate_filtered_idx < self.similarity_matrix.shape[0] and \
                                   selected_filtered_idx < self.similarity_matrix.shape[1]:
                                    similarity = self.similarity_matrix[candidate_filtered_idx][selected_filtered_idx]
                                    max_sim_to_selected = max(max_sim_to_selected, similarity)

                    # MMR formula
                    mmr_score = (lambda_ * relevance) - ((1 - lambda_) * max_sim_to_selected)
                    mmr_scores.append((name, mmr_score))
                except Exception as e:
                    print(f"MMR Error calculating score for '{name}': {e}")

            if not mmr_scores: break # No scores calculated

            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate_name = mmr_scores[0][0]
            selected_names.append(best_candidate_name)
            candidate_names.remove(best_candidate_name) # Remove from candidates

        # Return full item dicts for selected names in selection order
        final_items_map = {item['name']: item for item in scored_results}
        return [final_items_map[name] for name in selected_names if name in final_items_map]


    def recommend(self, top_n=10, use_mmr=True, lambda_=0.7):
        """Generates final recommendations, optionally using MMR."""
        if not isinstance(self.results, list):
             self.results = []

        if not self.results:
             print(f"CB Recommend: No results available for user {self.user_id}.")
             return pd.DataFrame(columns=['item', 'item_type', 'name', 'description', 'tags', 'destinationType', 'score'])

        # Get ranked list (either MMR or simple sort)
        if use_mmr:
            print(f"CB Recommend: Using MMR reranking for user {self.user_id}")
            ranked_results_list = self.MMR_rerank(top_n, lambda_=lambda_)
        else:
            print(f"CB Recommend: Using simple score ranking for user {self.user_id}")
            # Filter for numeric positive scores before sorting
            results_with_numeric_scores = []
            for item in self.results:
                 score = item.get('score', 0)
                 if isinstance(score, (int, float)) and score > 0:
                      results_with_numeric_scores.append(item)
            ranked_results_list = sorted(results_with_numeric_scores, key=lambda x: x.get('score', 0), reverse=True)[:top_n]

        if not ranked_results_list: # Check if ranking yielded results
             print(f"CB Recommend: No ranked results found for user {self.user_id}")
             return pd.DataFrame(columns=['item', 'item_type', 'name', 'description', 'tags', 'destinationType', 'score'])

        # Convert final list to DataFrame
        df_ranked = pd.DataFrame(ranked_results_list)

        # --- Ensure standard output columns ---
        if 'type' in df_ranked.columns and 'item_type' not in df_ranked.columns:
             df_ranked = df_ranked.rename(columns={'type': 'item_type'})

        # Add missing essential columns with defaults
        if 'item' not in df_ranked.columns: df_ranked['item'] = "Unknown ID" # Ideally map this!
        if 'item_type' not in df_ranked.columns: df_ranked['item_type'] = "Unknown Type"
        if 'name' not in df_ranked.columns: df_ranked['name'] = "Unknown Name"
        if 'score' not in df_ranked.columns: df_ranked['score'] = 0.0

        # Ensure score is numeric before returning
        df_ranked['score'] = pd.to_numeric(df_ranked['score'], errors='coerce').fillna(0.0)

        # Select and order output columns (adjust as needed)
        output_cols = ['item', 'item_type', 'name', 'score'] # Core columns
        for col in ['description', 'tags', 'destinationType']: # Optional descriptive cols
             if col in df_ranked.columns: output_cols.append(col)

        # Handle cases where core columns might still be missing after additions
        final_output_cols = [col for col in output_cols if col in df_ranked.columns]

        return df_ranked[final_output_cols]

# --- Main block remains the same ---
if __name__ == '__main__':
    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)
    cbf = ContentBasedRecommender(
        content_fetcher,
        new_user=False, # Test with existing user assumed to have styles/history
        user_id="4bf0b634-076d-4d9e-9679-b83fdcaabf81", # Example User
        limit=100
    )
    recommendations = cbf.recommend(top_n=10, use_mmr=True) # Use MMR
    print("Content-Based Recommendations:")
    print(recommendations)
    db_client.close()