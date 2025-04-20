import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from typing import Optional, List, Union


class ContentBasedRecommender:
    def __init__(self,
                 content_fetcher: ContentBasedFetcher,  # Inject the fetcher
                 new_user,
                 user_id,
                 limit: Optional[int] = None):
        self.new_user = new_user
        self.user_id = user_id
        self.limit = limit
        self.content_fetcher = content_fetcher
        self.similarity_matrix = None
        self.results = []
        self.user_styles = []
        self.user_item_matrix = None

        if self.new_user:
            self.results, self.user_styles = self.content_fetcher.fetch_new_user_data(
                new_user=self.new_user, user_id=self.user_id, limit=self.limit
            )
        else:
            self.results, self.user_styles = self.content_fetcher.fetch_existing_user_data(
                user_id=self.user_id, limit=self.limit
            )

        if self.results:
            self._compute_cosine_similarity()
        else:
            self.scores = pd.Series()

    def _compute_cosine_similarity(self):
        # combine tags for more relevant results
        df_results = pd.DataFrame(self.results)
        df_results['combined_text'] = df_results['description'].fillna('') + " " + df_results['tags'].apply(
            lambda tags: " ".join(tags) if isinstance(tags, list) else ""
        )
        self.user_item_matrix = df_results.pivot_table(index="user", columns=["item", "type"], values="avg", fill_value=0, observed=True)

        # Compute cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        results_matrix = vectorizer.fit_transform(
            df_results["combined_text"])
        self.similarity_matrix = cosine_similarity(results_matrix)
        self.scores = pd.Series(self.similarity_matrix.sum(
            axis=1), index=df_results.index)
        # boost the score of the destination/event with matching tags
        matching_tags = df_results['tags'].apply(
            lambda tags: len(set(tuple(tags)) & set(tuple(style)
                             for style in self.user_styles))
            if isinstance(tags, list) and isinstance(self.user_styles, list) else 0
        )
        self.scores += matching_tags * 2
        max_score = self.scores.max()
        if max_score > 0:
            self.scores = self.scores / max_score
        # Add the score back to the list of dictionaries
        self.results = [
            {**item, 'score': self.scores.get(idx, 0)}
            for idx, item in pd.DataFrame(self.results).iterrows()
        ]

    # Diversity-aware reranking
    def MMR_rerank(self, top_n, lambda_=0.7):
        if not self.results or self.similarity_matrix is None:
            return []

        # Create a map from name to full item and index
        name_to_index = {item['name']: idx for idx,
                         item in enumerate(self.results)}
        name_to_item = {item['name']: item for item in self.results}
        candidates = list(name_to_index.keys())

        selected = []

        while len(selected) < top_n and candidates:
            mmr_scores = []
            for candidate in candidates:
                candidate_index = name_to_index[candidate]
                relevance = name_to_item[candidate].get('score', 0)

                max_similarity = 0
                if selected:
                    for sel in selected:
                        if sel in name_to_index:
                            sel_index = name_to_index[sel]
                            sim = self.similarity_matrix[candidate_index][sel_index]
                            max_similarity = max(max_similarity, sim)

                mmr_score = lambda_ * relevance - \
                    (1 - lambda_) * max_similarity
                mmr_scores.append((candidate, mmr_score))

            # Sort candidates by MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            selected.append(best_candidate)
            candidates.remove(best_candidate)

        # Return re-ranked results in the MMR-selected order
        reranked_results = [name_to_item[name] for name in selected]
        return reranked_results

    def greedy_tag_rerank(self, top_n, lambda_=0.7):
        if not self.results:
            return []

        # Sort by relevance (score) first
        sorted_results = sorted(
            self.results, key=lambda x: x.get('score', 0), reverse=True)

        selected = []
        selected_tags = set()
        for item in sorted_results:
            item_tags = set(item['tags'])
            if not item_tags & selected_tags:  # Ensure no tag overlap
                selected.append(item)
                selected_tags.update(item_tags)

            if len(selected) >= top_n:
                break

        # Apply the lambda weighting
        for item in selected:
            item['score'] = lambda_ * item.get('score', 0)

        return selected

    def recommend(self, top_n=10, use_mmr=True, type=None, lambda_=0.5):
        if not self.results:
            return pd.DataFrame(columns=['name', 'description', 'tags', 'destinationType', 'type', 'score'])
        
        # First filter by type if specified
        filtered_results = self.results
        if type is not None:
            # Convert to list if it's a single string
            if isinstance(type, str):
                type = [type]
            
            # Case-insensitive filtering
            filtered_results = [
                item for item in self.results 
                if item.get('type', '').lower() in [t.lower() for t in type]
            ]
            
            # Check if we have any results after filtering
            if not filtered_results:
                print(f"No items found matching specified type(s): {type}")
                return pd.DataFrame(columns=['name', 'description', 'tags', 'destinationType', 'type', 'score'])
        
        # Store original results
        original_results = self.results
        
        # Temporarily replace self.results with filtered results for reranking
        self.results = filtered_results
        
        # Apply reranking on filtered results
        if use_mmr:
            ranked_results = self.MMR_rerank(top_n, lambda_=lambda_)
        else:
            ranked_results = self.greedy_tag_rerank(top_n, lambda_=lambda_)
        
        # Restore original results
        self.results = original_results
        
        # Create DataFrame from ranked results
        df_ranked = pd.DataFrame(ranked_results)
        
        # Ensure all needed columns exist
        for col in ['name', 'description', 'tags', 'destinationType', 'type', 'score']:
            if col not in df_ranked.columns:
                df_ranked[col] = None
                
        return df_ranked[['name', 'description', 'tags', 'destinationType', 'type', 'score']]


if __name__ == '__main__':
    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)
    cbf = ContentBasedRecommender(
        content_fetcher,
        new_user=True,
        user_id='65ab857a-6ff4-493f-aa8d-ddde6463cc20',
        limit=100
    )
    
    # Print all available types for debugging
    if cbf.results:
        types = set(item.get('type', '') for item in cbf.results)
        print(f"Available types in the dataset: {types}")
    
    # Get recommendations filtered by destination type
    destination_recommendations = cbf.recommend(top_n=15, use_mmr=True, type='Destination')
    print("\nDESTINATION RECOMMENDATIONS:")
    print(destination_recommendations[['name', 'type', 'score']])
    
    # Get recommendations filtered by event type
    event_recommendations = cbf.recommend(top_n=15, use_mmr=True, type='Event')
    print("\nEVENT RECOMMENDATIONS:")
    print(event_recommendations[['name', 'type', 'score']])
    
    # Get all recommendations (without type filtering)
    all_recommendations = cbf.recommend(top_n=15, use_mmr=True)
    print("\nALL RECOMMENDATIONS:")
    print(all_recommendations[['name', 'type', 'score']])
    
    db_client.close()