import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from typing import Optional


class ContentBasedRecommender:
    def __init__(self,
                 content_fetcher: ContentBasedFetcher,  # Inject the fetcher
                 new_user,
                 user_id,
                 item_type: Optional[str] = None,
                 limit: Optional[int] = None):
        self.new_user = new_user
        self.user_id = user_id
        self.limit = limit
        self.item_type = item_type
        self.content_fetcher = content_fetcher
        self.similarity_matrix = None
        self.results = []
        self.user_styles = []

        if self.new_user:
            self.results, self.user_styles = self.content_fetcher.fetch_new_user_data(
                new_user=self.new_user, user_id=self.user_id
            )
        else:
            self.results, self.user_styles = self.content_fetcher.fetch_existing_user_data(
                user_id=self.user_id
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
        name_to_index = {item['item']: idx for idx,
                         item in enumerate(self.results)}
        name_to_item = {item['item']: item for item in self.results}
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
        reranked_results = [name_to_item[item] for item in selected]
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

    def recommend(self, top_n=10, use_mmr=True, lambda_=0.5, item_type=None):
        if not self.results:
            return pd.DataFrame(columns=['user', 'item', 'item_type', 'score'])

        if item_type:
            filtered_results = [
                item for item in self.results if item.get('item_type') == item_type]
        else:
            filtered_results = self.results
        item_type = item_type or self.item_type

        if use_mmr:
            self.results = filtered_results
            ranked_results = self.MMR_rerank(top_n, lambda_=lambda_)
        else:
            self.results = filtered_results
            ranked_results = self.greedy_tag_rerank(top_n, lambda_=lambda_)
            for r in ranked_results:
                r['user'] = r.get('user', self.user_id)
                r['item'] = r.get('item', r.get(
                    'id', r.get('name')))  # fallback
                r['item_type'] = r.get('item_type', item_type)
                r['score'] = r.get('score', 0.0)

        df_ranked = pd.DataFrame(ranked_results)
        return df_ranked[['user', 'item', 'item_type', 'score']]


if __name__ == '__main__':
    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)
    cbf = ContentBasedRecommender(
        content_fetcher,
        new_user=True,
        user_id="43c2b772-c135-4791-b5fd-d073a106217b",
    )
    print("\nALL RECOMMENDATIONS:")
    All_recommendations = cbf.recommend(top_n=200, use_mmr=True)
    print(All_recommendations)
    db_client.close()
