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
                 limit: Optional[int] = None):
        self.new_user = new_user
        self.user_id = user_id
        self.limit = limit
        self.content_fetcher = content_fetcher
        self.similarity_matrix = None
        self.results = []
        self.user_styles = []

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
        # Compute cosine similarity
        vectorizer = TfidfVectorizer()
        results_matrix = vectorizer.fit_transform(
            df_results["combined_text"])
        self.similarity_matrix = cosine_similarity(results_matrix)
        self.scores = pd.Series(self.similarity_matrix.sum(axis=1), index=df_results.index)
        # boost the score of the destination/event with matching tags
        matching_tags = df_results['tags'].apply(
            lambda tags: len(set(tuple(tags)) & set(tuple(style) for style in self.user_styles))
            if isinstance(tags, list) and isinstance(self.user_styles, list) else 0
        )
        self.scores += matching_tags * 2
        # Add the score back to the list of dictionaries
        self.results = [
            {**item, 'score': self.scores.get(idx, 0)}
            for idx, item in pd.DataFrame(self.results).iterrows()
        ]

    # Diversity-aware reranking
    def MMR_rerank(self, top_n, lambda_=0.7):
        if not self.results:
            return []

        selected = []
        candidates = [item['name'] for item in self.results]
        original_indices = {item['name']: i for i, item in enumerate(self.results)}

        while len(selected) < top_n and candidates:
            mmr_scores = []
            for candidate in candidates:
                candidate_index = original_indices[candidate]
                relevance = self.results[candidate_index].get('score', 0)

                diversity = 0
                if selected and self.similarity_matrix is not None and candidate in original_indices:
                    candidate_row_index = original_indices[candidate]
                    max_similarity = 0
                    for sel in selected:
                        if sel in original_indices:
                            selected_row_index = original_indices[sel]
                            similarity = self.similarity_matrix[candidate_row_index][selected_row_index]
                            max_similarity = max(max_similarity, similarity)
                    diversity = max_similarity

                mmr_score = lambda_ * relevance - (1 - lambda_) * diversity
                mmr_scores.append((candidate, mmr_score))

            if mmr_scores:
                mmr_scores.sort(key=lambda x: x[1], reverse=True)
                best_candidate = mmr_scores[0][0]
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break

        return [item for item in self.results if item['name'] in selected]

    def recommend(self, top_n=10, use_mmr=True, lambda_=0.7):
        if not self.results:
            return pd.DataFrame(columns=['name', 'description', 'tags', 'destinationType', 'type', 'score'])

        if use_mmr:
            ranked_results = self.MMR_rerank(top_n, lambda_=lambda_)
        else:
            ranked_results = sorted(self.results, key=lambda x: x.get('score', 0), reverse=True)[:top_n]

        df_ranked = pd.DataFrame(ranked_results)
        return df_ranked[['name', 'description', 'tags', 'destinationType', 'type', 'score']]


if __name__ == '__main__':
    db_client = Neo4jClient()
    content_fetcher = ContentBasedFetcher(db_client)
    cbf = ContentBasedRecommender(
        content_fetcher,
        new_user=True,
        user_id="99ae6489-05d2-49df-bb62-490a2a3f707b",
        limit=10
    )
    recommendations = cbf.recommend(top_n=115)
    print(recommendations)
    db_client.close()