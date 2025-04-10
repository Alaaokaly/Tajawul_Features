import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Data_Fetching import Fetch


class CBF:
    def __init__(self, new_user, user_id, limit=None):
        self.new_user = new_user
        self.user_id = user_id
        self.limit = limit
        self.results, self.user_styles = Fetch(new_user=self.new_user,
                                               user_id=self.user_id, limit=self.limit)
        self.results = pd.DataFrame(self.results)
        self.similarity_matrix = None
        self.scores = []
        self._compute_cosine_similarity()

    def _compute_cosine_similarity(self):
        # combine tags for more relevant results
        self.results['combined_text'] = self.results['description'] + " "
        + self.results['tags'].apply(
            lambda tags: " ".join(tags) if tags else ""
        )
        # Compute cosine similarity
        vectorizer = TfidfVectorizer()
        results_matrix = vectorizer.fit_transform(
            self.results["combined_text"])
        self.similarity_matrix = cosine_similarity(results_matrix)
        self.scores = self.similarity_matrix.sum(axis=1)
        # boost the score of the destination/event with matching tags
        matching_tags = self.results['tags'].apply(
            lambda tags: len(set(tags) & set(self.user_styles))
        )
        self.scores += matching_tags * 2
        self.results['score'] = self.scores

    # Diversity-aware reranking
    def MMR_rerank(self, top_n, lambda_=0.7):
        selected = []
        candidates = self.results.index.tolist()
        while len(selected) < top_n and candidates:
            mmr_scores = []
            for candidate in candidates:
                # Relevance
                relevance = self.results.loc[candidate, 'score']

                # Diversity
                if selected:
                    diversity = max(
                        self.similarity_matrix[candidate][selected])
                else:
                    diversity = 0

                # MMR formula
                mmr_score = lambda_ * relevance - (1 - lambda_) * diversity
                mmr_scores.append((candidate, mmr_score))

            # Select the candidate with the highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate = mmr_scores[0][0]
            selected.append(best_candidate)
            candidates.remove(best_candidate)

        return self.results.loc[selected]

    def recommend(self, top_n=10, use_mmr=True, lambda_=0.7):
        if use_mmr:
            ranked_results = self.MMR_rerank(top_n, lambda_=lambda_)
        else:
            ranked_results = self.results.sort_values(
                by='score', ascending=False
            ).head(top_n)
        return ranked_results[['name', 'description', 'tags', 'destinationType', 'type', 'score']]


cbf = CBF(new_user=True, user_id="99ae6489-05d2-49df-bb62-490a2a3f707b", limit=10)
recommendations = cbf.recommend(top_n=5)
print(recommendations)
