import logging
from recommender import ContentBasedRecommender, UserBasedCF, HybridRecommender
from recommender.collaborative_filtering import fetch_interactions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the Recommender System...")

    # Step 1: Fetch interactions from Neo4j and train CF model
    logger.info("Fetching user interactions from database...")
    interactions = fetch_interactions()


    # Step 2: Load Content-Based and Collaborative Filtering Models
    cb_model = ContentBasedRecommender()
    cf_model = UserBasedCF()
     
    logger.info("Training Content-Based Recommender...")
    cb_model.fit()


    logger.info("Training Collaborative Filtering Recommender...")
    cf_model.fit(interactions)

    # Step 4: Initialize the Hybrid Recommender
    hybrid_model = HybridRecommender(cb_model, cf_model)

    # Step 5: Get recommendations for a sample user
    user_id = 1  # Change this to test different users
    top_n = 5  # Number of recommendations

    recommendations = hybrid_model.recommend(user_id, top_n)
    logger.info(f"Hybrid Recommendations for User {user_id}: {recommendations}")

if __name__ == "__main__":
    main()
