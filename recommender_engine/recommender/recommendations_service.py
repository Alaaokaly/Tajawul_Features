import logging.config
from neo4j_data_fetcher import Neo4jClient, ContentBasedFetcher
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF
from CB_recommendations import ContentBasedRecommender
from hybird import HybridRecommender  # Note: you may want to rename this file to "hybrid.py"
import os
import sys
import time
import logging
import argparse
import datetime
from typing import Any, List, Dict
import pandas as pd


# Configure logging
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "rec_service.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger('recs-server')


class RecommendationService:
    def __init__(self,
                 db_client: Neo4jClient,
                 rec_limit: int = 5,
                 use_epsilon_greedy: bool = True,
                 epsilon: float = 0.2,
                 dry_run: bool = False,
                 user_id_for_single_run: str = None,
                 recommendation_retention_days: int = 30,
                 threshold_interactions: int = 5,
                 use_mmr_for_cb: bool = True):
        
        self.db_client = db_client
        self.rec_limit = rec_limit
        self.recommendation_retention_days = recommendation_retention_days
        self.use_epsilon_greedy = use_epsilon_greedy
        self.epsilon = epsilon
        self.dry_run = dry_run
        self.user_id_for_single_run = user_id_for_single_run
        self.threshold_interactions = threshold_interactions
        self.use_mmr_for_cb = use_mmr_for_cb
        self.hybrid_model = None
        self.item_types = ['Destination', 'Event', 'Trip', 'Post']
        self.user_cf_model = None

        logger.info(f"RecommendationService initialized with:")
        logger.info(f"  - dry_run={self.dry_run}")
        logger.info(f"  - epsilon_greedy={self.use_epsilon_greedy}, epsilon={self.epsilon}")
        logger.info(f"  - recommendation_retention_days={self.recommendation_retention_days}")
        logger.info(f"  - threshold_interactions={self.threshold_interactions}")
        logger.info(f"  - single_user_mode_user_id={self.user_id_for_single_run if self.user_id_for_single_run else 'N/A'}")

    def _initialize_models(self):
        """Initialize and fit all recommendation models"""
        logger.info("Initializing recommendation models...")
        
        try:
            # Initialize CF models with appropriate parameters
            self.user_cf_model = user_cf_model = UserBasedCF(
                self.db_client,
                k_neighbors=25,
                min_overlap=1,  # Should match threshold_interactions
                min_sim=0.001
            )

            item_cf_model = ItemBasedCF(
                self.db_client,
                k_neighbors=15,
                min_overlap=1,  # Should match threshold_interactions
                min_sim=0.001
            )

            logger.info("Fitting user-based CF model...")
            self.user_cf_model.fit()
            logger.info("User-based CF model fitted successfully.")

            logger.info("Fitting item-based CF model...")
            item_cf_model.fit()
            logger.info("Item-based CF model fitted successfully.")

            # Initialize hybrid model
            self.hybrid_model = HybridRecommender(
                db_client=self.db_client,
                user_cf_model=self.user_cf_model,
                item_cf_model=item_cf_model,
                threshold_interactions=self.threshold_interactions
            )
            
            logger.info("Hybrid recommendation model initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def create_item_recommendation(self, user_id: str, item_id: str, item_type: str, score: float, rank: int):
        """Create a recommendation relationship in the database"""
        timestamp = int(datetime.datetime.now().timestamp())
        query = f"""
        MATCH (u:User {{id: $user_id}})
        MATCH (i:{item_type} {{id: $item_id}})
        CREATE (u)-[r:RECOMMENDED {{
            score: $score,
            rank: $rank,
            timestamp: $timestamp
        }}]->(i)
        RETURN r
        """
        
        if self.dry_run:
            logger.info(f'DRY RUN: Skipping DB write for {item_type} {item_id} by user {user_id} (score: {score:.4f}, rank: {rank})')
            return
            
        try:
            self.db_client.execute(query, {
                "user_id": user_id,
                "item_id": item_id,
                "score": float(score),
                "rank": int(rank),
                "timestamp": timestamp
            })
            logger.debug(f"Created RECOMMENDED relationship: user {user_id} -> {item_type} {item_id}")
        except Exception as e:
            logger.error(f"Failed to create RECOMMENDED relationship for user {user_id} to {item_type} {item_id}: {e}")

    def create_user_recommendation(self,user_id, recommended_user_id, rank ):
        timestamp = int(datetime.datetime.now().timestamp())

        ## revisit query 
        query = """
        MATCH (u:User {id: $user_id})
        MATCH (ru:User {id: $recommended_user_id})
        CREATE (u)-[r:SIMILAR{
            rank: $rank,
            timestamp: $timestamp
        }]->(ru)
        RETURN r
        """
        if self.dry_run :
            logger.debug(f"DRY RUN : creating a relation for user {user_id} with top simialr users")
            return 
        try: 
            self.db_client.execute(query, {
                    "user_id": user_id, 
                    "recommended_user_id": recommended_user_id, 
                    "rank": rank,
                    "timestamp": timestamp})
        except Exception as  e :
            logger.error(f"ERROR: creating recs for  similar users didn't resolve, error {e}")
    

    def generate_user_recommendations(self, user_id: str):
        """Generate recommendations for a single user"""
        logger.info(f"Generating recommendations for user: {user_id}")
        
        if not self.hybrid_model:
            logger.error("Hybrid model not initialized. Cannot generate recommendations.")
            return
            
        total_recommendations = 0
        
        # Generate item recommendations for each item type
        for item_type in self.item_types:
            try:
                logger.debug(f"Generating {item_type} recommendations for user {user_id}")
                
                if self.use_epsilon_greedy:
                    recommendations_df = self.hybrid_model.recommend_with_epsilon_greedy(
                        user_id=user_id,
                        top_n=self.rec_limit,
                        item_type=item_type,
                        use_mmr_for_cb=self.use_mmr_for_cb,
                        epsilon=self.epsilon
                    )
                    logger.debug(f"Generated {len(recommendations_df)} {item_type} recommendations using epsilon-greedy")
                else:
                    recommendations_df = self.hybrid_model.recommend(
                        user_id=user_id,
                        top_n=self.rec_limit,
                        item_type=item_type,
                        use_mmr_for_cb=self.use_mmr_for_cb
                    )
                    logger.debug(f"Generated {len(recommendations_df)} {item_type} recommendations using hybrid model")
    
                if recommendations_df.empty:
                    logger.warning(f"No {item_type} recommendations generated for user {user_id}")
                    continue
    
                # Validate required columns
                required_columns = ['item', 'type', 'score', 'rank']
                missing_columns = [col for col in required_columns if col not in recommendations_df.columns]
                if missing_columns:
                    logger.error(f"Missing required columns in {item_type} recommendations: {missing_columns}")
                    continue
    
                # Create recommendation relationships
                for _, rec in recommendations_df.iterrows():
                    self.create_item_recommendation(
                        user_id=user_id,
                        item_id=str(rec['item']),
                        item_type=str(rec['type']),
                        score=float(rec['score']),
                        rank=int(rec['rank'])
                    )
                
                total_recommendations += len(recommendations_df)
                logger.info(f"Created {len(recommendations_df)} {item_type} recommendations for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error generating {item_type} recommendations for user {user_id}: {e}")
                continue
    
        # Generate user-to-user recommendations
        try:
            logger.debug(f"Generating user-to-user recommendations for user {user_id}")

            # Call the method on the hybrid_model which delegates to user_cf_model
            user_recommendations = self.user_cf_model.recommend_users_to_user(
                user_id=user_id,
                top_n=5
            )
            
            if user_recommendations.empty:
                logger.warning(f"No user-to-user recommendations returned for user {user_id}")
            else:
                logger.info(f"Received {len(user_recommendations)} user-user recommendations for user {user_id}")
                for idx, rec in user_recommendations.iterrows():
                    recommended_user_id = rec.get('user_id') or rec.get('id') or rec.get('user')
                    if not recommended_user_id:
                        logger.warning(f"Skipping row without recommended user id: {rec}")
                        continue
            
                    similarity = rec.get('similarity', None)
                    rank = rec.get('rank', idx + 1)
            
                    logger.info(f"Creating SIMILAR relation: {user_id} → {recommended_user_id} | rank={rank}, similarity={similarity}")
            
                    self.create_user_recommendation(
                        user_id=user_id,
                        recommended_user_id=str(recommended_user_id),
                        rank=int(rank),
                        similarity=float(similarity) if similarity is not None else None
                    )
            
            
                            
                    logger.info(f"Created {len(user_recommendations)} user-to-user recommendations for user {user_id}")
            
                
        except Exception as e:
            logger.error(f"Error generating user-to-user recommendations for user {user_id}: {e}")
    

    def clean_past_recommendations(self):
        """Clean old recommendation relationships"""
        threshold_date = datetime.datetime.now() - datetime.timedelta(days=self.recommendation_retention_days)
        timestamp_cutoff = int(threshold_date.timestamp())
        
        # Clean item recommendations
        item_query = """
        MATCH (u:User)-[r:RECOMMENDED]->(i)
        WHERE r.timestamp < $cutoff
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        # Clean user recommendations
        user_query = """
        MATCH (u:User)-[r:SIMILAR]->(ru:User)
        WHERE r.timestamp < $cutoff
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        total_deleted = 0
        
        if self.dry_run:
            logger.info(f"DRY RUN: Skipping cleanup of recommendations older than {self.recommendation_retention_days} days")
            return 0
            
        try:
            # Clean item recommendations
            result = self.db_client.execute(item_query, {"cutoff": timestamp_cutoff})
            item_deleted = result[0]['deleted_count'] if result and result[0] else 0
            
            # Clean user recommendations
            result = self.db_client.execute(user_query, {"cutoff": timestamp_cutoff})
            user_deleted = result[0]['deleted_count'] if result and result[0] else 0
            
            total_deleted = item_deleted + user_deleted
            logger.info(f"Cleaned {item_deleted} item recommendations and {user_deleted} user recommendations older than {self.recommendation_retention_days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning past recommendations: {e}")
            
        return total_deleted

    def get_all_users(self) -> List[str]:
        """Fetch all user IDs from the database"""
        query = """
        MATCH (u:User)
        RETURN DISTINCT u.id AS user_id
        ORDER BY u.id
        """
        
        try:
            result = self.db_client.execute(query)
            users = [record["user_id"] for record in result if record["user_id"]]
            logger.info(f"Fetched {len(users)} users from the database")
            return users
        except Exception as e:
            logger.error(f"Error fetching users from database: {e}")
            return []

    def get_user_interaction_count(self, user_id: str) -> int:
        """Get the interaction count for a user (for debugging/logging)"""
        if self.hybrid_model:
            return self.hybrid_model.get_interactions_count(user_id)
        return 0

    def run_single_user_cycle(self, user_id: str):
        """Run recommendation generation for a single user"""
        if not user_id:
            logger.error("Cannot run single user cycle: No user ID provided")
            return
            
        interaction_count = self.get_user_interaction_count(user_id)
        logger.info(f"Running single user cycle for {user_id} (interactions: {interaction_count})")
        
        self.generate_user_recommendations(user_id)

    def run_periodic_cycle(self):
        """Run recommendation generation for all users"""
        logger.info("Starting periodic recommendation cycle for all users")
        
        # Optional: Clean old recommendations first
        # self.clean_past_recommendations()
        
        users = self.get_all_users()
        if not users:
            logger.warning("No users found in database. Skipping recommendation generation.")
            return

        successful_users = 0
        failed_users = 0
        
        for i, user_id in enumerate(users, 1):
            try:
                interaction_count = self.get_user_interaction_count(user_id)
                logger.info(f"Processing user {i}/{len(users)}: {user_id} (interactions: {interaction_count})")
                
                self.generate_user_recommendations(user_id)
                successful_users += 1
                
            except Exception as e:
                logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
                failed_users += 1
                
        logger.info(f"Periodic cycle completed. Success: {successful_users}, Failed: {failed_users}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Recommendation Service")
    parser.add_argument("--rec-limit", type=int, default=5,
                        help="Max recommendations per user-item type (default: 5)")
    parser.add_argument("--retention", type=int, default=30,
                        help="Days to keep generated recommendations before deletion (default: 30)")
    parser.add_argument("--user", type=str, default=None,
                        help="Generate recommendations for a specific user only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without writing to database")
    parser.add_argument("--use-epsilon-greedy", action="store_true", default=True,
                        help="Use epsilon-greedy for exploration (default: True)")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Epsilon value for exploration (default: 0.2)")
    parser.add_argument("--periodic", action="store_true",
                        help="Run for all users in database (overrides --user)")
    parser.add_argument("--threshold", type=int, default=5,
                        help="Interaction threshold for hybrid model (default: 5)")
    parser.add_argument("--no-mmr", action="store_true",
                        help="Disable MMR for content-based recommendations")
    
    args = parser.parse_args()

    logger.info(f"Starting Hybrid Recommendation Service")
    logger.info(f"Version: {os.getenv('BUILD_VERSION', 'local')}")
    
    # Initialize database client
    try:
        db_client = Neo4jClient()
        logger.info("Database client initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize database client: {e}")
        sys.exit(1)

    # Initialize service
    service = RecommendationService(
        db_client=db_client,
        rec_limit=args.rec_limit,
        recommendation_retention_days=args.retention,
        use_epsilon_greedy=args.use_epsilon_greedy,
        epsilon=args.epsilon,
        dry_run=args.dry_run,
        user_id_for_single_run=args.user,
        threshold_interactions=args.threshold,
        use_mmr_for_cb=not args.no_mmr
    )

    # Initialize models
    try:
        service._initialize_models()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize recommendation models: {e}")
        db_client.close()
        sys.exit(1)

    # Run service
    try:
        if args.dry_run:
            logger.info("=== DRY RUN MODE ACTIVE (No database writes) ===")
        else:
            logger.info("=== LIVE MODE ACTIVE (Writing to database) ===")

        if args.periodic:
            logger.info("Running in PERIODIC mode (all users)")
            service.run_periodic_cycle()
        elif args.user:
            logger.info(f"Running in SINGLE USER mode for user: {args.user}")
            service.run_single_user_cycle(args.user)
        else:
            logger.error("No run mode specified. Use --periodic or --user <user_id>")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during service execution: {e}")
        sys.exit(1)
    finally:
        db_client.close()
        logger.info("Database connection closed. Service finished.")


if __name__ == "__main__":
    main()
