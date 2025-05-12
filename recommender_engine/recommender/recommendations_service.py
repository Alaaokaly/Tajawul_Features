import logging.config
from neo4j_data_fetcher import Neo4jClient
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF
from CB_recommendations import ContentBasedRecommender
from hybird import HybridRecommender
import os
import sys
import time
import logging
import argparse
import datetime
from typing import Any


# cloud automation using git actions or cloud functions 
# 
#use Azure Container Instances with Timer trigger via Azure Logic Apps
logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("recs_service.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('recs-server')



class RecommendationService:
    def __init__(self, 
                 db_client: Neo4jClient,
                 rec_limit: int = 10,
                 retention_days: int = 30,
                 use_epsilon_greedy: bool = False,
                 epsilon: float = 0.2,
                 dry_run: bool = False):
        
        self.db_client = db_client
        self.rec_limit = rec_limit
        self.retention_days = retention_days
        self.use_epsilon_greedy = use_epsilon_greedy
        self.epsilon = epsilon
        self.dry_run = dry_run
        self.hybrid_model = None
        self.item_types = ['Destination', 'Event', 'Trip', 'Post']
    
    def _initialize_models(self):
        logger.info("Initializing recommendation models...")
        
        # Initialize CF models
        user_cf_model = UserBasedCF(
            self.db_client, 
            k_neighbors=10, 
            min_overlap=1, 
            min_sim=0.05
        )
        
        item_cf_model = ItemBasedCF(
            self.db_client, 
            k_neighbors=15, 
            min_overlap=1, 
            min_sim=0.05
        )
        
        # Fit the models
        logger.info("Fitting user-based CF model...")
        user_cf_model.fit()
        
        logger.info("Fitting item-based CF model...")
        item_cf_model.fit()
        
        # Initialize hybrid model
        self.hybrid_model = HybridRecommender(
            db_client=self.db_client,
            user_cf_model=user_cf_model,
            item_cf_model=item_cf_model,
            threshold_interactions=1
        )
    def create_item_recommendation(self, user_id, item_id, item_type, score, rank):
        timestamp = int(datetime.datetime.now().timestamp())
        ## revise the query 
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
        if self.dry_run :
            logger.info(f'DRY RUN : creating recommendations from {user_id} to item id {item_id} that is a {item_type}')
            return 
        try:
            self.db_client.execute(query, {
                    "user_id": user_id, 
                    "item_id": item_id, 
                    "item_type": item_type,
                    "score": score, 
                    "rank": rank,
                    "timestamp": timestamp
                })
        except Exception as e :
            logger.error(f"ERROR: creating recs for item didn't resolve, error {e}")

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
    
    def generate_user_recommendations(self, user_id):
        for item_type in self.item_types:
            if self.use_epsilon_greedy:
                    recommendations = self.hybrid_model.recommend_with_epsilon_greedy(
                        user_id=user_id,
                        top_n=self.rec_limit,
                        item_type=item_type,
                        epsilon=self.epsilon
                    )
                    logger.debug(f"Generating Recommendations For {user_id} with epsion {self.epsilon} ")
            else:
                    
                    recommendations = self.hybrid_model.recommend(
                        user_id=user_id,
                        top_n=self.rec_limit,
                        item_type=item_type
                    ) 
                    logger.info(f"Generating Recommendations For {user_id} ")
            if recommendations.empty:
                    logger.warning(f"No {item_type} recommendations generated for user {user_id}")
                    continue
                
                # Create recommendation relationships
            for _, rec in recommendations.iterrows():
                    
                    self.create_item_recommendation(
                        user_id=user_id,
                        item_id=rec['item'],
                        item_type=rec['type'],
                        score=rec['score'],
                        rank=rec['rank']
                    )  
                    logger.info(f"Creating Item Recommendations For {user_id} ")
        user_recommendations = self.hybrid_model.user_cf_model.recommend_users_to_user(
                user_id=user_id,
                top_n=5
            ) 
        logger.debug(f"Creating Users Recommendations For {user_id} ") 
        if not user_recommendations.empty:
                for _, rec in user_recommendations.iterrows():
                    self.create_user_recommendation(
                        user_id=user_id,
                        recommended_user_id=rec['user_id'],
                        rank=rec.get('rank', 0)
                    )
                
                logger.debug(f"Created {len(user_recommendations)} user recommendations for user {user_id} ")
        else:
                logger.warning(f"No user recommendations generated for user {user_id}")               


    def clean_past_rec(self):
        threshold_date  =  datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
        ## revisit query 
        query = f"""
        MATCH (u:User)-[r:RECOMMENDED]->(i)
        WHERE r.timestamp < {threshold_date}
        DELETE r
        RETURN count(r) as deleted_count
        """
        if self.dry_run:
            logger.debug(f"DRY RUN: Would delete recommendations older than {self.retention_days} days")
            return 0
        result = self.db_client.execute(query)
        deleted_count = result[0]['deleted_count'] if result else 0 
        logger.debug(f"Deleted {deleted_count} recommendations older than {self.retention_days} days")
        return deleted_count

    def get_active_users(self):
        query = """MATCH (u:User)-[r:CREATE|SHARE|REACT]->(p:Post)
        WHERE datetime() - duration({days: 2}) < r.date
        RETURN DISTINCT u.id AS user_id"""
        new_user_query =  """MATCH (u:User)
                    WHERE NOT (u)-[:CREATE|SHARE|REACT]->(:Post)
                    AND datetime() - duration({days: 0}) < datetime(u.creationDate)
                    RETURN u.id AS user_id
                    """
        result = self.db_client.execute(query)
        active_users = [record["user_id"] for record in result]
        new_user_result = self.db_client.execute(new_user_query)
        new_users = [record["user_id"] for record in new_user_result]
        active_users.extend(new_users)
        return active_users

    def run_cycle(self, user_id):
        self.generate_user_recommendations(user_id)

    def run_periodic_cycle (self):
        self.clean_past_rec()
        users = self.get_active_users()
        for user in users :
            self.generate_user_recommendations(user)

def main():
    parser = argparse.ArgumentParser(description="Recommendation Service")
    parser.add_argument("--rec-limit", type=int, default=10,
                        help="Max recommendations per user-item type (default: 10)")

    parser.add_argument("--retention", type=int, default=30,
                        help="Days to keep recommendations before deletion (default: 30)")

    parser.add_argument("--user", type=str, default=None,
                        help="Generate recommendations for a specific user only")
    
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without writing to database")

    parser.add_argument("--use-epsilon-greedy",  action="store_true", default=True,
                        help="Use epsilon-greedy for exploration (default: False)")

    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Epsilon value for exploration (default: 0.2)")

    parser.add_argument("--periodic", type=bool, default=False,  help= "Generating periodic recommedations and decaying" )
    args = parser.parse_args()
    logger.info(f"Starting recommendation service with parameters: {args}")
    db_client = Neo4jClient()
        
        # Initialize recommendation service

    service = RecommendationService(
        db_client=db_client,
        rec_limit=args.rec_limit,
        retention_days=args.retention,
        use_epsilon_greedy=args.use_epsilon_greedy,
        epsilon=args.epsilon,
        dry_run=args.dry_run
    )
    service._initialize_models()
    if args.dry_run:
        logger.info(f"Dry-Run trial with user {args.user}")

        if args.periodic:
            service.run_periodic_cycle()
            
        else :
            service.run_cycle(args.user)
    else: 
         
         if args.periodic:
            service.run_periodic_cycle()
            logger.info(f"Starting recommendation service Periodic In DB with parameters: {args}")
         else :
            service.run_cycle(args.user)
            logger.info(f"Starting recommendation service  In DB with parameters: {args}")


        
if __name__ == "__main__":
     main()