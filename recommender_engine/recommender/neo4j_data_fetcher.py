import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase, Driver, Session, Record, Result
from typing import Optional

class Neo4jClient:
    def __init__(self, NEO4J_URI=os.getenv("NEO4J_URI"),
                 NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD"),
                 NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")):
        self.NEO4J_URI = NEO4J_URI
        self.NEO4J_PASSWORD = NEO4J_PASSWORD
        self.NEO4J_USERNAME = NEO4J_USERNAME

        if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
            raise ValueError("Neo4j credentials are missing! Check your .env file.")
        self._driver: Optional[Driver] = None
        try:
            self._driver = GraphDatabase.driver(
                self.NEO4J_URI, auth=(self.NEO4J_USERNAME, self.NEO4J_PASSWORD))
            self._driver.verify_connectivity()
            print('Neo4j connection is successful.')
        except Exception as e:
            raise ConnectionError(f'Failed to connect: {e}')  # from e

    def execute(self, query, params=None):
        if not self._driver:
            raise ConnectionError('Neo4j driver failed.')
        params = params or {}
        try:
            with self._driver.session() as session:
                result = session.run(query, params)
                return list(result)
        except Exception as e:
            raise RuntimeError(f'Query Excuting error :{e}')

    def close(self):
        if self._driver:
            self._driver.close()
            print('Neo4j driver closed')


class InteractionsFetcher:
    def __init__(self, db: Neo4jClient):
        self.db = db

    def fetch_interactions(self):
        # Define interaction type weights
        interaction_weights = {
            "VISITED": 5, "WISHED": 3, "SEARCHED_FOR": 2,
            "REVIEWED": 4, "FAVORITED": 4, "CLONED": 5, "CREATED": 2
        }

        default_weight = 1  # Default weight for unknown interaction types

        # Neo4j query to fetch interactions for Trip, Destination, and Event nodes
        query = """
              MATCH (u:User)-[r]->(t:Trip)
              RETURN u.id AS user, t.id AS item, 'Trip' AS type, type(r) AS interaction, t.title AS name
              UNION ALL
              MATCH (u:User)-[r]->(d:Destination)
              RETURN u.id AS user, d.id AS item, 'Destination' AS type, type(r) AS interaction, d.name AS name
              UNION ALL
              MATCH (u:User)-[r]->(e:Event)
              RETURN u.id AS user, e.id AS item, 'Event' AS type, type(r) AS interaction, e.name AS name
              """

        # Execute the query and fetch records
        records = self.db.execute(query)

        # If no records returned, handle gracefully
        if not records:
            print("No interactions found.")
            return pd.DataFrame(columns=["user", "item", "type", "interaction", "name", "weight"])
        # Prepare data from the fetched records
        data = [
            (record["user"], record["item"], record["type"], record["interaction"], record["name"])
            for record in records
        ]

        # Create DataFrame
        df = pd.DataFrame(data, columns=["user", "item", "type", "interaction", "name"])

        # Ensure categorical data types for user, item, and item_type
       

        # Map interaction types to weights, use default_weight if interaction is not found
        df["weight"] = df["interaction"].map(interaction_weights).fillna(0.0)

        df['avg'] = df.groupby('user')['weight'].transform(lambda x: self.normalize(x))
        df[["user", "item", "type"]] = df[["user", "item", "type"]].astype('category')


        return df
    def normalize(self,x):
        x = x.astype(float)
        x_sum = x.sum()
        x_num = x.astype(bool).sum()
        x_mean = x_sum / x_num
        if x.std() == 0:
            return 0.0
        return (x - x_mean) / (x.max() - x.min())
        

class ContentBasedFetcher:
    def __init__(self, db: Neo4jClient):
        self.db = db

    def get_user_styles(self, user_id):
        query ="""MATCH (u:User {id: $user_id})-[:HAD_STYLE]->(s:Tag)
        RETURN COLLECT(s.name) AS style_name"""
        
   
        records = self.db.execute(query, {"user_id": user_id})
        return [record["style_name"] for record in records]  # styles

    def fetch_new_user_data(self, new_user=True, user_id=None, limit=15):
        query = """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(a)
            OPTIONAL MATCH (u)-[:PREFERED_DURATION]->(du)
            OPTIONAL MATCH (u)-[:PREFERED_GROUP_SIZE]->(g)
            OPTIONAL MATCH (u)-[:HAD_STYLE]->(s)
            
            WITH u, a.value AS activity, du.value AS duration, g.value AS group_size, s.value AS style

            MATCH (d:Destination)
            OPTIONAL MATCH (d)-[:HAS_STYLE]->(ds:Tag)
            OPTIONAL MATCH (d)-[:HAS_TYPE]->(dt:DestinationType)
            WHERE (activity IS NULL OR d.activity = activity)
            AND (duration IS NULL OR d.duration = duration)
            AND (group_size IS NULL OR d.group_size = group_size)
            AND (style IS NULL OR (d)-[:HAS_STYLE]->(:Tag {name: style}))

            WITH d, 
                COLLECT(DISTINCT ds.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType

            RETURN d.name AS name,
                   d.description AS description,
                   tags AS tags,
                   destinationType AS destinationType,
                   'Destination' AS type

            UNION

            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(a)
            OPTIONAL MATCH (u)-[:PREFERED_DURATION]->(du)
            OPTIONAL MATCH (u)-[:PREFERED_GROUP_SIZE]->(g)
            OPTIONAL MATCH (u)-[:HAD_STYLE]->(s)
            WITH u, a.value AS activity, du.value AS duration, g.value AS group_size, s.value AS style

            MATCH (e:Event)
            OPTIONAL MATCH (e)-[:HAD_STYLE]->(es:Tag)
            WHERE (activity IS NULL OR e.activity = activity)
            AND (duration IS NULL OR e.duration = duration)
            AND (group_size IS NULL OR e.group_size = group_size)
            AND (style IS NULL OR (e)-[:HAD_STYLE]->(:Tag {name: style}))

            WITH e, 
                COLLECT(DISTINCT es.name) AS tags

            RETURN e.name AS name,
                   e.description AS description,
                   tags AS tags,
                   NULL AS destinationType,
                   'Event' AS type
            LIMIT $limit
                """
        params = {
            "user_id": user_id,
            "limit": limit
        }
        records = self.db.execute(query, params)
        results = [{"name": record["name"], 
            "description": record["description"], 
            "tags": record["tags"], 
            "destinationType": record["destinationType"], 
            "type": record["type"]} for record in records]      
        
        styles = self.get_user_styles(user_id)
        return results, styles

    def fetch_existing_user_data(self, user_id=None, limit=15):
        query = """
        MATCH (u:User {id: $user_id})

        OPTIONAL MATCH (u)-[:VISITED]->(d:Destination)
        OPTIONAL MATCH (d)-[:HAS_STYLE]->(ds:Tag)
        OPTIONAL MATCH (d)-[:HAS_TYPE]->(dt:DestinationType)
        
        WITH d, 
        COLLECT(DISTINCT ds.name) AS tags, 
        COLLECT(DISTINCT dt.name) AS destinationType
                
        RETURN d.name AS name,
               d.description AS description,
               tags AS tags,
               destinationType AS destinationType,
              'Destination' AS type
        
        UNION
        
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[:ATTEND]->(e:Event)
        OPTIONAL MATCH (e)-[:HAD_STYLE]->(es:Tag)
        
        WITH e, 
        COLLECT(DISTINCT es.name) AS tags
        
        RETURN e.name AS name,
               e.description AS description,
               tags AS tags,
               NULL AS destinationType,
               'Event' AS type
        LIMIT $limit
        """
        params = {
            "user_id": user_id,
            "limit": limit if limit is not None else 10
        }
        records = self.db.execute(query, params)
        
        records = self.db.execute(query, params)
        results = [{"name": record["name"], 
            "description": record["description"], 
            "tags": record["tags"], 
            "destinationType": record["destinationType"], 
            "type": record["type"]} for record in records]
       
        
        styles = self.get_user_styles(user_id)
        return results, styles


if __name__ == "__main__":

    db_client = Neo4jClient()
    interaction_fetcher = InteractionsFetcher(db_client)
    print("Fetching interactions...")
    interactions_df = interaction_fetcher.fetch_interactions()
    print("\nInteractions DataFrame Sample:")
    print(interactions_df['user'].head(15))
    print(f"Total interactions fetched: {len(interactions_df)}")
    example_user_id = "f580b257-861e-48b8-a145-4cef50b14229"
    print(type(interactions_df['user'][0]))
    print(interactions_df.columns)
    print(interactions_df[['avg','weight']].head())
    print(interactions_df[[ 'type', 'interaction', 'name', 'weight', 'avg']])
    db_client.close()