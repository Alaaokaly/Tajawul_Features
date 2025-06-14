from typing import Optional
from neo4j import GraphDatabase, Driver
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


class Neo4jClient:
    def __init__(self, NEO4J_URI=os.getenv("NEO4J_URI"),
                 NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD"),
                 NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")):
        self.NEO4J_URI = NEO4J_URI
        self.NEO4J_PASSWORD = NEO4J_PASSWORD
        self.NEO4J_USERNAME = NEO4J_USERNAME

        if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
            raise ValueError(
                "Neo4j credentials are missing! Check your .env file.")
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
              UNION ALL
              MATCH (u:User)-[r]->(p:Post)
              RETURN u.id AS user, p.id AS item, 'Post' AS type, type(r) AS interaction, p.name AS name
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
        df['name'] = df['name'].str.strip()

        # Ensure categorical data types for user, item, and item_type
       

        # Map interaction types to weights, use default_weight if interaction is not found
        df["weight"] = df["interaction"].map(interaction_weights).fillna(0.0)

        df['avg'] = df.groupby('user')['weight'].transform(lambda x: self.normalize(x))
        df[["user", "item", "type"]] = df[["user", "item", "type"]].astype('category')


        return df
    def normalize(self, x):
        x = x.astype(float)
        x_sum = x.sum()
        x_num = x.astype(bool).sum()
        
        # Fix division by zero warning
        if x_num == 0:
            return x * 0.0  # Return zeros with the same shape as x
        
        x_mean = x_sum / x_num
        
        # Check if all values are the same (which would result in max-min = 0)
        range_value = x.max() - x.min()
        if range_value == 0:
            return x * 0.0  # Return zeros with the same shape as x
        
        return (x - x_mean) / range_value
    
class ContentBasedFetcher:
    def __init__(self, db: Neo4jClient):
        self.db = db

    def get_user_styles(self, user_id=None):
        """Get user styles for single user or all users"""
        if user_id:
            query = """MATCH (u:User {id: $user_id})-[:PREFERED_ACTIVITY]->(s:Tag)
            RETURN u.id as user_id, COLLECT(s.name) AS style_names"""
            params = {"user_id": user_id}
        else:
            query = """MATCH (u:User)-[:PREFERED_ACTIVITY]->(s:Tag)
            RETURN u.id as user_id, COLLECT(s.name) AS style_names"""
            params = {}
        
        records = self.db.execute(query, params)
        if user_id:
            return records[0]["style_names"] if records else []
        else:
            return {record["user_id"]: record["style_names"] for record in records}

    def get_all_user_styles(self):
        """Get styles for all users"""
        return self.get_user_styles()

    def get_user_interactions(self, user_id=None):
        """Get user interactions for single user or all users"""
        if user_id:
            user_filter = "MATCH (u:User {id: $user_id})"
            params = {"user_id": user_id}
        else:
            user_filter = "MATCH (u:User)"
            params = {}
            
        query = f"""
                {user_filter}
                MATCH (u)-[r]->(d:Destination)
                WHERE type(r) IN ['VISITED', 'FOLLOWED', 'FAVORITED', 'WISHED','CREATED']
                WITH u, d
                MATCH (d)-[:HAD_TAG]->(t:Tag)
                WITH u, d, collect(DISTINCT t.name) AS tags
                RETURN u.id as user,
                       d.id as item,
                       "Destination" as item_type,
                       tags,
                       d.description as description,
                       1 as score
                        
               UNION

                {user_filter}
                MATCH (u)-[r]->(e:Event)
                WHERE type(r) IN ['CREATED', 'FOLLOWED', 'FAVORITED', 'WISHED']
                WITH u, e
                MATCH (e)-[:HAD_TAG]->(t:Tag)
                WITH u, e, collect(DISTINCT t.name) AS tags
                RETURN u.id as user,
                       e.id as item,
                       "Event" as item_type,
                       tags,
                       e.description as description,
                       1 as score
        """
        
        records = self.db.execute(query, params)
        results = [{'user': record['user'],
                    'item': record['item'],
                    'item_type': record['item_type'],
                    "description": record["description"],
                    "tags": record["tags"],
                    "score": record["score"]
                    } for record in records]
        return results

    def get_all_user_interactions(self):
        """Get interactions for all users"""
        return self.get_user_interactions()

    def fetch_new_user_data(self, user_id=None):
        """Fetch data for new users (single user or all users)"""
        if user_id:
            user_filter = "MATCH (u:User {id: $user_id})"
            params = {"user_id": user_id}
        else:
            user_filter = "MATCH (u:User)"
            params = {}
            
        query = f"""
            {user_filter}
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(s:Tag) 
            WITH u, COLLECT(s.name) AS preferred_styles

            MATCH (d:Destination)
            OPTIONAL MATCH (d)-[:HAD_TAG]->(ds:Tag)
            OPTIONAL MATCH (d)-[:HAD_DESTINATION_TYPE]->(dt:Type)
            
            WITH u, d, ds, dt, preferred_styles
            WHERE size([style IN preferred_styles WHERE style = ds.name]) > 0 OR size(preferred_styles) = 0

            WITH u, d, 
                COLLECT(DISTINCT ds.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType

            RETURN u.id as user_id,
                   d.id as id,
                   d.name AS name,
                   d.description AS description,
                   tags,
                   destinationType AS destinationType,
                   'Destination' AS item_type 
            
            UNION

            {user_filter}
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(s:Tag)
            WITH u, s.name AS style

            MATCH (e:Event)
            MATCH (e)-[:HAD_TAG]->(es:Tag)
            WHERE (style IS NULL OR (e)-[:HAD_TAG]->(:Tag {{name: style}}))
            WITH u, e, 
                COLLECT(DISTINCT es.name) AS tags

           RETURN u.id as user_id,
                  e.id as id, 
                  e.name AS name,
                  e.description AS description,
                  tags,
                  NULL AS destinationType,
                  'Event' AS item_type 
        """
        
        records = self.db.execute(query, params)
        results = [{'user': record['user_id'],
                    "item": record['id'],
                    "item_type": record["item_type"],
                    "name": record["name"],
                    "description": record["description"],
                    "tags": record["tags"],
                    "destinationType": record["destinationType"]} for record in records]
        
        if user_id:
            styles = self.get_user_styles(user_id)
            return results, styles
        else:
            styles = self.get_all_user_styles()
            return results, styles

    def fetch_all_new_user_data(self):
        """Fetch data for all new users"""
        return self.fetch_new_user_data()

    def fetch_existing_user_data(self, user_id=None):
        """Fetch data for existing users (single user or all users)"""
        if user_id:
            user_filter = "MATCH (u:User {id: $user_id})"
            params = {"user_id": user_id}
        else:
            user_filter = "MATCH (u:User)"
            params = {}
            
        query = f"""
            {user_filter}
            MATCH (u)-[:VISITED]->(d1:Destination)
            MATCH (d1)-[:HAD_TAG]->(preferredTag:Tag)

            WITH u, COLLECT(DISTINCT preferredTag.name) AS userPreferredTags

            MATCH (d2:Destination)
            WHERE NOT (u)-[:VISITED]->(d2) AND NOT (u)-[:FOLLOWED]->(d2)
            OPTIONAL MATCH (d2)-[:HAD_TAG]->(tag:Tag)
            OPTIONAL MATCH (d2)-[:HAD_DESTINATION_TYPE]->(dt:Type)

            WITH u, d2, 
                COLLECT(DISTINCT tag.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType, 
                userPreferredTags
            WHERE size([tag IN tags WHERE tag IN userPreferredTags]) > 0

            RETURN u.id as user_id,
                d2.id as id,
                d2.name AS name,
                d2.description AS description,
                tags AS tags,
                destinationType AS destinationType,
                'Destination' AS item_type

            UNION ALL

            {user_filter}
            MATCH (u)-[:VISITED]->(d1:Destination)
            MATCH (d1)-[:HAD_TAG]->(preferredTag:Tag)

            WITH u, COLLECT(DISTINCT preferredTag.name) AS userPreferredTags

            MATCH (e2:Event)
            WHERE NOT (u)-[:CREATED]->(e2) AND NOT (u)-[:FOLLOWED]->(e2) AND NOT (u)-[:FAVORITED]->(e2)
            OPTIONAL MATCH (e2)-[:HAD_TAG]->(etag:Tag)

            WITH u, e2, 
                COLLECT(DISTINCT etag.name) AS tags, 
                userPreferredTags
            WHERE size([tag IN tags WHERE tag IN userPreferredTags]) > 0

            RETURN u.id as user_id,
                e2.id as id,
                e2.name AS name,
                e2.description AS description,
                tags AS tags,
                NULL AS destinationType,
                'Event' AS item_type
        """
        
        records = self.db.execute(query, params)
        results = [{'user': record['user_id'],
                    "item": record['id'],
                    "item_type": record["item_type"],
                    "name": record["name"],
                    "description": record["description"],
                    "tags": record["tags"],
                    "destinationType": record["destinationType"]} for record in records]
        
        if user_id:
            styles = self.get_user_styles(user_id)
            return results, styles
        else:
            styles = self.get_all_user_styles()
            return results, styles

    def fetch_all_existing_user_data(self):
        """Fetch data for all existing users"""
        return self.fetch_existing_user_data()

    def get_recommendations_for_all_users(self, user_type='existing'):
        """Get recommendations for all users based on their type"""
        if user_type == 'new':
            return self.fetch_all_new_user_data()
        else:
            return self.fetch_all_existing_user_data()

    def get_user_recommendations_by_batch(self, user_ids, user_type='existing', batch_size=100):
        """Process users in batches to avoid memory issues with large datasets"""
        all_results = []
        all_styles = {}
        
        for i in range(0, len(user_ids), batch_size):
            batch_ids = user_ids[i:i + batch_size]
            
            # Create batch query
            user_filter = f"MATCH (u:User) WHERE u.id IN {batch_ids}"
            
            if user_type == 'new':
                results, styles = self._fetch_batch_new_user_data(batch_ids)
            else:
                results, styles = self._fetch_batch_existing_user_data(batch_ids)
            
            all_results.extend(results)
            all_styles.update(styles)
        
        return all_results, all_styles

    def _fetch_batch_new_user_data(self, user_ids):
        """Helper method for batch processing new users"""
        query = """
            MATCH (u:User) WHERE u.id IN $user_ids
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(s:Tag) 
            WITH u, COLLECT(s.name) AS preferred_styles

            MATCH (d:Destination)
            OPTIONAL MATCH (d)-[:HAD_TAG]->(ds:Tag)
            OPTIONAL MATCH (d)-[:HAD_DESTINATION_TYPE]->(dt:Type)
            
            WITH u, d, ds, dt, preferred_styles
            WHERE size([style IN preferred_styles WHERE style = ds.name]) > 0 OR size(preferred_styles) = 0

            WITH u, d, 
                COLLECT(DISTINCT ds.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType

            RETURN u.id as user_id,
                   d.id as id,
                   d.name AS name,
                   d.description AS description,
                   tags,
                   destinationType AS destinationType,
                   'Destination' AS item_type 
            
            UNION

            MATCH (u:User) WHERE u.id IN $user_ids
            OPTIONAL MATCH (u)-[:PREFERED_ACTIVITY]->(s:Tag)
            WITH u, s.name AS style

            MATCH (e:Event)
            MATCH (e)-[:HAD_TAG]->(es:Tag)
            WHERE (style IS NULL OR (e)-[:HAD_TAG]->(:Tag {name: style}))
            WITH u, e, 
                COLLECT(DISTINCT es.name) AS tags

           RETURN u.id as user_id,
                  e.id as id, 
                  e.name AS name,
                  e.description AS description,
                  tags,
                  NULL AS destinationType,
                  'Event' AS item_type 
        """
        
        records = self.db.execute(query, {"user_ids": user_ids})
        results = [{'user': record['user_id'],
                    "item": record['id'],
                    "item_type": record["item_type"],
                    "name": record["name"],
                    "description": record["description"],
                    "tags": record["tags"],
                    "destinationType": record["destinationType"]} for record in records]
        
        # Get styles for batch
        style_query = """MATCH (u:User)-[:PREFERED_ACTIVITY]->(s:Tag)
                        WHERE u.id IN $user_ids
                        RETURN u.id as user_id, COLLECT(s.name) AS style_names"""
        style_records = self.db.execute(style_query, {"user_ids": user_ids})
        styles = {record["user_id"]: record["style_names"] for record in style_records}
        
        return results, styles

    def _fetch_batch_existing_user_data(self, user_ids):
        """Helper method for batch processing existing users"""
        query = """
            MATCH (u:User) WHERE u.id IN $user_ids
            MATCH (u)-[:VISITED]->(d1:Destination)
            MATCH (d1)-[:HAD_TAG]->(preferredTag:Tag)

            WITH u, COLLECT(DISTINCT preferredTag.name) AS userPreferredTags

            MATCH (d2:Destination)
            WHERE NOT (u)-[:VISITED]->(d2) AND NOT (u)-[:FOLLOWED]->(d2)
            OPTIONAL MATCH (d2)-[:HAD_TAG]->(tag:Tag)
            OPTIONAL MATCH (d2)-[:HAD_DESTINATION_TYPE]->(dt:Type)

            WITH u, d2, 
                COLLECT(DISTINCT tag.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType, 
                userPreferredTags
            WHERE size([tag IN tags WHERE tag IN userPreferredTags]) > 0

            RETURN u.id as user_id,
                d2.id as id,
                d2.name AS name,
                d2.description AS description,
                tags AS tags,
                destinationType AS destinationType,
                'Destination' AS item_type

            UNION ALL

            MATCH (u:User) WHERE u.id IN $user_ids
            MATCH (u)-[:VISITED]->(d1:Destination)
            MATCH (d1)-[:HAD_TAG]->(preferredTag:Tag)

            WITH u, COLLECT(DISTINCT preferredTag.name) AS userPreferredTags

            MATCH (e2:Event)
            WHERE NOT (u)-[:CREATED]->(e2) AND NOT (u)-[:FOLLOWED]->(e2) AND NOT (u)-[:FAVORITED]->(e2)
            OPTIONAL MATCH (e2)-[:HAD_TAG]->(etag:Tag)

            WITH u, e2, 
                COLLECT(DISTINCT etag.name) AS tags, 
                userPreferredTags
            WHERE size([tag IN tags WHERE tag IN userPreferredTags]) > 0

            RETURN u.id as user_id,
                e2.id as id,
                e2.name AS name,
                e2.description AS description,
                tags AS tags,
                NULL AS destinationType,
                'Event' AS item_type
        """
        
        records = self.db.execute(query, {"user_ids": user_ids})
        results = [{'user': record['user_id'],
                    "item": record['id'],
                    "item_type": record["item_type"],
                    "name": record["name"],
                    "description": record["description"],
                    "tags": record["tags"],
                    "destinationType": record["destinationType"]} for record in records]
        
        # Get styles for batch
        style_query = """MATCH (u:User)-[:PREFERED_ACTIVITY]->(s:Tag)
                        WHERE u.id IN $user_ids
                        RETURN u.id as user_id, COLLECT(s.name) AS style_names"""
        style_records = self.db.execute(style_query, {"user_ids": user_ids})
        styles = {record["user_id"]: record["style_names"] for record in style_records}
        
        return results, styles
    
if __name__ == "__main__":

    db_client = Neo4jClient()
    interaction_fetcher = InteractionsFetcher(db_client)
    print("Fetching interactions...")
    interactions_df = interaction_fetcher.fetch_interactions()
    print("\nInteractions DataFrame Sample:")
    print(interactions_df.head())
    print(f"Total interactions fetched: {len(interactions_df)}")

    # #4. Fetch Content-Based Data (Example User)
    content_fetcher = ContentBasedFetcher(db_client)
    example_user_id = "222e1a80-a195-4f19-be67-99a05fd081dd"
    example_limit = 10

    print(f"\n--- Content-Based Fetching for User: {example_user_id} ---")

    print(f"\nFetching 'new user' recommendations (limit {example_limit})...")
    new_data, new_styles = content_fetcher.fetch_new_user_data(
        example_user_id)
    print(f"User Styles: {new_styles}")
    print("Recommended Data Sample:")

    for item in new_data[:5]:
        print(f"  - {item['item_type']}: {item['name']}, Tags: {item['tags']}")

    print(
        f"\nFetching 'existing user' activity data (limit {example_limit})...")
    existing_data, existing_styles = content_fetcher.fetch_existing_user_data(
        example_user_id)

    print(f"User Styles: {existing_styles}")
    print("Visited/Attended Data Sample:")
    for item in existing_data[:5]:
        # Example to print the first row
        print(f"  - {item['item_type']}: {item['name']}, Tags: {item['tags']}")

    if not existing_data:
        print("  (No data found)")

    db_client.close()
