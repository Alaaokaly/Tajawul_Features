import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from neo4j import GraphDatabase, Driver, Session, Record, Result
from typing import Optional
import random 


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

    def get_user_styles(self, user_id):
        query = """MATCH (u:User {id: $user_id})-[:HAD_STYLE]->(s:Tag)
        RETURN COLLECT(s.name) AS style_name"""

        records = self.db.execute(query, {"user_id": user_id})
        return [record["style_name"] for record in records]  # styles

    def fetch_new_user_data(self, new_user=True, user_id=None, limit=15):
        query = """
            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:PREFERED_STYLE]->(s) 
            WITH u, s.name AS style

            MATCH (d:Destination)
            OPTIONAL MATCH (d)-[:HAD_STYLE]->(ds:Tag)
            OPTIONAL MATCH (d)-[:HAD_TYPE]->(dt:DestinationType)
            WHERE (style IS NULL OR (d)-[:HAS_STYLE]->(:Tag {name: style}))

            WITH d, 
                COLLECT(DISTINCT ds.name) AS tags, 
                COLLECT(DISTINCT dt.name) AS destinationType

            RETURN d.name AS name,
                   d.description AS description,
                   tags,
                   destinationType AS destinationType,
                   'Destination' AS type

            UNION

            MATCH (u:User {id: $user_id})
            OPTIONAL MATCH (u)-[:PREFERED_STYLE]->(s)
            WITH u, s.name AS style

            MATCH (e:Event)
            MATCH (e)-[:HAD_STYLE]->(es:Tag)
            WHERE (style IS NULL OR (e)-[:HAD_STYLE]->(:Tag {name: style}))

            WITH e, 
                COLLECT(DISTINCT es.name) AS tags

           RETURN e.name AS name,
                   e.description AS description,
                   tags,
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
        OPTIONAL MATCH (d)-[:HAD_STYLE]->(ds:Tag)
        OPTIONAL MATCH (d)-[:HAD_TYPE]->(dt:DestinationType)
        
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
    print(interactions_df.head())
    print(f"Total interactions fetched: {len(interactions_df)}")

    # #4. Fetch Content-Based Data (Example User)
    content_fetcher = ContentBasedFetcher(db_client)
    example_user_id = "99ae6489-05d2-49df-bb62-490a2a3f707b"
    example_limit = 10

    print(f"\n--- Content-Based Fetching for User: {example_user_id} ---")

    print(f"\nFetching 'new user' recommendations (limit {example_limit})...")
    new_data, new_styles = content_fetcher.fetch_new_user_data(
        example_user_id, example_limit)
    print(f"User Styles: {new_styles}")
    print("Recommended Data Sample:")

    for item in new_data[:5]:
        print(f"  - {item['type']}: {item['name']}, Tags: {item['tags']}")

    print(
        f"\nFetching 'existing user' activity data (limit {example_limit})...")
    existing_data, existing_styles = content_fetcher.fetch_existing_user_data(
        example_user_id, example_limit)

    print(f"User Styles: {existing_styles}")
    print("Visited/Attended Data Sample:")
    for item in existing_data[:5]:
        # Example to print the first row
        print(f"  - {item['type']}: {item['name']}, Tags: {item['tags']}")

    if not existing_data:
        print("  (No data found)")

    db_client.close()

    """# Neo4j Data Fetchers for Recommendation System

This repository contains Python classes designed to interact with a Neo4j graph database to fetch data relevant for building recommendation systems. It includes:

1.  A client class (`Neo4jClient`) to manage the connection and execution of Cypher queries against a Neo4j database.
2.  An interaction fetcher class (`InteractionsFetcher`) to retrieve user-item interactions (e.g., visits, wishes, reviews) with associated weights and normalization, suitable for collaborative filtering approaches.
3.  A content-based fetcher class (`ContentBasedFetcher`) to retrieve item features and user preferences/history, suitable for content-based filtering approaches.

## Features

*   **Neo4j Connection Management:** Securely connects to a Neo4j instance using credentials from environment variables. Handles connection verification and closure.
*   **Robust Query Execution:** Executes Cypher queries with parameter support and basic error handling.
*   **Interaction Data Retrieval:** Fetches interactions between `User` nodes and `Trip`, `Destination`, or `Event` nodes.
*   **Interaction Weighting:** Assigns predefined weights to different interaction types (e.g., `VISITED`, `WISHED`).
*   **Interaction Normalization:** Calculates a normalized interaction score (`avg`) per user based on their interaction weights using the formula `(weight - user_mean_weight) / (user_max_weight - user_min_weight)`.
*   **Content Data Retrieval:**
    *   Fetches potential items (`Destination`, `Event`) for *new users* based on their profile preferences (activity, duration, group size, style tags).
    *   Fetches items (`Destination`, `Event`) that an *existing user* has previously interacted with (`VISITED`, `ATTEND`).
    *   Retrieves user style preferences (`Tag` nodes connected via `HAD_STYLE`).
*   **Pandas Integration:** Returns fetched interaction data as a structured Pandas DataFrame.

## Prerequisites

*   Python 3.x
*   A running Neo4j database instance.
*   Required Python packages:
    *   `neo4j`
    *   `pandas`
    *   `python-dotenv`

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install neo4j pandas python-dotenv
    ```
    Alternatively, if you have a `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires Neo4j connection details to be set as environment variables. Create a `.env` file in the root directory of your project with the following content:

```dotenv
# .env file
NEO4J_URI=bolt://your_neo4j_host:7687 # Or neo4j://, neo4j+s:// etc.
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password"""