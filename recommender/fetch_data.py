import os
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from py2neo import Graph
from typing import Optional, List, Tuple

# Load environment variables
load_dotenv()

# Neo4j Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validate credentials
if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
    raise ValueError("Neo4j credentials are missing! Check your .env file.")

def fetch_interactions():

    """Fetch user interactions (Trips, Destinations, Events) from Neo4j."""
    query = """
    MATCH (u:User)-[r]->(t:Trip)
    RETURN u.id AS user, t.id AS item, 'Trip' AS item_type, type(r) AS interaction
    UNION
    MATCH (u:User)-[r]->(d:Destination)
    RETURN u.id AS user, d.id AS item, 'Destination' AS item_type, type(r) AS interaction
    UNION
    MATCH (u:User)-[r]->(e:Event)
    RETURN u.id AS user, e.id AS item, 'Event' AS item_type, type(r) AS interaction
    """

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            results = session.run(query)
            data = [(record["user"], record["item"], record["item_type"], record["interaction"]) for record in results]
            
            # Convert results to DataFrame
            df = pd.DataFrame(data, columns=["user", "item", "item_type", "interaction"])
            df[["user", "item", "item_type"]] = df[["user", "item", "item_type"]].astype('category')

            # Define interaction weights
            interaction_weights = {
                "VISITED": 5, "WISHED": 3, "SEARCHED_FOR": 2, 
                "REVIEWED": 4, "FAVORITED": 4, "CLONED": 5, "CREATED": 2
            }
            df["weight"] = df["interaction"].map(interaction_weights)

    return df

def fetc_newuser_data(new_user=True, user_id=None, limit=None):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    user_styles = graph.run("""
        MATCH (u:User {id: $user_id})-[:HAD_STYLE]->(s:Tag)
        RETURN COLLECT(s.name) AS styles
    """, user_id=user_id).data()[0]['styles']
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
    results = graph.run(query, user_id=user_id, limit=limit).data()
    return results, user_styles
    
def fetch_cb_data (new_user=True, user_id=None, limit=None):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    user_styles = graph.run("""
        MATCH (u:User {id: $user_id})-[:HAD_STYLE]->(s:Tag)
        RETURN COLLECT(s.name) AS styles
    """, user_id=user_id).data()[0]['styles']
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
    results = graph.run(query, user_id=user_id, limit=limit).data()
    return results, user_styles