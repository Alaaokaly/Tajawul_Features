from py2neo import Graph
import pandas as pd

uri = "neo4j+s://f7a0c2b2.databases.neo4j.io"
username = "neo4j"
password = "Ml8aeOW5Ra0RlM5Wa6pAYa5_PnAN2PPxcKvzESoWZuE"


def Fetch(new_user=True, user_id=None, limit=None):
    graph = Graph(uri, auth=(username, password))

    user_styles = graph.run("""
        MATCH (u:User {id: $user_id})-[:HAD_STYLE]->(s:Tag)
        RETURN COLLECT(s.name) AS styles
    """, user_id=user_id).data()[0]['styles']

    if new_user:
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
    else:
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
