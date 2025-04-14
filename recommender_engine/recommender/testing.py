"""Recommender Engines Testing :
   - we evaluate the pattern (taste) the user have 
   - Variation of the recommendations (don't make the rich, richer and the poor poorer) 
   - Serendipity 'surprise me', exploit-exlore testing 
   - coverage is to  recommend everything in your catalog"""


from neo4j_data_fetcher import Neo4jClient
from CF_KNN_user_based import UserBasedCF
from CF_KNN_item_based import ItemBasedCF


db_client = Neo4jClient()

model_user = UserBasedCF(db_client, k_neighbors=25, min_sim=0.001, min_overlap=0)
model_user.fit()

model_item = ItemBasedCF(db_client, k_neighbors=25, min_sim=0.001, min_overlap=0)
model_item.fit()

def calculate_coverage(model):
    items_in_rec = {}
    users_with_recs = []
    for user in model.user_id_to_index:
        
        recset = model.recommend(user, top_n=3)
        if recset:
            users_with_recs.append(user)

            for rec in recset:
                items_in_rec[rec['item']]  +=1
    no_items = model.user_item_matrix['item']
    no_items_in_rec = len(items_in_rec.items())
    no_users  = len(model.user_id_to_index)
    no_users_in_rec = len(users_with_recs)
    user_coverage = float(no_users_in_rec/ no_users)
    item_coverage = float(no_items_in_rec/ no_items)
    return user_coverage, item_coverage

