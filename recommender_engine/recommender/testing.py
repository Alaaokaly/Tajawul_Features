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

""" Evaluation :
                - Offline 
                - Online 
                - controlled group 
"""

""" Offline Evaluation : 
                        - split data  verify on one set that the recommender predicts items that
                           are close to what the user actually chooses in the hidden set 
                        -  see if the recommender can predict the type of interaction the user had with the item
                        -  test the predictions by the hidden set 
                        - measuer error/decision  matrix 
                        -  
                              """

""" Precision : number of relevent items between k-items/users 
               relevant item  can be an item that has higher frequency in relations than others
                - you want relevent items to be recommended more 
       
                  
    The average precision (AP) :
                               can be used to measure how good the rank is by running
                               the precision from 1 to m, where m is the number of items 
                               that are recommended (usually denoted as k) 
                                works per recommendation, so if you want to use it as a
                                measure to evaluate a recommender, you can then take the 
                                mean of the average (or mean average precision—MAP) over 
                                all recommendations
                                  
                                    
   discounted cumulative gain (DCG):
                                   Its about finding each relevant item and then penalizing 
                                   the items the farther down itis in the list.   
                                   the DCG looks at levels of relevancy.
                                   you give each item a relevancy score. 
                                   it can be the predicted rating or profit on the item. 
                                     """