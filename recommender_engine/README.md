### Hybrid Travel Recommender System
This module is part of the Travel Assistant AI and provides personalized trip recommendations using a hybrid recommender system. It combines content-based filtering, collaborative filtering, and graph-based recommendations to suggest the best travel experiences for users.

# Features
✔ Cold-Start Handling → Uses onboarding questionnaires to collect initial user preferences.
✔ Content-Based Filtering → Matches users to trips based on similarity in destinations, activities, and interests.
✔ Collaborative Filtering → Identifies similar users and recommends popular trips.
✔ Graph-Based Recommendations → Uses Neo4j to model relationships between users, destinations, and activities.
✔ Ranking & Personalization → Scores recommendations based on user behavior and preferences.

# Tech Stack
Machine Learning: Scikit-Learn, Pandas, NumPy
Graph Database: Neo4j
Deployment: Docker (Optional)
