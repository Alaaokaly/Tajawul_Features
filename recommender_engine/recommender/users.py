from faker import Faker
import random
import uuid
from neo4j import GraphDatabase
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import os 
# Initialize Faker
faker = Faker()



URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print("Connecting to Neo4j at:", URI)  # optional debug

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))




# Connection
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

trip_durations = ['day', '3 days', 'week', '2 weeks', 'month', 'more than a month']
group_sizes = ['solo', 'couple', 'family', 'group', 'big-group']
budgets = ['low', 'midRange', 'luxury']

def create_user(tx, user_data):
    tx.run("""
        CREATE (:User {
            id: $id,
            firstName: $firstName,
            lastName: $lastName,
            bio: $bio,
            phoneNumber: $phoneNumber,
            birthDate: $birthDate,
            profileImage: $profileImage,
            creationDate: $creationDate,
            lastEditDate: $lastEditDate,
            PostsCount: $PostsCount,
            wishedDestinationCount: $wishedDestinationCount,
            visitedDestinationCount: $visitedDestinationCount,
            createdTripCount: $createdTripCount,
            favoriteTripCount: $favoriteTripCount,
            clonedTripCount: $clonedTripCount,
            wishedTripCount: $wishedTripCount,
            editedDestinationCount: $editedDestinationCount,
            followedDestinationCount: $followedDestinationCount,
            createdDestinationCount: $createdDestinationCount,
            favoriteDestinationCount: $favoriteDestinationCount,
            topTraveler: $topTraveler,
            socialMediaLinks: $socialMediaLinks,
            tripDuration: $tripDuration,
            groupSize: $groupSize,
            budget: $budget
        })
    """, user_data)

def generate_fake_user():
    return {
        "id": str(uuid.uuid4()),
        "firstName": faker.first_name(),
        "lastName": faker.last_name(),
        "bio": faker.text(max_nb_chars=100),
        "phoneNumber": faker.phone_number(),
        "birthDate": faker.date_of_birth(minimum_age=18, maximum_age=70).isoformat(),
        "profileImage": faker.image_url(),
        "creationDate": faker.date_this_decade().isoformat(),
        "lastEditDate": datetime.now().isoformat(),
        "PostsCount": random.randint(0, 100),
        "wishedDestinationCount": random.randint(0, 20),
        "visitedDestinationCount": random.randint(0, 50),
        "createdTripCount": random.randint(0, 10),
        "favoriteTripCount": random.randint(0, 15),
        "clonedTripCount": random.randint(0, 5),
        "wishedTripCount": random.randint(0, 20),
        "editedDestinationCount": random.randint(0, 5),
        "followedDestinationCount": random.randint(0, 30),
        "createdDestinationCount": random.randint(0, 10),
        "favoriteDestinationCount": random.randint(0, 10),
        "topTraveler": random.choice([True, False]),
        "socialMediaLinks": [faker.url() for _ in range(random.randint(1, 3))],
        "tripDuration": random.choice(trip_durations),
        "groupSize": random.choice(group_sizes),
        "budget": random.choice(budgets)
    }

# Insert 200 users
with driver.session() as session:
    for _ in range(200):
        user = generate_fake_user()
        session.write_transaction(create_user, user)

print("✅ 200 synthetic users created in Neo4j.")

driver.close()

