import requests
import pandas as pd
# List of Arabic-speaking countries
arabic_countries = [
    "Egypt", "Algeria", "Saudi Arabia", "Morocco", "UAE", "Iraq",
    "Tunisia", "Yemen", "Syria", "Sudan", "Jordan", "Lebanon",
    "Libya", "Palestine", "Oman", "Kuwait", "Mauritania", "Bahrain",
    "Qatar", "Somalia", "Djibouti", "Comoros"
]

def get_festivals(country):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:Festivals_in_{country}",
        "cmlimit": 150
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    festivals = []
    if "query" in data:
        for item in data["query"]["categorymembers"]:
            festivals.append(item["title"])
    
    return festivals

data = []
for country in arabic_countries:
    festivals = get_festivals(country)
    if festivals:
        data.append({'COUNTRY':country,'EVENT':festivals})


df = pd.DataFrame(data,columns=['EVENT','COUNTRY','CITY'])
df.head()
