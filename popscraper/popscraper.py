import string

import pandas as pd
import requests
from bs4 import BeautifulSoup as btsp
from urlpath import URL

base = URL("https://population.us/")
df   = pd.DataFrame(columns=["state", "county", "population_density"])

for char in string.ascii_lowercase:
    req = requests.get(base/'county'/char)
    letter_page = btsp(req.text, 'html.parser')
    link_wrapper = letter_page.find_all(class_ = "biglink")[0]
    for child in link_wrapper.findChildren("a"):
        county = btsp(requests.get(base/child["href"]).text, 'html.parser')
        county_name, state = child.text.split(", ")
        county_name = county_name.replace(" County", "")
        pop_density = next(_.text for _ in county.find_all('b') if "p/mi²" in _.text).split(" ")[0]
        print(county_name, state, pop_density)
        df = pd.concat([df, pd.DataFrame([[state, county_name, pop_density]], columns=["state", "county", "population_density"])])

df.sort_values("state").to_csv("us_county_popdens.csv", index=False)