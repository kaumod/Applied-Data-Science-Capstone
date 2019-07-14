# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:50:05 2019

@author: LENOVO
"""
import numpy as np # library to handle data in a vectorized manner
import geopy
import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes
 # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

# for webscraping import Beautiful Soup 
from bs4 import BeautifulSoup

import xml

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')

# scrape Wikipedia 
import requests
url = ('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')
result = requests.get(url)
print(url)
print(result.status_code)
print(result.headers)

from bs4 import BeautifulSoup
soup = BeautifulSoup(result.content, 'html.parser')
table = soup.find('table')
trs = table.find_all('tr')
rows = []
for tr in trs:
    i = tr.find_all('td')
    if i:
        rows.append(i)
        
table_post = soup.find('table')
fields = table_post.find_all('td')

postcode = []
borough = []
neighbourhood = []

for i in range(0, len(fields), 3):
    postcode.append(fields[i].text.strip())
    borough.append(fields[i+1].text.strip())
    neighbourhood.append(fields[i+2].text.strip())
        
df_pc = pd.DataFrame(data=[postcode, borough, neighbourhood]).transpose()
df_pc.columns = ['postcode', 'Borough', 'Neighbourhood']
df_pc.head()

#Convert to DataFrame
cols = ['postcode', 'Borough', 'Neighbourhood']
df_pcn = pd.DataFrame(lst, columns=cols)
print(df_pcn.shape)
# df[df.duplicated(['PostalCode'], keep=False)] - this would have shown the duplicate PostalCodes

df_pcn = df_pcn.groupby('postcode').agg(
    {
        'Borough':'first', 
        'Neighbourhood': ', '.join,}
    ).reset_index()

df_pcn.loc[df_pcn['postcode'] == 'M5A']

df_pcn.shape




df_geo = pd.read_csv('http://cocl.us/Geospatial_data')
df_geo.columns = ['postcode', 'Latitude', 'Longitude']


df_pos = pd.merge(df_pcn, df_geo, on=['postcode'], how='inner')

df_tor = df_pos[['Borough', 'Neighbourhood', 'postcode', 'Latitude', 'Longitude']].copy()

df_tor.head()

address = 'Toronto, Canada'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of the City of Toronto are {}, {}.'.format(latitude, longitude))


# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_tor['Latitude'], df_tor['Longitude'], df_tor['Borough'], df_tor['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=3,
        popup=label,
        color='green',
        fill=True,
        fill_color='#3199cc',
        fill_opacity=0.3,
        parse_html=False).add_to(map_toronto)  
    
map_toronto

map_toronto.save("mymap.html")