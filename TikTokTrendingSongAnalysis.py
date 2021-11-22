import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import figure
import plotly
import plotly.express as px
from scipy.stats import norm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from math import pi
from SpotifyAPI import SpotifyAPI

spotify = SpotifyAPI(client_id, client_secret)
df = pd.read_csv("audd_music_spotify_music.csv")

# Data cleaning and pre-processing
df.drop_duplicates(inplace=True)
df = df[df['id'].notna()]
# Drop unneccessary columns
df.drop(['album.album_group', 'album.release_date_precision',
  'album.available_markets', 'linked_from', 'external_ids.isrc',
  'external_urls.spotify', "available_markets",
  "disc_number"], axis=1, inplace=True)
df = df.reset_index(drop=True)

# Use  track_ids & query function to get audio features & add to the df.
df["audio_features"] = df["id"].apply(spotify.query_track_audio_features)
# Audio features are a dictionaries. Convert to df and concatenate with original df.
audio_features_df = pd.DataFrame(list(df["audio_features"]))
df = pd.concat([df, audio_features_df], axis=1)
df = df.loc[:,~df.columns.duplicated()]

# The popularity of the artist. The value will be between 0 and 100 (100 = popular).
# Does previous popularity have any effect on how well it would do on tiktok?
popularity_sorted_df = df.sort_values('popularity', ascending = False)
popularity = popularity_sorted_df['popularity']

# Top ten most popular and least 10 popular songs trending on TikTok
popularity_sorted_df['name'].head(10)
popularity_sorted_df['name'].tail(10)

#Charting popularity
mean_popularity = popularity.mean() # ---> 53.01
median_popularity = popularity.median() # ---> 61.0
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(popularity, bins=bins, edgecolor="black")
plt.title("Song Popularity Level According to Spotify")
plt.xlabel("Popularity Range")
plt.ylabel("Number of Songs")
plt.axvline(mean_popularity, color='red', label="Popularity Mean")
plt.axvline(median_popularity, color='yellow', label="Popularity Median")
plt.tight_layout()
plt.legend()

# Charting Audio Features

audio_names = ['danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo']

sns.set_style('dark')
sns.displot(df.danceability, kde=True, bins=8, color='green').set(title='DANCEABILITY')
sns.displot(df.energy, kde=True, bins=8, color='purple').set(title='ENERGY')
sns.displot(df.valence, kde=True, bins=8, color='orange').set(title='VALENCE')
sns.displot(df.instrumentalness, kde=True, bins=8, color='yellow').set(title='INSTRUMENTALNESS')
sns.displot(df.speechiness, kde=True, bins=8, color='red').set(title='SPEECHINESS')
sns.displot(df.tempo, kde=True, bins=8, color='black').set(title='TEMPO')

# Charting Release Year

df['release_year'] = pd.DatetimeIndex(df["album.release_date"]).year
release_years = df['release_year']

mode_year = release_years.mode()
bins = [1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
x_min = 1969
x_max = 2020
mode_year = int(release_years.mode())
plt.hist(df['release_year'], bins=bins, edgecolor="black", rwidth=.8, log=True)
plt.axvline(mode_year, color='red', label="Mode year")
plt.title("Trending Songs by Release Year")
plt.xlabel("Years")
plt.ylabel("Number of Songs")
plt.tight_layout()
plt.legend()

# Spider Chart
audio_features_radar_chart = audio_features_df[['danceability', 'energy',
'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
'valence', 'tempo', 'duration_ms',]]
min_max_scaler = MinMaxScaler()
audio_features_radar_chart.loc[:]=min_max_scaler.fit_transform(audio_features_radar_chart.loc[:])

# Set the size and colors of the graph
fig=plt.figure(figsize=(12,8), facecolor="black", edgecolor="black")

# Convert audio features into list and save their count in a variable
categories=list(audio_features_radar_chart.columns)
N=len(categories)

# Save the averages for each audio feature
value=list(audio_features_radar_chart.mean())

# Append the first item to the end of the list to make a closed ciricle
value+=value[:1]
# Calculate angle for each category
angles=[n/float(N)*2*pi for n in range(N)]
angles+=angles[:1]

# Plot the radar chart with respective angles. Add colors and design.
plt.polar(angles, value)
plt.fill(angles,value,alpha=0.3, color="#25F4EE")
plt.grid(True,color='k',linestyle=':')
plt.title('TikTok Trending Audio Features', size=35, color="white", pad=40)
plt.xticks(angles[:-1],categories, size=15)
plt.xticks(color='#FE2C55',size=15)
plt.yticks(color='#FE2C55',size=15)
plt.tight_layout()
plt.show()






