#!/usr/bin/env python
# coding: utf-8

# In[251]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


# In[252]:


from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import spotipy.util as util
import time
import pandas as pd
from getpass import getpass


# In[257]:


#Initialize SpotiPy with user credentias
client_id="a4be012f20d04441b3c176e7b02b27bb"
client_secret="f6f0364dbe9544dcb976e9af52855cc7"


# In[258]:


client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# In[259]:


playlist_b=sp.user_playlist('xlaibarra26', '3BoFIY8RSTLCfukg7AXqZb') 
playlist_b # in order to check the structure and the to generalize and iterate


# In[260]:


def getTrackIDs(user, playlist_id):
    ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids

ids1 = getTrackIDs('xlaibarra26', '3BoFIY8RSTLCfukg7AXqZb')
#ids2 = getTrackIDs('xlaibarra26', '6BD6BEs8x6SwCNpW7FtKVl')
#ids3 = getTrackIDs('xlaibarra26', '7BxeYYWhL4Uz9YpjtQmNtE')
#ids4 = getTrackIDs('xlaibarra26', '7shgaZ5DnHwHOdumZYgQcB')
#ids5 = getTrackIDs('xlaibarra26', '14yiuyIZTsXJr9PlvUq7vu')


# #### Note: the idea was to use five playlists to build a consistent database, which would bring 11700 songs. Due to the limitiations of my computer, my only option was to work just with one playlist.

# In[261]:


ids=ids1#+ids2 #+ids3+ids4+ids5
print(len(ids))
print(ids)


# In[262]:


sp.track('1YrDXC5z5rbJnbzCQOjOgQ')['album']['artists'][0]['id']


# In[263]:


sp.audio_features('1YrDXC5z5rbJnbzCQOjOgQ')


# In[264]:


def getTrackFeatures(id):
  song_data = sp.track(id)
  song_feat = sp.audio_features(id)
 

  # songs' data
  name = song_data['name']
  artist_id=song_data['album']['artists'][0]['id']
  album = song_data['album']['name']
  artist = song_data['album']['artists'][0]['name']
  release_date = song_data['album']['release_date']
  length = song_data['duration_ms']
  popularity = song_data['popularity']
    
    
  # songs' features
  acousticness = song_feat[0]['acousticness']
  danceability = song_feat[0]['danceability']
  energy = song_feat[0]['energy']
  instrumentalness = song_feat[0]['instrumentalness']
  liveness = song_feat[0]['liveness']
  loudness = song_feat[0]['loudness']
  speechiness = song_feat[0]['speechiness']
  tempo = song_feat[0]['tempo']
  time_signature = song_feat[0]['time_signature']

  

  track = [name,artist_id, album, artist, release_date, length, popularity, danceability,acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
  return track


# In[266]:


# loop over track ids 
tracks = []
for i in range(len(ids)):
  time.sleep(.5)
  track = getTrackFeatures(ids[i])
  tracks.append(track)

# create dataset
df_p = pd.DataFrame(tracks, columns = ['name','artist_id', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
df_p.to_csv("xabis_playlists.csv", sep = ',')


# In[267]:


df_p


# In[268]:


artist_list=[]
for i in df_p['artist']:
    if i not in artist_list:
        artist_list.append(i)


# In[269]:


artist_list, len(artist_list)


# In[270]:


artist_id_list=[]
for i in df_p['artist_id']:
    if i not in artist_id_list:
        artist_id_list.append(i)
        
artist_id_list


# In[272]:


len(artist_list)


# In[271]:


#album_name=[]
album_code=[]
# here we are going to store all the albums of the artist we found at the beginning in the 5 spotify playlists.
for i in artist_id_list:
    for j in range(len(sp.artist_albums(i, album_type='album')['items'])):
        #album_name.append(sp.artist_albums(i, album_type='album')['items'][j]['name'])
        album_code.append(sp.artist_albums(i, album_type='album')['items'][j]['id'])


# In[ ]:


#len(album_name)==len(album_code)


# In[273]:


all_album_songs=[]
for i in range(len(album_code)):
    for j in range(len(sp.album_tracks(album_code[i])['items'])):
        all_album_songs.append(sp.album_tracks(album_code[i], limit=50, offset=0)['items'][j]['id'])


# In[274]:


all_album_songs


# In[278]:


len(all_album_songs)


# In[279]:


# all_songs = []

#for i in range(len(all_album_songs)):
  #time.sleep(.5)
  #track1 = getTrackFeatures(all_album_songs[i])
  #all_songs.append(track)
    
#Original code: I divided the 2776 in 7 blocks in order to make the code execution faster/possible.


# In[280]:


all_album_songs1=all_album_songs[0:350]
all_album_songs2=all_album_songs[350:700]
all_album_songs3=all_album_songs[700:1050]
all_album_songs4=all_album_songs[1050:1250]
all_album_songs5=all_album_songs[1400:1750]
all_album_songs6=all_album_songs[2150:2500]
all_album_songs7=all_album_songs[2500:2776]


# In[281]:


all_songs1 = []

for i in range(len(all_album_songs1)):
  time.sleep(.5)
  track1 = getTrackFeatures(all_album_songs1[i])
  all_songs1.append(track1)


# In[282]:


all_songs2 = []

for i in range(len(all_album_songs2)):
  time.sleep(.5)
  track2 = getTrackFeatures(all_album_songs2[i])
  all_songs2.append(track2)


# In[283]:


all_songs3 = []

for i in range(len(all_album_songs3)):
  time.sleep(.5)
  track3 = getTrackFeatures(all_album_songs3[i])
  all_songs3.append(track3)


# In[284]:


all_songs4 = []

for i in range(len(all_album_songs4)):
  time.sleep(.5)
  track4 = getTrackFeatures(all_album_songs4[i])
  all_songs4.append(track4)


# In[285]:


all_songs5 = []

for i in range(len(all_album_songs5)):
  time.sleep(.5)
  track5 = getTrackFeatures(all_album_songs5[i])
  all_songs5.append(track5)


# In[286]:


all_songs6 = []

for i in range(len(all_album_songs6)):
  time.sleep(.5)
  track6 = getTrackFeatures(all_album_songs6[i])
  all_songs6.append(track6)


# In[287]:


all_songs7 = []

for i in range(len(all_album_songs7)):
  time.sleep(.5)
  track7 = getTrackFeatures(all_album_songs7[i])
  all_songs7.append(track7)


# In[288]:


all_songs=[all_songs1,all_songs2,all_songs3,all_songs4,all_songs5,all_songs6,all_songs7]


# In[289]:


frames=[]
for i in range(len(all_songs)):
    df=pd.DataFrame(all_songs[i], columns = ['name','artist_id', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
    frames.append(df)
    
result=pd.concat(frames)


# In[290]:


result.shape


# In[291]:


result


# In[292]:


result.to_csv("all_songs_result.csv", sep = ',')


# ### We are going to scrap the 100 most popular songs and their artists' information

# In[293]:


r = requests.get('https://www.billboard.com/charts/hot-100')
r.status_code


# In[294]:


html = r.content
html


# In[295]:


soup = BeautifulSoup(html, 'html.parser')
soup


# In[296]:


soup.title.get_text(strip=True)


# In[297]:


soup.find_all("a")


# In[298]:


html_table = soup.find_all('div', attrs={'class': 'o-chart-results-list-row-container'})


# In[299]:


html_table


# In[300]:


soup.find_all("h3")


# In[301]:


songs = [i.get_text(strip=True) for i in soup.find_all("h3", attrs={'class': "c-title a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 lrv-u-font-size-18@tablet lrv-u-font-size-16 u-line-height-125 u-line-height-normal@mobile-max a-truncate-ellipsis u-max-width-330 u-max-width-230@tablet-only" })]
songs


# In[302]:


damn = [i.get_text(strip=True) for i in soup.find_all("h3", attrs={'class': "c-title a-no-trucate a-font-primary-bold-s u-letter-spacing-0021 u-font-size-23@tablet lrv-u-font-size-16 u-line-height-125 u-line-height-normal@mobile-max a-truncate-ellipsis u-max-width-245 u-max-width-230@tablet-only u-letter-spacing-0028@tablet" })]
damn


# In[303]:


all_songs=damn+songs


# In[304]:


all_songs


# In[305]:


soup.find_all("li")


# In[306]:


soup.find_all("span")


# In[307]:


lizzo=[i.get_text(strip=True) for i in soup.find_all("span", attrs={'class': "c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only u-font-size-20@tablet" })]
lizzo


# In[308]:


artists=[i.get_text(strip=True) for i in soup.find_all("span", attrs={'class': "c-label a-no-trucate a-font-primary-s lrv-u-font-size-14@mobile-max u-line-height-normal@mobile-max u-letter-spacing-0021 lrv-u-display-block a-truncate-ellipsis-2line u-max-width-330 u-max-width-230@tablet-only" })]


# In[309]:


artists


# In[310]:


artist_names=lizzo+artists


# In[311]:


artist_names


# In[312]:


ranking=list(range(1,101))
d = {"This_week":ranking,"Artist_name":artist_names,"Song_title":all_songs}


# In[313]:


df=pd.DataFrame(d).set_index("This_week")


# In[314]:


df


# In[315]:


df.to_csv("top100_billboard.csv")


# ### Obtain all the songs of all the artists in my spotify playlists. 
# #### I am going to take the dataset directly from the previous lab (*Lab-Web scraping single page*), as running the code take to long

# In[316]:


df_my_spotify=pd.read_csv("all_songs_result.csv")
df_my_spotify.head() # index and "Unnamed:0" column have the same value. So we are going to reset the index


# In[317]:


df_my_spotify.tail(2)


# In[318]:


df_my_spotify.set_index("Unnamed: 0",inplace=True)


# ### In order to have a dataset with more songs, we are going to use a spotify dataset available in Kaggle.

# In[319]:


df_songs=pd.read_csv("tracks.csv")
df_songs.head()


# In[320]:


columns_songs=df_songs.columns


# In[321]:


columns_songs


# In[322]:


len(columns_songs)


# In[323]:


df_my_spotify.columns


# In[324]:


len(df_my_spotify.columns)


# In[325]:


to_remove=['id','valence','key','explicit','mode']


# In[326]:


df_songs=df_songs.rename(columns={"name":"song_name","id_artists":"artist_id","artists":"artist","duration_ms":"length"})


# In[327]:


df_my_spotify=df_my_spotify.rename(columns={"name":"song_name"})


# In[328]:


df_songs.columns


# In[329]:


df_songs=df_songs.drop(columns=to_remove, axis=1)


# In[330]:


df_songs.columns


# In[331]:


df_my_spotify=df_my_spotify.drop(columns='album',axis=1)


# In[332]:


len(df_my_spotify.columns),len(df_songs.columns)


# In[333]:


df_concat = pd.concat([df_songs, df_my_spotify], axis=0)


# In[334]:


df_concat.tail(2)


# In[335]:


df_concat.reset_index(inplace=True)


# In[336]:


df_concat.drop(columns="index",inplace=True)


# In[337]:


df_concat.tail(2)


# * We need to clean the **['** and **']** from "artist_id" and "artist" column.

# In[338]:



df_concat['artist_id']=df_concat['artist_id'].str.replace("[",'')
df_concat['artist_id']=df_concat['artist_id'].str.replace("']",'')

df_concat['artist']=df_concat['artist'].str.replace("[",'')
df_concat['artist']=df_concat['artist'].str.replace("]",'')


# In[339]:


df_concat['artist_id']=df_concat['artist_id'].str.replace("'",'')
df_concat['artist']=df_concat['artist'].str.replace("'",'')


# In[340]:


df_concat.head(5)


# # Next step: Song recommender

# In[341]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min


# In[342]:


X = df_concat[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo','time_signature']]
# y = df['artist']

scaler = StandardScaler()

X_prep = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_prep)

clusters = kmeans.predict(X_prep)

scaled_df = pd.DataFrame(X_prep, columns=X.columns)
scaled_df['song name'] = df_concat['song_name']
scaled_df['artist'] = df_concat['artist']
scaled_df['cluster'] = clusters
scaled_df


# In[343]:


song_name = input('Choose a song: ')


# In[344]:


if song_name.lower() in df.Song_title.str.strip().str.lower().values :
    print("Your song appears in the 100 Billboard list")
    
else:
    print("Ooooh!Your song is not that popular yet. Not in the 100 Billboard list")


# In[345]:


results = sp.search(q=f'track:{song_name}', limit=1)
track_id = results['tracks']['items'][0]['id']
audio_features = sp.audio_features(track_id)

df_ = pd.DataFrame(audio_features)
new_features = df_[X.columns]

scaled_x = scaler.transform(new_features)
cluster = kmeans.predict(scaled_x)

filtered_df = scaled_df[scaled_df['cluster'] == cluster[0]][X.columns]


closest, _ = pairwise_distances_argmin_min(np.array(scaled_x.copy(order='C')), np.array(filtered_df).copy(order='C'))


# In[346]:


closest


# In[347]:


scaled_df.loc[closest[0]]['song name'], scaled_df.loc[closest[0]]['artist']


# In[348]:


def recommend_song():
    # get song id
    song_name = input('Choose a song: ')
    results = sp.search(q=f'track:{song_name}', limit=1)
    track_id = results['tracks']['items'][0]['id']
    # get song features with the obtained id
    audio_features = sp.audio_features(track_id)
    # create dataframe
    df_ = pd.DataFrame(audio_features)
    new_features = df_[X.columns]
    # scale features
    scaled_x = scaler.transform(new_features)
    # predict cluster
    cluster = kmeans.predict(scaled_x)
    # filter dataset to predicted cluster
    filtered_df = scaled_df[scaled_df['cluster'] == cluster[0]][X.columns]
    # get closest song from filtered dataset
    closest, _ = pairwise_distances_argmin_min(np.array(scaled_x).copy(order='C'), np.array(filtered_df).copy(order='C'))
    # return it in a readable way
    print('\n[RECOMMENDED SONG]')
    return ' - '.join([scaled_df.loc[closest]['song name'].values[0], scaled_df.loc[closest]['artist'].values[0]])


# In[349]:


recommend_song()


# In[ ]:




