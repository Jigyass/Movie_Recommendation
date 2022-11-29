import pandas as pd 
from nltk.tokenize import RegexpTokenizer
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


def get_crew(list):
	names = []
	for i in literal_eval(list):
		if i['job'] == 'Director' or i['job'] == "Producer" or i['job'] == 'Casting' or i['job'] == 'Screenplay':
			names.append(i['name'].lower())
	return names

def get_names(list, n):
	names = []
	if n == 0:
		for i in literal_eval(list):
			names.append(i['name'].lower())
	else:
		length = n
		for i in literal_eval(list):
			names.append(i['name'].lower())
			if len(names) > length:
				break
	return names    


def rem_spaces(list):
	names = []
	for i in list:
		
		names.append(i.replace(" ", ""))
	return names   

def recommender(movie):
	ndx = X['original_title'] == movie
	index = X[ndx].index[0]
	kk = []
	for i in range(len(similarity[index])):
		kk.append([i, similarity[index][i]])
	recommend = sorted(kk, key = lambda x: x[1], reverse=True )
	for i in recommend[1:7]:
		print(X.iloc[i[0]].original_title)

feat = ['cast','crew_mem','genres','keywords', 'production_companies']
data_credit = pd.read_csv("tmdb_credits.csv")

data_movies = pd.read_csv("movies.csv")
data_credit = data_credit.rename(columns=({'movie_id': "id"}))
data_movies = data_movies.merge(data_credit, on='id')
data_movies.dropna(inplace=True)
tokenizer = RegexpTokenizer(r'\w+')
data_movies['overview'] = data_movies['overview'].apply(lambda X:tokenizer.tokenize(X.lower()))
data_movies['crew_mem'] = data_movies['crew'].apply(get_crew)




data_movies['genres'] = data_movies['genres'].apply(get_names,n=(0))
data_movies['cast'] = data_movies['cast'].apply(get_names,n=(0))
data_movies['keywords'] = data_movies['keywords'].apply(get_names, n=(5))
data_movies['production_companies'] = data_movies['production_companies'].apply(get_names,n=(0))

for i in feat:
	if i != "original_title":
		data_movies[i] = data_movies[i].apply(rem_spaces)


X = data_movies.drop(['popularity','vote_average','budget','homepage', 'original_language', 'crew','overview', 'title_y', 'vote_count','title_x','tagline', 'status','spoken_languages', 'production_countries','revenue','release_date','runtime'], axis=1)
feature = ['genres','keywords','production_companies','cast','crew_mem']

X['all'] = X['genres'] + X['keywords'] + X["production_companies"]  + X['cast'] + X['crew_mem']


X = X.drop(['genres','keywords','production_companies','cast','crew_mem'], axis =1)
X['all'] = X['all'].apply(lambda x:' '.join(x))


count_vect = CountVectorizer(max_features=10000, stop_words='english')  #count the number of texts
cv = count_vect.fit_transform(X['all']).toarray()  #into an array

similarity = cosine_similarity(cv) #similarity  matrix

recommender("Spectre")
