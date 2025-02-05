import pandas as pd

df = pd.read_csv('movies_metadata.csv')

df = df[['title', 'budget', 'genres', 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime']]

df.to_csv('cleaned_movies.csv', index=False)


