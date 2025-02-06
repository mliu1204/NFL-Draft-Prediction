import pandas as pd
import re

# Read the file and parse it into a DataFrame
data = []
with open('movie_gender_ethnicity.txt', 'r') as file:
    current_movie = {}
    line_count = 0
    
    for line in file:
        # Split the line into movie_id and values
        parts = line.strip().split()
        if len(parts) == 3:  # Each line has 3 parts: movie_id, category, value
            movie_id = parts[0]
            category = parts[1]
            
            # Skip rows where value is None
            if parts[2] == 'None':
                value = 'None'  # or None if you prefer
            else:
                value = float(parts[2])
            
            # Map the category names to more readable versions
            category_mapping = {
                'black': 'Black',
                'east_asian': 'East Asian',
                'hispanic_latino': 'Hispanic/Latino',
                'south_asian': 'South Asian',
                'white': 'White',
                'other': 'Other',
                'male': 'Male',
                'female': 'Female'
            }
            
            if category in category_mapping:
                current_movie['Name'] = movie_id
                current_movie[category_mapping[category]] = value
                line_count += 1
            
            # After 8 lines (complete movie data), append and reset
            if line_count == 8:
                data.append(current_movie)
                current_movie = {}
                line_count = 0

# Convert to DataFrame and ensure column order
df = pd.DataFrame(data)
column_order = ['Name', 'Black', 'East Asian', 'Hispanic/Latino', 
                'South Asian', 'White', 'Other', 'Male', 'Female']
df = df[column_order]

# Display first few rows
print(df.head())

df.to_csv('movie_gender_ethnicity.csv', index=False)
