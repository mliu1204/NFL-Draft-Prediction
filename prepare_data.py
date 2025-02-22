import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

def coalesce(*args):
    """Combines multiple Series, taking first non-null value"""
    return pd.Series([next((y for y in row if pd.notna(y)), row[0]) 
                     for row in zip(*args)])

def load_raw_data():
    """Load raw data from feather files"""
    combine_table = pd.read_feather('data/combines.feather')
    draft_table = pd.read_feather('data/drafts.feather')
    college_stats = pd.read_feather('data/college_stats.feather')
    return combine_table, draft_table, college_stats

def prepare_draft_data(draft_table):
    """Prepare draft table with key column"""
    left = draft_table[['year', 'round', 'pick', 'team', 'player', 'college', 
                        'pos', 'age', 'carav', 'drav', 'url']].copy()
    # Clean URLs before creating key
    left['url'] = left['url'].str.extract(r'(.*?\.html)')
    left['key'] = left.apply(lambda x: f"{x['player']}-{x['year']}" if pd.isna(x['url']) else x['url'], axis=1)
    return left

def prepare_combine_data(combine_table):
    """Prepare combine table with key column"""
    right = combine_table[['year', 'player', 'pos', 'college', 'height', 'weight',
                          'forty', 'vertical', 'broad', 'bench', 'threecone',
                          'shuttle', 'url']].copy()
    # Clean URLs before renaming
    right['url'] = right['url'].str.extract(r'(.*?\.html)')
    right = right.rename(columns={
        'year': 'year_combine',
        'player': 'player_combine',
        'pos': 'pos_combine',
        'college': 'college_combine',
        'url': 'url_combine'
    })
    right['key'] = right.apply(lambda x: f"{x['player_combine']}-{x['year_combine']}" 
                              if pd.isna(x['url_combine']) else x['url_combine'], axis=1)
    return right.sort_values('key').groupby('key').first().reset_index()

def merge_tables(left, right):
    """Merge draft and combine tables"""
    combined = pd.merge(left, right, on='key', how='outer')
    combined['player'] = coalesce(combined['player'], combined['player_combine'])
    combined['pos'] = coalesce(combined['pos'], combined['pos_combine'])
    combined['college'] = coalesce(combined['college'], combined['college_combine'])
    combined['year'] = coalesce(combined['year'], combined['year_combine'])
    combined['url'] = coalesce(combined['url'], combined['url_combine'])
    return combined

def process_height_and_metrics(combined):
    """Process height measurements and convert to long format"""
    training1 = combined[['key', 'carav', 'height', 'weight', 'forty', 'vertical',
                         'bench', 'age', 'threecone', 'shuttle', 'broad']].copy()
    
    # Convert height
    training1['height'] = training1['height'].fillna('NA-NA')
    training1[['feet', 'inches']] = (training1['height']
                                    .str.split('-', expand=True)
                                    .replace('NA', np.nan))
    training1['height'] = (training1['feet'].astype(float) * 12 + 
                          training1['inches'].astype(float))
    training1 = training1.drop(['feet', 'inches'], axis=1)
    
    # Convert to long format
    training1 = pd.melt(training1, 
                        id_vars=['key'],
                        value_vars=['carav', 'height', 'weight', 'forty', 'vertical',
                                  'bench', 'age', 'threecone', 'shuttle', 'broad'],
                        var_name='metric',
                        value_name='value')
    training1 = training1[training1['value'].notna() & (training1['value'] != '')]
    training1['value'] = pd.to_numeric(training1['value'])
    return training1

def impute_combine_data(training1):
    """Impute missing combine measurements"""
    training1a = training1.pivot(index='key', columns='metric', values='value').reset_index()
    
    imputer = IterativeImputer(random_state=0)
    cols_to_impute = training1a.columns.difference(['key', 'carav'])
    training1b = training1a.copy()
    training1b[cols_to_impute] = imputer.fit_transform(training1a[cols_to_impute])
    
    return pd.melt(training1b,
                   id_vars=['key'],
                   value_vars=training1b.columns.difference(['key']))

def process_college_stats(college_stats):
    """Process college statistics"""
    # Clean URLs to only include up to .html
    college_stats['url'] = college_stats['url'].str.extract(r'(.*?\.html)')
    
    # Define the features we want to extract
    target_stats = [
        'games', 'seasons',
        'pass_cmp', 'pass_att', 'pass_yds', 'pass_int', 'pass_td',
        'rec_yds', 'rec_td', 'rec', 'rush_att', 'rush_yds', 'rush_td',
        'tackles_solo', 'tackles_combined', 'tackles_loss', 'tackles_assists',
        'fumbles_forced', 'fumbles_rec', 'fumbles_rec_tds', 'fumbles_rec_yds',
        'sacks', 'def_int', 'def_int_td', 'def_int_yards', 'pass_defended',
        'punt_ret', 'punt_ret_td', 'punt_ret_yds',
        'kick_ret', 'kick_ret_td', 'kick_ret_yds'
    ]
    
    # Filter for only the stats we want and aggregate by url
    training2 = (college_stats[college_stats['stat'].isin(target_stats)]
                 .sort_values('url')
                 .groupby(['url', 'stat'])
                 .first()
                 .reset_index()
                 .rename(columns={'url': 'key', 'stat': 'metric'}))
    
    # Ensure all target stats exist for each player (filling with 0 if missing)
    training2 = (training2.pivot(index='key', columns='metric', values='value')
                 .reindex(columns=target_stats)
                 .fillna(0)
                 .reset_index()
                 .melt(id_vars=['key'], value_vars=target_stats))
    
    return training2

def create_final_dataset(combined, training1c, training2):
    """Create final training dataset"""
    # Combine all features
    training3 = pd.concat([training1c, training2])
    training3 = training3.pivot(index='key', columns='metric', values='value').fillna(0).reset_index()
    
    # Process final dataset
    training = combined[['key', 'player', 'pick', 'pos', 'college', 'year', 'team']].copy()
    college_counts = training.groupby('college').size().reset_index(name='n_college_picks')
    training = training.merge(college_counts, on='college')
    training['short_college'] = np.where(training['n_college_picks'] < 50, 'SMALL SCHOOL', training['college'])
    training['pick'] = training['pick'].fillna(257).astype(float)
    return training.merge(training3, on='key', how='inner')

def create_data_splits(training, random_seed=42):
    """Create train/test/holdout splits"""
    N = len(training)
    np.random.seed(random_seed)
    train_set = (np.random.binomial(1, 0.9, N) == 1) & (training['year'] < 2016)
    test_set = (~train_set) & (training['year'] < 2016)
    holdout_set = ~(test_set | train_set)
    return train_set, test_set, holdout_set

def save_processed_data(training, train_set, test_set, holdout_set):
    """Save processed data and splits, with separate files for each draft year"""
    # Save the complete dataset
    training.to_feather('data/processed/training.feather')
    
    # Save the standard splits
    pd.Series(train_set).to_csv('data/processed/train_set.csv', index=False)
    pd.Series(test_set).to_csv('data/processed/test_set.csv', index=False)
    pd.Series(holdout_set).to_csv('data/processed/holdout_set.csv', index=False)
    
    # Save separate files for each draft year
    for year in training['year'].unique():
        year_mask = training['year'] == year
        year_data = training[year_mask]
        
        # Create directory if it doesn't exist
        year_dir = f'data/processed/by_year/{year}'
        os.makedirs(year_dir, exist_ok=True)
        
        # Save year-specific data
        year_data.to_feather(f'{year_dir}/training.feather')
        year_data.to_csv(f'{year_dir}/training.csv', index=False)

def print_summary_statistics(training, train_set, test_set, holdout_set):
    """Print summary statistics of the dataset"""
    print("\nDataset Summary:")
    print(f"Total samples: {len(training)}")
    print(f"Training samples: {sum(train_set)}")
    print(f"Test samples: {sum(test_set)}")
    print(f"Holdout samples: {sum(holdout_set)}")
    print("\nFeature Statistics:")
    print(training.describe())

def prepare_data():
    """Main function to run the data preparation pipeline"""
    # Load data
    combine_table, draft_table, college_stats = load_raw_data()
    
    # Prepare and merge tables
    left = prepare_draft_data(draft_table)
    right = prepare_combine_data(combine_table)
    combined = merge_tables(left, right)
    
    # Process features
    training1 = process_height_and_metrics(combined)
    training1c = impute_combine_data(training1)
    training2 = process_college_stats(college_stats)
    
    # Create final dataset
    training = create_final_dataset(combined, training1c, training2)
    
    # Create data splits
    train_set, test_set, holdout_set = create_data_splits(training)
    
    # Save data
    save_processed_data(training, train_set, test_set, holdout_set)
    
    print("\nData saved successfully!")
    print("Complete dataset saved to: data/processed/")
    print("Year-specific datasets saved to: data/processed/by_year/")
    
    return training, train_set, test_set, holdout_set

if __name__ == "__main__":
    training, train_set, test_set, holdout_set = prepare_data()
