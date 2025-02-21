import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
    left['key'] = left.apply(lambda x: f"{x['player']}-{x['year']}" if pd.isna(x['url']) else x['url'], axis=1)
    return left

def prepare_combine_data(combine_table):
    """Prepare combine table with key column"""
    right = combine_table[['year', 'player', 'pos', 'college', 'height', 'weight',
                          'forty', 'vertical', 'broad', 'bench', 'threecone',
                          'shuttle', 'url']].copy()
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
    training2 = college_stats.sort_values('url').groupby(['url', 'stat']).first().reset_index()
    training2 = training2.rename(columns={'url': 'key', 'stat': 'metric'}).drop('section', axis=1)
    training2['metric'] = training2['metric'].str.replace('.', '_')
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
    """Save processed data and splits"""
    training.to_feather('data/processed/training.feather')
    pd.Series(train_set).to_csv('data/processed/train_set.csv', index=False)
    pd.Series(test_set).to_csv('data/processed/test_set.csv', index=False)
    pd.Series(holdout_set).to_csv('data/processed/holdout_set.csv', index=False)

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
    
    # Print summary
    print_summary_statistics(training, train_set, test_set, holdout_set)
    
    return training, train_set, test_set, holdout_set

if __name__ == "__main__":
    training, train_set, test_set, holdout_set = prepare_data()
