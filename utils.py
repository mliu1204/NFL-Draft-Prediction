import os
import pandas as pd

def clean_html_filenames():
    cache_dir = "cache"  # Adjust this path if your cache folder is elsewhere
    
    # Check if cache directory exists
    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' not found")
        return
    
    # Iterate through all files in the cache directory
    for filename in os.listdir(cache_dir):
        if ".html" in filename:
            # Find the position of .html in the filename
            html_pos = filename.find(".html")
            # Create new filename by keeping everything up to and including .html
            new_filename = filename[:html_pos + 5]  # +5 to include ".html"
            
            # Construct full file paths
            old_path = os.path.join(cache_dir, filename)
            new_path = os.path.join(cache_dir, new_filename)
            
            # Rename the file
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")
                    
def convert_feather_to_csv():
    try:
        # Read feather file
        df = pd.read_feather('data/college_stats.feather')
        
        # Write to CSV
        output_path = 'college_stats.csv'
        df.to_csv(output_path, index=False)
        print(f"Successfully converted data/college_stats.feather to {output_path}")
        
    except Exception as e:
        print(f"Error converting feather to CSV: {e}")
        
def analytics():
    df = pd.read_feather('data/processed/training.feather')
    
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
    
    # Count players with at least one non-zero stat
    players_with_stats = (df[target_stats] != 0).any(axis=1).sum()
    print(f"\nPlayers with at least one non-zero stat: {players_with_stats}")
    
    # Print total number of rows in dataset
    total_rows = len(df)
    print(f"\nTotal number of players: {total_rows}")

def main():
    analytics()

if __name__ == "__main__":
    main()