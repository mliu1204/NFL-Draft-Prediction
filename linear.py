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


if __name__ == "__main__":
    convert_feather_to_csv()