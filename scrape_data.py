from io import StringIO
import os
from pathlib import Path
import random
import time
from typing import List

import requests
import pandas as pd
from bs4 import BeautifulSoup

from selenium_scraper import SeleniumScraper

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping.log'),
        logging.StreamHandler()
    ]
)

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Create necessary directories if they don't exist
Path('cache').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)

# Define headers for college stats tables, mirroring those used in the R script
HEADERS = {
    'defense_standard': ['year', 'school', 'conf', 'class', 'pos', 'games', 'solo.tackes', 'ast.tackles',
                'tackles', 'loss.tackles', 'sacks', 'int', 'int.yards', 'int.yards.avg', 'int.td',
                'pd', 'fum.rec', 'fum.yds', 'fum.tds', 'fum.forced'],
    'scoring_standard': ['year', 'school', 'conf', 'class', 'pos', 'games', 'td.rush', 'td.rec', 'td.int',
                'td.fr', 'td.pr', 'td.kr', 'td.oth', 'td.tot', 'kick.xpm', 'kick.fgm', 'twopm',
                'safety', 'total.pts'],
    'punt_return_standard': ['year', 'school', 'conf', 'class', 'pos', 'games',
                 'punt.returns', 'punt.return.yards', 'punt.return.avg', 'punt.return.td',
                 'kick.returns', 'kick.return.yards', 'kick.return.avg', 'kick.return.td'],
    'receiving_standard': ['year', 'school', 'conf', 'class', 'pos', 'games',
                  'receptions', 'rec.yards', 'rec.avg', 'rec.td',
                  'rush.att', 'rush.yds', 'rush.avg', 'rush.td',
                  'scrim.plays', 'scrim.yds', 'scrim.avg', 'scrim.tds'],
    'rushing_standard': ['year', 'school', 'conf', 'class', 'pos', 'games',
                'receptions', 'rec.yards', 'rec.avg', 'rec.td',
                'rush.att', 'rush.yds', 'rush.avg', 'rush.td',
                'scrim.plays', 'scrim.yds', 'scrim.avg', 'scrim.tds'],
    'passing_standard': ['year', 'school', 'conf', 'class', 'pos', 'games',
                'completions', 'attempts', 'comp.pct', 'pass.yards',
                'yards.per.attempt', 'adj.yards.per.attempt', 'pass.tds',
                'pass.ints', 'int.rate']
}

# Define column headers for draft and combine pages
DRAFT_HEADER = ['round', 'pick', 'team', 'player', 'pos', 'age', 'to', 'ap1', 'pb', 'st',
                'carav', 'drav', 'games', 'pass.cmp', 'pass.att', 'pass.yds', 'pass.tds',
                'pass.ints', 'rush.att', 'rush.yds', 'rush.tds', 'receptions', 'rec.yds',
                'rec.tds', 'tackles', 'ints', 'sacks', 'college', 'stats']

COMBINE_HEADER = ['player', 'pos', 'college', 'stats', 'height', 'weight', 'forty', 'vertical',
                  'bench', 'broad', 'threecone', 'shuttle', 'drafted']

def requests_read_html_cache(url: str, cache_dir: str = 'cache', file_name: str = None) -> BeautifulSoup:
    """
    Download and cache HTML content from a URL.
    If the file already exists in the cache, it is read from disk.
    """
    fn = url.split('/')[-1]
    if file_name is not None:
        fn = file_name + '.htm'
    fn_path = os.path.join(cache_dir, fn)
    if not os.path.exists(fn_path):
        time.sleep(random.randint(5, 6))
        response = requests.get(url)
        response.raise_for_status()
        with open(fn_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    with open(fn_path, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'html.parser')
    
def stats_requests_read_html_cache(url: str, cache_dir: str = 'cache', file_name: str = None) -> BeautifulSoup:
    """
    Download and cache HTML content from a URL.
    If the file already exists in the cache, it is read from disk.
    """
    fn = url.split('/')[-1]
    if file_name is not None:
        fn = file_name + '.html'
    fn_path = os.path.join(cache_dir, fn)
    if not os.path.exists(fn_path):
        time.sleep(random.randint(5, 6))
        response = requests.get(url)
        response.raise_for_status()
        with open(fn_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    with open(fn_path, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'html.parser')

def read_html_cache(url: str, cache_dir: str = 'cache', scraper: SeleniumScraper = None, file_name: str = None) -> BeautifulSoup:
    """
    Download and cache HTML content from a URL.
    If the file already exists in the cache, it is read from disk.
    """
    fn_path = os.path.join(cache_dir, url.split('/')[-1])
    if file_name is not None:
        fn_path = os.path.join(cache_dir, file_name + '.htm')
        
    if not os.path.exists(fn_path):
        response = scraper.get_html(url)
        with open(fn_path, 'w', encoding='utf-8') as f:
            f.write(response)
    with open(fn_path, 'r', encoding='utf-8') as f:
        return BeautifulSoup(f.read(), 'html.parser')

def scrape_draft_data(years: List[int], scraper: SeleniumScraper) -> pd.DataFrame:
    """
    Scrape NFL draft data pages for the specified years.
    Extracts the main draft table and the URL (from 29th column) for each row.
    """
    all_data = []
    for year in years:
        url = f'http://www.pro-football-reference.com/years/{year}/draft.htm#drafts'
        # soup = requests_read_html_cache(url, file_name=f'draft_{year}')
        soup = read_html_cache(url, scraper=scraper, file_name=f'draft_{year}')
        table = soup.find('table', {'id': 'drafts'})
        if table:
            df_list = pd.read_html(StringIO(str(table)))
            if df_list:
                df = df_list[0]
                # Set column names if the table has the expected number of columns
                if df.shape[1] == len(DRAFT_HEADER):
                    df.columns = DRAFT_HEADER
                # Extract URLs from the 29th column (using CSS selector 'tr td:nth-child(29)')
                urls = []
                cells = table.select('tr td:nth-child(29)')
                for cell in cells:
                    a_tag = cell.find('a')
                    urls.append(a_tag.get('href') if a_tag else None)
                # Remove header rows that might have been parsed as data (where pos equals 'Pos')
                logging.info(df)
                df = df[df['pos'] != 'Pos'].copy()
                df['url'] = urls
                df['year'] = year
                all_data.append(df)
        else:
            logging.error("NO TABLE FOUND DUMBASS")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def scrape_combine_data(years: List[int], scraper: SeleniumScraper) -> pd.DataFrame:
    """
    Scrape NFL combine data pages for the specified years.
    Extracts the main combine table and the URL (from 4th column) for each row.
    """
    all_data = []
    for year in years:
        url = f'http://www.pro-football-reference.com/draft/{year}-combine.htm'
        soup = read_html_cache(url, scraper=scraper)
        table = soup.find('table', {'id': 'combine'})
        if table:
            df_list = pd.read_html(StringIO(str(table)))
            if df_list:
                df = df_list[0]
                if df.shape[1] == len(COMBINE_HEADER):
                    df.columns = COMBINE_HEADER
                # Extract URLs from the 4th column (using CSS selector 'tr td:nth-child(4)')
                urls = []
                cells = table.select('tr td:nth-child(4)')
                for cell in cells:
                    a_tag = cell.find('a')
                    urls.append(a_tag.get('href') if a_tag else None)
                df = df[df['pos'] != 'Pos'].copy()
                df['url'] = urls
                df['year'] = year
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def parse_pfr_tables(tables) -> pd.DataFrame:
    """
    Process and melt Pro Football Reference stat tables.
    For each HTML table with an id found in HEADERS, it:
      - Reads the table,
      - Removes the first and last rows (as done in the R code),
      - Renames the columns based on the HEADERS list,
      - Drops columns: 'year', 'school', 'conf', 'class', 'pos',
      - Adds a "seasons" column,
      - Melts the table so that each stat becomes a row,
      - Filters out empty values and converts the stat values to numeric.
    """
    results = []
    for table in tables:
        table_id = table.get('id')
        if table_id in HEADERS:
            try:
                df_list = pd.read_html(StringIO(str(table)))
                if not df_list:
                    continue
                df = df_list[0]
                
                # Handle multi-level columns by taking the lowest level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                
                # Rename 'Season' to 'year' if it exists
                df = df.rename(columns={'Season': 'year', 'Team': 'school', 'Conf': 'conf', 'Class': 'class', 'Pos': 'pos'})
                # Keep only the last row (Career stats)
                df = df.iloc[[-1]]
                # if df.shape[1] == len(HEADERS[table_id]):
                #     df.columns = HEADERS[table_id]
                # else:
                #     continue
                
                # Drop non-numeric identifying columns
                drop_cols = ['year', 'school', 'conf', 'class', 'pos', 'Awards']
                df_melt = df.drop(columns=drop_cols, errors='ignore')
                df_melt['seasons'] = 1
                # Melt the dataframe to have one row per stat
                melted = pd.melt(df_melt, id_vars=['seasons'], var_name='stat', value_name='value')
                # Remove empty string values and convert stat values to numeric
                melted = melted[melted['value'].astype(str) != '']
                melted['value'] = pd.to_numeric(melted['value'], errors='coerce')
                melted['section'] = table_id.replace('_standard', '')
                results.append(melted)
            except Exception as e:
                print(f"Error processing table {table_id}: {e}")
                continue
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def main():
    try:
        # scraper = SeleniumScraper()
        # ---------------------------
        # Step 1: Scrape Draft Data
        # ---------------------------
        drafts_file = 'data/drafts.feather'
        if not os.path.exists(drafts_file):
            draft_data = scrape_draft_data(list(range(2000, 2024)), scraper=scraper)
            if not draft_data.empty:
                draft_data.reset_index(drop=True, inplace=True)
                draft_data.to_feather(drafts_file)
        else:
            draft_data = pd.read_feather(drafts_file)
            draft_data.to_csv('data/drafts.csv', index=False)
        # ---------------------------
        # Step 2: Scrape Combine Data
        # ---------------------------
        combines_file = 'data/combines.feather'
        if not os.path.exists(combines_file):
            combine_data = scrape_combine_data(list(range(2000, 2024)), scraper=scraper)
            if not combine_data.empty:
                combine_data.reset_index(drop=True, inplace=True)
                combine_data.to_feather(combines_file)
        else:
            combine_data = pd.read_feather(combines_file)
            combine_data.to_csv('data/combines.csv', index=False)

        # ---------------------------------
        # Step 3: Combine URLs for College Stats
        # ---------------------------------
        # Merge the URL column from both draft and combine datasets.
        # Clean URLs by removing query parameters after .html
        draft_data['url'] = draft_data['url'].str.split('.html').str[0] + '.html'
        combine_data['url'] = combine_data['url'].str.split('.html').str[0] + '.html'
        urls_series = pd.concat([draft_data['url'], combine_data['url']], ignore_index=True)
        urls = urls_series.dropna().unique()

        # # ---------------------------------
        # # Step 4: Scrape College Stats from Each URL
        # # ---------------------------------
        college_stats_list = []
        for i, url in enumerate(urls):
            try:
                soup = stats_requests_read_html_cache(url)
                tables = soup.find_all('table', id=['defense_standard', 'passing_standard', 'rushing_standard', 'scoring_standard', 'kick_return_standard', 'kicking_standard', 'all_punt_return_standard', 'receiving_standard'])
                stats = parse_pfr_tables(tables)
                if not stats.empty:
                    # Group by 'section' and 'stat', summing the values just like the R code
                    grouped = stats.groupby(['section', 'stat'], as_index=False)['value'].sum()
                    grouped['url'] = url
                    college_stats_list.append(grouped)
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue

        
        college_stats_file = 'data/college_stats.feather'
        if college_stats_list:
            college_stats_df = pd.concat(college_stats_list, ignore_index=True)
            college_stats_df.to_feather(college_stats_file)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass

if __name__ == "__main__":
    main()
