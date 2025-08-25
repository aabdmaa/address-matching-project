import re
import pandas as pd
from rapidfuzz import fuzz
import usaddress
import torch
import numpy as np
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Transformer
from sklearn.neighbors import BallTree
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")


# Loading the data from CCS and GIS
apt_df_ccs = pd.read_csv('ccs_data.csv', dtype=str)
apt_df_gis = pd.read_csv('gis_data.csv', dtype=str)

# Removing NaN values from CCS and GIS fields
apt_df_ccs.fillna('', inplace=True)
apt_df_gis.fillna('', inplace=True)

# Coverting to lowercase, removing punctuation, and stripping white space for prefixes, street names, and suffixes in CCS
apt_df_ccs['New Prefix'] = apt_df_ccs['New Prefix']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()

apt_df_ccs['New Street Name'] = apt_df_ccs['New Street Name']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()

apt_df_ccs['New Suffix'] = apt_df_ccs['New Suffix']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()

# Removing punctuation, converting to lowercase, and stripping white space for services addresses in GIS (using for usaddress parsing)
apt_df_gis['SERVICEADDRESS'] = apt_df_gis['SERVICEADDRESS']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.replace(r'\s+', ' ', regex=True)\
    .str.strip()

# Manually joining known compound names before parsing
apt_df_gis['SERVICEADDRESS'] = apt_df_gis['SERVICEADDRESS'].str.replace(r'\bmc fall\b', 'mcfall', regex=True)
apt_df_gis['SERVICEADDRESS'] = apt_df_gis['SERVICEADDRESS'].str.replace(r'\bmc kinney\b', 'mckinney', regex=True)


# Normalizing CCS apartment info using New Occupancy Identifier
def normalize_ccs_apartment(text):
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    if '-' in text:
        text = text.split('-')[-1]
    text = re.sub(r'[^\w\d]', '', text)
    return text


apt_df_ccs['Normalized Apt'] = apt_df_ccs['New Occupancy Identifier'].apply(
    normalize_ccs_apartment)


# Function to parse SERVICEADDRESS from GIS and extract components using usaddress
def parse_service_address(address):
    try:
        parsed = usaddress.tag(address)[0]
        street_number = parsed.get('AddressNumber', '')
        street_prefix = parsed.get('StreetNamePreDirectional', '')
        street_name = parsed.get('StreetName', '')
        street_suffix = parsed.get('StreetNamePostType', '')
        return pd.Series([street_number, street_prefix, street_name, street_suffix])
    except usaddress.RepeatedLabelError:
        return pd.Series(['', '', '', ''])


# Applying parsing only to non-empty SERVICEADDRESS rows
apt_df_gis[['Parsed Street Number', 'Parsed Street Prefix', 'Parsed Street Name', 'Parsed Street Suffix']] = apt_df_gis['SERVICEADDRESS'].apply(
    lambda x: parse_service_address(x) if isinstance(x, str) and x.strip() else pd.Series(['', '', '', '']))

# Filling empty Parsed Street Number with STREETNUM, if available
apt_df_gis['Parsed Street Number'] = apt_df_gis.apply(
    lambda row: row['STREETNUM'] if row['Parsed Street Number'].strip(
    ) == '' and pd.notna(row['STREETNUM']) else row['Parsed Street Number'], axis=1)

# Coverting to lowercase, removing punctuation, and stripping white space for prefixes, street names, and suffixes in GIS
apt_df_gis['Parsed Street Prefix'] = apt_df_gis['Parsed Street Prefix']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()

apt_df_gis['Parsed Street Name'] = apt_df_gis['Parsed Street Name']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()

apt_df_gis['Parsed Street Suffix'] = apt_df_gis['Parsed Street Suffix']\
    .str.lower()\
    .str.replace(r'[^\w\s]', '', regex=True)\
    .str.strip()


# Function to extract structured apartment info from SLADDITIONALINFO
def extract_apartments(text):
    if not isinstance(text, str):
        return {"units": [], "ranges": []}

    text = text.lower()

    # Removing irrelevant phrases
    text = re.sub(r'(house meter|washer meters|w/ hm|office|club house|community center|street lighting)', '', text)

    # Cleaning up extra whitespace and punctuation
    text = re.sub(r'[^\w\s\-&,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Extracting numeric ranges (101-104)
    range_matches = re.findall(r'\b(\d{1,4})\s*[-–]\s*(\d{1,4})\b', text)
    ranges = [(int(start), int(end)) for start, end in range_matches if start.isdigit() and end.isdigit()]

    # Extracting individual numeric units (101, 2101)
    unit_matches = re.findall(r'\b(?:apt[s]?\.?|unit[s]?\.?|space[s]?\.?)?\s*(\d{1,4})\b', text)
    units = list(set(unit_matches))

    # Extracting letter-based units (A, B, C)
    letter_units = re.findall(r'\b(?:apt[s]?\.?|unit[s]?\.?)?\s*([a-z])\b', text)
    units.extend(letter_units)

    return {
        "units": sorted(set(units)),
        "ranges": ranges,
    }


# Applying to GIS dataset
apt_df_gis['Structured Apts'] = apt_df_gis['SLADDITIONALINFO'].apply(extract_apartments)


# Function for comparing street numbers
def compare_street_numbers(num1, num2):
    num1, num2 = str(num1).strip(), str(num2).strip()
    return 1.0 if num1 == num2 else 0.0


# Function for comparing street names using Jaccard similarity with k-shingles
def jaccard_k_shingles(str1, str2, k=2):
    str1, str2 = str(str1).strip(), str(str2).strip()
    shingles1 = {str1[i:i+k] for i in range(len(str1) - k + 1)}
    shingles2 = {str2[i:i+k] for i in range(len(str2) - k + 1)}
    intersection = shingles1 & shingles2
    union = shingles1 | shingles2
    return len(intersection) / len(union) if union else 0.0


# Function for comparing strings using RapidFuzz (Levenshtein/token-based similarity)
def rapidfuzz_score(str1, str2):
    str1, str2 = str(str1).strip(), str(str2).strip()
    score = fuzz.ratio(str1, str2)
    return score / 100


# Function for comparing strings using RapidFuzz's token sort ratio
def token_sort_similarity(str1, str2):
    str1, str2 = str(str1).strip(), str(str2).strip()
    score = fuzz.token_sort_ratio(str1, str2)
    return score / 100


# Function for comparing strings using RapidFuzz's token set ratio
def token_set_similarity(str1, str2):
    str1, str2 = str(str1).strip(), str(str2).strip()
    score = fuzz.token_set_ratio(str1, str2)
    return score / 100


# Creating blocking keys for both datasets
apt_df_ccs['blocking_key'] = (
    apt_df_ccs['New Street Number'].astype(str).str.strip() + ' ' +
    apt_df_ccs['New Street Name'].astype(str).str.strip()).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

apt_df_gis['blocking_key'] = (
    apt_df_gis['Parsed Street Number'].astype(str).str.strip() + ' ' +
    apt_df_gis['Parsed Street Name'].astype(str).str.strip()).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()


# Creating a MinHash signature from input text using k-shingles
def create_minhash(text, num_perm=128, k=2):
    m = MinHash(num_perm=num_perm)
    text = str(text).strip()
    shingles = {text[i:i+k] for i in range(len(text) - k + 1)}
    for shingle in shingles:
        m.update(shingle.encode('utf8'))
    return m


# Initializing lists to store MinHash signatures and indexing info for GIS records
minhash_gis = []
gis_index_to_number = []
gis_number_to_indices = defaultdict(list)

# Generating MinHash signatures for each GIS record and organize them by street number
for idx, row in apt_df_gis.iterrows():
    number = row['Parsed Street Number'].strip()
    key = row['blocking_key']
    m = create_minhash(key)
    minhash_gis.append(m)
    gis_index_to_number.append(number)
    gis_number_to_indices[number].append(idx)

# Creating the LSH index
lsh = MinHashLSH(threshold=0.6, num_perm=128)

# Inserting GIS records into the LSH index
for idx, m in enumerate(minhash_gis):
    lsh.insert(f"gis_{idx}", m)


# Apartment matching logic
def is_apartment_match(ccs_apt, gis_structured_apts):
    if not ccs_apt or not gis_structured_apts:
        return False

    ccs_apt = str(ccs_apt).strip().lower()

    try:
        ccs_num = int(ccs_apt)
    except ValueError:
        ccs_num = None

    # Checking against individual units
    for unit in gis_structured_apts.get("units", []):
        if ccs_apt == unit:
            return True
        if len(ccs_apt) == 1 and ccs_apt == unit:
            return True

    # Checking numeric containment in ranges
    if ccs_num is not None:
        for start, end in gis_structured_apts.get("ranges", []):
            if start <= ccs_num <= end:
                return True

    return False


# # Pipeline 1: string/semantic matching pipeline

# Caches for embeddings
ccs_embeddings = {}
gis_embeddings = {}


@delayed
def compute_match(ccs_idx, ccs_row):
    ccs_number = ccs_row['New Street Number'].strip()
    ccs_apt = str(ccs_row['Normalized Apt']).strip().lower()

    results = []
    for gis_idx in gis_number_to_indices.get(ccs_number, []):
        gis_structured_apts = apt_df_gis.loc[gis_idx, 'Structured Apts']

        if ccs_apt and isinstance(gis_structured_apts, dict):
            if not is_apartment_match(ccs_apt, gis_structured_apts):
                continue

        # Building full CCS name with apartment
        ccs_prefix = str(ccs_row['New Prefix']).strip().lower()
        ccs_name = str(ccs_row['New Street Name']).strip().lower()
        ccs_suffix = str(ccs_row['New Suffix']).strip().lower()
        ccs_full_name = f"{ccs_prefix} {ccs_name} {ccs_suffix} apt {ccs_apt}".strip()

        # Building full GIS name with apartment info
        gis_prefix = str(apt_df_gis.loc[gis_idx, 'Parsed Street Prefix']).strip().lower()
        gis_name = str(apt_df_gis.loc[gis_idx, 'Parsed Street Name']).strip().lower()
        gis_suffix = str(apt_df_gis.loc[gis_idx, 'Parsed Street Suffix']).strip().lower()
        gis_apt = f"apt {ccs_apt}" if ccs_apt else ""
        gis_full_name = f"{gis_prefix} {gis_name} {gis_suffix} {gis_apt}".strip()

        if not ccs_full_name or not gis_full_name:
            continue

        # Computing Jaccard similarity on full names
        jaccard_sim = jaccard_k_shingles(ccs_full_name, gis_full_name)

        if jaccard_sim >= 0.6:
            if ccs_idx not in ccs_embeddings:
                ccs_embeddings[ccs_idx] = model.encode(ccs_full_name, convert_to_tensor=True).cpu().numpy()
            if gis_idx not in gis_embeddings:
                gis_embeddings[gis_idx] = model.encode(gis_full_name, convert_to_tensor=True).cpu().numpy()

            semantic_sim = util.cos_sim(
                torch.from_numpy(ccs_embeddings[ccs_idx]),
                torch.from_numpy(gis_embeddings[gis_idx])).item()

            if semantic_sim >= 0.6:
                results.append({
                    'ccs_index': ccs_idx,
                    'gis_index': gis_idx,
                    'ccs_number': ccs_row['New Street Number'],
                    'gis_number': apt_df_gis.loc[gis_idx, 'Parsed Street Number'],
                    'ccs_name': ccs_full_name,
                    'gis_name': gis_full_name,
                    'jaccard_similarity': jaccard_sim,
                    'semantic_similarity': semantic_sim
                })
    return results


# Creating delayed tasks with tqdm progress bar
tasks = [compute_match(idx, row) for idx, row in tqdm(apt_df_ccs.iterrows(), total=len(apt_df_ccs), desc="Creating tasks")]

# Computing with Dask's progress bar
with ProgressBar():
    results = compute(*tasks)

flat_results = [match for sublist in results for match in sublist]
string_semantic_matches_df = pd.DataFrame(flat_results)

string_semantic_matches_df.to_pickle("full_string_semantic_matches.pkl")
print(f"Saved {len(string_semantic_matches_df)} matched address pairs to 'full_string_semantic_matches.pkl'")


# Pipeline 2: geospatial matching pipeline (with 50-meter filter)

# Ensuring coordinate fields are numeric
apt_df_gis['X'] = pd.to_numeric(apt_df_gis['X'], errors='coerce')
apt_df_gis['Y'] = pd.to_numeric(apt_df_gis['Y'], errors='coerce')
apt_df_ccs['PREMISE_LONG'] = pd.to_numeric(apt_df_ccs['PREMISE_LONG'], errors='coerce')
apt_df_ccs['PREMISE_LAT'] = pd.to_numeric(apt_df_ccs['PREMISE_LAT'], errors='coerce')

matches_df = pd.read_pickle("full_string_semantic_matches.pkl")

# Filtering unmatched CCS records
matched_ccs_indices = set(matches_df['ccs_index'])
unmatched_ccs_df = apt_df_ccs[~apt_df_ccs.index.isin(matched_ccs_indices)].copy()

# Removing CCS records with 0.0 coordinates
unmatched_ccs_df = unmatched_ccs_df[
    (unmatched_ccs_df['PREMISE_LAT'] != 0) &
    (unmatched_ccs_df['PREMISE_LONG'] != 0)].copy()

# Converting GIS X/Y to lat/lon
transformer = Transformer.from_crs("EPSG:2903", "EPSG:4326", always_xy=True)
apt_df_gis[['GIS_LON', 'GIS_LAT']] = apt_df_gis.apply(
    lambda row: pd.Series(transformer.transform(row['X'], row['Y'])) if pd.notnull(
        row['X']) and pd.notnull(row['Y']) else pd.Series([None, None]), axis=1)

# Preparing GIS coordinates in radians
gis_coords = apt_df_gis[['GIS_LAT', 'GIS_LON']].dropna().copy()
gis_coords_rad = np.radians(gis_coords[['GIS_LAT', 'GIS_LON']].values)

# Building BallTree for GIS coordinates
tree = BallTree(gis_coords_rad, metric='haversine')

# Preparing unmatched CCS coordinates in radians
ccs_coords = unmatched_ccs_df[['PREMISE_LAT', 'PREMISE_LONG']].dropna().copy()
ccs_coords_rad = np.radians(ccs_coords[['PREMISE_LAT', 'PREMISE_LONG']].values)

# Querying nearest neighbor
distances, indices = tree.query(ccs_coords_rad, k=1)


# Defining delayed task for each CCS record (with 50-meter filter)
@delayed
def process_match(i):
    ccs_idx = ccs_coords.iloc[i].name
    gis_idx = indices[i][0]
    distance_meters = distances[i][0] * 6371000  # Converting radians to meters

    if distance_meters <= 50:
        return {
            'ccs_index': ccs_idx,
            'gis_index': apt_df_gis.index[gis_idx],
            'ccs_lat': unmatched_ccs_df.loc[ccs_idx, 'PREMISE_LAT'],
            'ccs_lon': unmatched_ccs_df.loc[ccs_idx, 'PREMISE_LONG'],
            'gis_lat': gis_coords.iloc[gis_idx]['GIS_LAT'],
            'gis_lon': gis_coords.iloc[gis_idx]['GIS_LON'],
            'distance_meters': distance_meters
        }
    else:
        return None


tasks = [process_match(i) for i in tqdm(
    range(len(ccs_coords)), desc="Creating tasks", unit="record")]

with ProgressBar():
    results = compute(*tasks)

flat_results = [r for r in results if r is not None]
geospatial_matches_df = pd.DataFrame(flat_results)
geospatial_matches_df.to_pickle("geospatial_matches.pkl")
print(f"\nSaved {len(geospatial_matches_df)} geospatial matches to 'geospatial_matches.pkl'")


# Graph to show the level of frequency and distribution of distances between matched geospatial points
df = pd.read_pickle("geospatial_matches.pkl")

plt.figure(figsize=(10, 6))
sns.histplot(df['distance_meters'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Geospatial Match Distances")
plt.xlabel("Distance (meters)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# Graphs to show the closest/furthest geospatial matches
geospatial_matches_df = pd.read_pickle("geospatial_matches.pkl")

closest_match = geospatial_matches_df.loc[geospatial_matches_df['distance_meters'].idxmin()]
furthest_match = geospatial_matches_df.loc[geospatial_matches_df['distance_meters'].idxmax()]

def plot_match(match, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(match['ccs_lon'], match['ccs_lat'], color='blue', label='CCS Location')
    plt.scatter(match['gis_lon'], match['gis_lat'], color='red', label='GIS Location')
    plt.plot([match['ccs_lon'], match['gis_lon']], [match['ccs_lat'], match['gis_lat']], color='gray', linestyle='--')

    plt.text(match['ccs_lon'], match['ccs_lat'], f"CCS\n({match['ccs_lat']:.5f}, {match['ccs_lon']:.5f})", fontsize=9, ha='right')
    plt.text(match['gis_lon'], match['gis_lat'], f"GIS\n({match['gis_lat']:.5f}, {match['gis_lon']:.5f})", fontsize=9, ha='left')

    plt.title(f"{title} Match\nDistance: {match['distance_meters']:.2f} meters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_match(closest_match, "Closest")
plot_match(furthest_match, "Furthest")
