import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ==========================================
# Configuration & Constants
# ==========================================

# 1. File Paths
DATA_PATHS = [
    "dataset/sample_nh.csv",
    "dataset/sample_us.csv",
    "dataset/sample_yc.csv",
    "dataset/sample_nj.csv"
]

# 2. Date Range (Target Year: 2021)
DATE_START = "20210701"
DATE_END = "20211231"

# 3. Filtering Thresholds
FILTER_MIN_AREA = 100       # Minimum area threshold (keep rows > this value)
FILTER_MIN_COUNT = 200        # Minimum sample count per Crop ID to retain the class
                            # NOTE: Set to 0 or a low value (e.g., 10) for sample datasets.
                            #       Increase to 200+ for the full dataset to remove sparse classes.

# 4. Augmentation & Splitting Config
TARGET_COUNT = 2000         # Target sample count per class for balancing (oversampling)
AUGMENT_MASK_RATE = 0.05    # Probability of masking features during augmentation (simulating clouds/missing data)
SPLIT_TEST_SIZE = 0.4       # Proportion of the dataset to include in the test split (Valid + Test)
SPLIT_VALID_SIZE = 0.5      # Proportion of the test split to include in the validation set (resulting in equal size for Valid and Test)

# 5. Mappings
CROP_MAPPING = {
    27: "Sesame", 2: "Pepper", 8: "Aralia", 1: "Sweet potato",
    17: "Sudangrass", 29: "Soybean", 9: "Perilla", 19: "Greenhouse",
    24: "Yuzu", 23: "Maize", 28: "Kiwi", 22: "Onion",
    16: "Apple", 30: "Grape", 14: "Peach", 10: "Garlic",
    12: "Pear", 13: "Cabbage", 11: "Sapling", 31: "Radish"
}

# ==========================================
# Helper Functions
# ==========================================

def zero_as_nan(df):
    """
    Replaces 0 values with NaN in feature columns (Vectorized operation).
    Feature columns are identified by the pattern 'b{band}_{date}'.
    """
    feat_cols = [c for c in df.columns if re.match(r"b[\da-zA-Z]+_\d{8}", c)]
    if not feat_cols:
        return df
    
    df_copy = df.copy()
    vals = df_copy[feat_cols].values
    vals[vals == 0] = np.nan
    df_copy[feat_cols] = vals
    return df_copy

def shift_column_years(df, years=-1):
    """
    Shifts the year in the column names by the specified amount.
    Used to align data from different years (e.g., aligning 2022 data to 2021).
    """
    new_columns = []
    for col in df.columns:
        match = re.match(r"(b[\da-zA-Z]+)_(\d{8})", col)
        if match:
            band, date_str = match.groups()
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            try:
                shifted_date = date_obj.replace(year=date_obj.year + years)
            except ValueError:
                # Handle leap years (Feb 29 -> Feb 28)
                shifted_date = date_obj.replace(year=date_obj.year + years, day=28)
            
            new_col = f"{band}_{shifted_date.strftime('%Y%m%d')}"
            new_columns.append(new_col)
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df

def filter_columns_by_date(df, start, end):
    """
    Filters feature columns based on a date range and retains specific metadata columns.
    Excluded: region, geometry, CR_NM
    """
    band_date_cols = [col for col in df.columns if re.match(r"b[\da-zA-Z]+_\d{8}", col)]
    selected_cols = [col for col in band_date_cols if start <= col.split('_')[1] <= end]
    
    # Define metadata columns to keep (Excluding 'geometry', 'CR_NM', 'region')
    meta_cols = ['INTPR_NM', 'AREA', 'CR_ID'] 
    
    keep_cols = [c for c in meta_cols if c in df.columns] + selected_cols
    return df[keep_cols].copy()

def load_and_filter(path, shift=False):
    """
    Loads a CSV file and applies filtering based on centralized thresholds.
    """
    print(f"Processing file: {path}")

    try:
        df = pd.read_csv(path, dtype={'CR_ID': 'Int64'}, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype={'CR_ID': 'Int64'}, encoding='cp949')
    
    # Pre-processing: Convert 0 to NaN
    df = zero_as_nan(df)

    # Shift Logic
    if shift:
        print(f"  -> Shifting years by -1 for {path} (2022 -> 2021)")
        df = shift_column_years(df, years=-1)
        
    df = filter_columns_by_date(df, start=DATE_START, end=DATE_END)
    df["AREA"] = pd.to_numeric(df["AREA"], errors="coerce")
    
    # [Config applied] Filter by Area
    df.dropna(subset=['AREA'], inplace=True)
    df = df[df["AREA"] > FILTER_MIN_AREA]

    # [Config applied] Filter by Sample Count per ID
    # This ensures small classes are removed if they don't meet the threshold
    if FILTER_MIN_COUNT > 0:
        df = df[df.groupby("CR_ID")["CR_ID"].transform("count") > FILTER_MIN_COUNT]
    
    # Remove rows where all feature columns are NaN
    feat_cols = [c for c in df.columns if re.match(r"b[\da-zA-Z]+_\d{8}", c)]
    if feat_cols:
        df = df.dropna(subset=feat_cols, how='all')

    print(f"  -> Rows retained: {len(df)}")
    return df

def augment_with_missing_features_fast(df_subset, num_augmented):
    """
    Augments data by sampling existing rows and randomly masking features to simulate missing data.
    """
    if num_augmented <= 0:
        return pd.DataFrame()

    feature_cols = [col for col in df_subset.columns if re.match(r"b[\da-zA-Z]+_\d{8}", col)]
    
    augmented_df = df_subset.sample(n=num_augmented, replace=True).copy()
    
    # [Config applied] Random Masking
    mask = np.random.rand(len(augmented_df), len(feature_cols)) < AUGMENT_MASK_RATE
    values = augmented_df[feature_cols].values
    values[mask] = np.nan
    augmented_df[feature_cols] = values
    
    return augmented_df

def interpolate_and_group_with_fill(df_subset):
    """
    Groups columns by time intervals (early/mid/late month) and performs linear interpolation
    to handle missing values in the time series.
    """
    df_subset = df_subset.reset_index(drop=True)
    
    def group_band_columns(df):
        band_date_cols = [col for col in df.columns if re.match(r"b[\da-zA-Z]+_\d{8}", col)]
        band_date_info = []
        for col in band_date_cols:
            match = re.match(r"(b[\da-zA-Z]+)_(\d{8})", col)
            if match:
                band, date_str = match.groups()
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                # Determine interval: 1 (1st-10th), 2 (11th-20th), 3 (21st-End)
                interval = 1 if date_obj.day <= 10 else 2 if date_obj.day <= 20 else 3
                key = f"{band}_{date_obj.year}{date_obj.month:02d}_{interval}"
                band_date_info.append((col, key))
        
        grouped_cols = defaultdict(list)
        for original_col, grouped_key in band_date_info:
            grouped_cols[grouped_key].append(original_col)
        return grouped_cols

    meta_cols = ['INTPR_NM', 'AREA', 'CR_ID', 'crop_name']
    meta_df = df_subset[[c for c in meta_cols if c in df_subset.columns]].copy()
    
    grouped_cols = group_band_columns(df_subset)
    bands = sorted({key.split("_")[0] for key in grouped_cols.keys()})
    
    interpolated_dfs = []

    for band in bands:
        keys = sorted(
            [key for key in grouped_cols.keys() if key.startswith(band)],
            key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2]))
        )
        band_df = pd.DataFrame(index=df_subset.index)
        
        # Aggregate columns within the same interval
        for key in keys:
            cols = grouped_cols[key]
            band_df[key] = df_subset[cols].mean(axis=1, skipna=True) if cols else np.nan

        # Linear Interpolation across time steps
        band_df_interp = band_df.transpose().interpolate(method='linear', axis=0, limit_direction='both')\
                                            .fillna(method='ffill').fillna(method='bfill').transpose()
        interpolated_dfs.append(band_df_interp)
    
    if not interpolated_dfs:
        return meta_df

    averaged_df = pd.concat(interpolated_dfs, axis=1)
    final_df = pd.concat([meta_df, averaged_df], axis=1)
    
    # Drop rows where interpolation failed completely
    final_df = final_df.dropna()
    
    return final_df

# ==========================================
# Main Execution Flow
# ==========================================

def main():
    print("--- 1. Loading and Filtering Data ---")
    print(f"Config: MIN_AREA={FILTER_MIN_AREA}, MIN_COUNT={FILTER_MIN_COUNT}")
    
    dfs = []
    for path in DATA_PATHS:
        # Check if filename contains 'sample_nh' to apply year shift (e.g., 2022 to 2021)
        should_shift = "sample_nh.csv" in path
        dfs.append(load_and_filter(path, shift=should_shift))
        
    full_df = pd.concat(dfs, ignore_index=True)

    if len(full_df) == 0:
        print("[Error] No data loaded. Check file paths or date ranges.")
        return

    # Apply Crop Name Mapping
    full_df["crop_name"] = full_df["CR_ID"].map(CROP_MAPPING)
    full_df.dropna(subset=["crop_name"], inplace=True)
    print(f"Total data after filtering: {len(full_df)} rows")

    print(f"\n--- 2. Splitting Data (Test Size={SPLIT_TEST_SIZE}, Valid Size={SPLIT_VALID_SIZE}) ---")
    # [Config applied] Stratified split based on crop_name
    train_df, temp_df = train_test_split(
        full_df, 
        test_size=SPLIT_TEST_SIZE, 
        stratify=full_df['crop_name'], 
        random_state=42
    )
    
    valid_df, test_df = train_test_split(
        temp_df, 
        test_size=SPLIT_VALID_SIZE, 
        stratify=temp_df['crop_name'], 
        random_state=42
    )

    # Reset Index
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"\n--- 3. Augmenting Training Data (Target={TARGET_COUNT}) ---")
    augmented_train_dfs = []
    for cname, group in train_df.groupby('crop_name'):
        n = len(group)
        if n < TARGET_COUNT:
            # Oversampling needed
            need = TARGET_COUNT - n
            augmented = augment_with_missing_features_fast(group, need)
            augmented_train_dfs.append(pd.concat([group, augmented], axis=0))
        else:
            # Downsampling
            sampled = group.sample(n=TARGET_COUNT, random_state=42)
            augmented_train_dfs.append(sampled)

    train_df = pd.concat(augmented_train_dfs).reset_index(drop=True)
    print(f"Total training data after augmentation: {len(train_df)} rows")

    print("\n--- 4. Interpolating Time Series Data ---")
    train_interp = interpolate_and_group_with_fill(train_df)
    valid_interp = interpolate_and_group_with_fill(valid_df)
    test_interp  = interpolate_and_group_with_fill(test_df)

    print("Interpolation complete.")
    print(f"Final processed train data shape: {train_interp.shape}")

    # Saving Results
    os.makedirs("dataset", exist_ok=True)
    train_interp.to_csv("dataset/train_interpolated.csv", index=False, encoding='utf-8-sig')
    valid_interp.to_csv("dataset/valid_interpolated.csv", index=False, encoding='utf-8-sig')
    test_interp.to_csv("dataset/test_interpolated.csv", index=False, encoding='utf-8-sig')
    print("\nAll files saved successfully.")

if __name__ == "__main__":
    main()