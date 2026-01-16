import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

# =================================
# 1. Data Loading and Preprocessing
# =================================
csv_path = r"dataset\processed_all_interpolated.csv"
df = pd.read_csv(csv_path)

# 1-1) Remove missing values in 'CR_NM'
df = df.dropna(subset=['CR_NM']).copy()

# 1-2) Map crop IDs to names
mapping_crop = {
    27: "Sesame", 2: "Pepper", 8: "Aralia", 1: "Sweet potato",
    17: "Sudangrass", 29: "Soybean", 9: "Perilla", 19: "Greenhouse", 24: "Yuzu",
    23: "Maize", 28: "Kiwi", 22: "Onion", 16: "Apple",
    30: "Grape", 14: "Peach", 10: "Garlic", 12: "Pear", 13: "Cabbage",
    11: "Sapling", 31: "Radish"
}

df["crop_name"] = df["CR_ID"].map(mapping_crop)
df.dropna(subset=["crop_name"], inplace=True)

# =================================
# 2. Filter Classes by Sample Count
# =================================
counts = df['crop_name'].value_counts()
# Keep classes with more than 200 samples
valid_classes = counts[counts > 200].index
df = df[df['crop_name'].isin(valid_classes)].copy()

# =================================
# 3. Create Feature List (Band-Month-Segment)
# =================================
months = [f"2021{m:02d}" for m in range(7, 13)]
bands  = ['b02','b03','b04','b05','b06','b07','b08','b8a','b11','b12']

features = [f"{b}_{mon}_{d}"
            for b in bands
            for mon in months
            for d in range(1, 4)]

# Keep only features that exist in the dataframe columns
features = [f for f in features if f in df.columns]

# Construct X (features) and y (labels) without missing values
X = df[features].dropna()
y = df.loc[X.index, 'crop_name']

# Check sample counts
class_counts = y.value_counts().sort_values(ascending=True)
print("■ Sample counts by Crop Name (Ascending):")
print(class_counts)

# ============================================
# 4. Consensus Clustering (Undersampling + K-Means++)
#    - Alignment metric: Crop Distribution Similarity (JS Distance)
# ============================================

# 4-1) Scaling (Fit on the entire dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index)

# --- Function to align labels based on distribution similarity ---
def align_by_labeldist(ref_labels: np.ndarray,
                       cur_labels: np.ndarray,
                       y_labels: pd.Series,
                       k: int,
                       eps: float = 1e-12) -> np.ndarray:
    """
    Aligns current labels to reference labels using the Hungarian algorithm,
    minimizing the Jensen-Shannon distance between crop distributions within clusters.
    """
    crops = np.sort(y_labels.unique())

    def get_dist_matrix(labels: np.ndarray) -> np.ndarray:
        # Create (k x C) matrix: Crop distribution per cluster
        mat = np.zeros((k, len(crops)), dtype=float)
        for c in range(k):
            vc = y_labels[labels == c].value_counts().reindex(crops, fill_value=0).values.astype(float)
            s = vc.sum()
            mat[c] = (vc / s) if s > 0 else vc
        
        # Add epsilon and normalize rows
        mat = mat + eps
        mat = (mat.T / mat.sum(axis=1)).T
        return mat

    A = get_dist_matrix(ref_labels)
    B = get_dist_matrix(cur_labels)

    # Cost matrix: Jensen-Shannon distance
    C_mat = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            C_mat[i, j] = jensenshannon(A[i], B[j])

    # Hungarian algorithm (Linear Sum Assignment)
    ri, cj = linear_sum_assignment(C_mat)
    mapping = {cj[m]: ri[m] for m in range(len(ri))}
    
    return np.vectorize(mapping.get)(cur_labels)

# 4-2) Clustering Configuration
n_min = int(class_counts.min())            # Smallest class size for balancing
labels_sorted = sorted(class_counts.index) # Sorted labels for consistent sampling
k_opt = 11                                 # Number of clusters
R = 20                                     # Number of repetitions
seeds = [10 + r for r in range(R)]         # Reproducible seeds

# 4-3) Iterative Clustering
N = len(X_scaled_df)
labels_runs = np.zeros((R, N), dtype=np.int32)

ref_labels = None

for r, seed in enumerate(seeds):
    # (a) Undersampling: Select n_min samples per class
    idx_balanced_list = [
        y[y == cls].sample(n=n_min, random_state=seed).index
        for cls in labels_sorted
    ]
    idx_balanced = np.concatenate([idx.to_numpy() for idx in idx_balanced_list])

    # (b) Fit K-Means on balanced subset
    km = KMeans(
        n_clusters=k_opt,
        init='k-means++',
        n_init=10, # Optimized for speed (default is 10 in newer sklearn)
        max_iter=300,
        algorithm='elkan',
        random_state=seed
    ).fit(X_scaled_df.loc[idx_balanced].values)

    # (c) Predict on full dataset
    labels_all = km.predict(X_scaled_df.values)

    # (d) Align labels across runs
    if r == 0:
        ref_labels = labels_all.copy()
        labels_runs[r] = ref_labels
    else:
        mapped = align_by_labeldist(ref_labels, labels_all, y, k_opt)
        labels_runs[r] = mapped

# 4-4) Compute Consensus (Majority Vote)
# Calculate votes for each sample (Transpose to get [N, k])
votes = np.apply_along_axis(
    lambda col: np.bincount(col, minlength=k_opt),
    axis=0, arr=labels_runs
).T 

consensus_labels = votes.argmax(axis=1).astype(np.int32)
agreement = (votes.max(axis=1) / R).astype(np.float32)

clusters = pd.Series(consensus_labels, index=X.index, name="cluster")
agree_series = pd.Series(agreement, index=X.index, name="agreement")

print(f"\nConsensus Complete: k={k_opt}, Repetitions={R}")
print(f"Mean Agreement={agree_series.mean():.3f}, Median Agreement={agree_series.median():.3f}")

# =================================
# 5. Visualization and Analysis
# =================================

# 5-1) Prepare Crosstab (Column Normalized %)
ct_cnt = pd.crosstab(clusters, y)
ct_pct = ct_cnt.div(ct_cnt.sum(axis=0), axis=1) * 100.0  # Normalize columns to 100%

# Sort clusters by total size (or purity) for better visualization
# Here: Sorting by total count descending
row_order = list(ct_cnt.sum(axis=1).sort_values(ascending=False).index)
col_order = sorted(ct_pct.columns) # Sort crops alphabetically

# Reorder Dataframe
ct_pct_re = ct_pct.loc[row_order, col_order].copy()

# Rename clusters for display (Cluster 1 to k)
new_labels = {old_id: i+1 for i, old_id in enumerate(row_order)}
ct_pct_re.index = [f"Cluster {new_labels[oid]} (ID:{oid})" for oid in row_order]

print("\n🧮 Cluster Distribution Summary (Top Clusters):")
print(ct_pct_re.head())

# 5-2) Plot Heatmap
M = ct_pct_re.values
rows_lbl = list(ct_pct_re.index)
cols_lbl = list(ct_pct_re.columns)

plt.figure(figsize=(max(10, len(cols_lbl)*0.6), max(6, len(rows_lbl)*0.5)))
# Changed cmap to 'Blues' (Standard: Darker = Higher density)
im = plt.imshow(M, aspect='auto', interpolation='nearest', cmap='Blues')
plt.colorbar(im, fraction=0.02, pad=0.02, label='Column-normalized (%)')

plt.xticks(ticks=np.arange(len(cols_lbl)), labels=cols_lbl, rotation=60, ha='right')
plt.yticks(ticks=np.arange(len(rows_lbl)), labels=rows_lbl)

plt.title('Crop Distribution Across Consensus Clusters')
plt.xlabel('Crop Class (Column sums to 100%)')
plt.ylabel('Cluster ID (Sorted by Size)')

# Annotate cells with values >= 5%
for i in range(len(rows_lbl)):
    for j in range(len(cols_lbl)):
        v = M[i, j]
        if v >= 5: # Threshold for text visibility
            # Determine text color based on background intensity
            text_color = "white" if v > 50 else "black"
            plt.text(j, i, f'{v:.0f}%', va='center', ha='center', 
                     fontsize=9, color=text_color)

plt.tight_layout()
plt.show()