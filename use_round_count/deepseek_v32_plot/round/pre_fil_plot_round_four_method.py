import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. Data Preparation & Parsing
# ==========================================
data_str = """
Method baseline filter_think filter_keep_last_think fiter_keep_last_tool long_context_filter_tool_p75 long_context_filter_tool_p100
acc 42.72 47.57 48.54 39.81 46.6 44.66
round 561 710 539 536 568 538
acc 45.63 44.66 45.63 41.75 41.75 47.57
round 559 681 547 538 536 545
acc 43.69 49.51 46.6 36.89 41.75 41.75
round 575 731 538 571 579 574
acc 44.66 46.6 43.69 36.89 43.69 46.6
round 601 868 676 761 651 649
acc 49.51 43.69 44.66 38.83 47.57 47.57
round 606 904 716 815 631 615
acc 45.63 45.63 51.46 36.89 42.72 45.63
round 606 918 621 777 686 645
acc 44.66 49.51 49.51 32.04 52.43 43.69
round 674 875 694 1088 726 732
acc 43.69 52.43 53.4 35.92 50.49 54.37
round 682 1010 633 1214 653 654
acc 42.72 52.43 45.63 36.89 47.57 49.51
round 730 940 739 923 756 749
acc 42.72 51.46 49.51 41.75 48.54 43.69
round 659 947 708 1542 923 823
acc 45.63 49.51 44.66 36.89 45.63 40.78
round 589 911 732 1696 929 859
acc 42.72 48.54 51.46 40.78 50.49 44.66
round 699 1072 711 1483 792 720
"""

lines = data_str.strip().split('\n')
header = lines[0].split()
methods = header[1:]
data_list = []

for i in range(1, len(lines), 2):
    acc_line = lines[i].split()
    round_line = lines[i+1].split()
    if acc_line[0] == 'acc' and round_line[0] == 'round':
        acc_values = [float(x) for x in acc_line[1:]]
        round_values = [float(x) for x in round_line[1:]]
        for method, acc, rnd in zip(methods, acc_values, round_values):
            data_list.append({'Method': method, 'Accuracy': acc, 'Round': rnd})

df = pd.DataFrame(data_list)

# ==========================================
# 2. Filtering (New Logic)
# ==========================================

# 2.1 Remove methods starting with 'long_context'
df = df[~df['Method'].str.startswith('long_context')]

# 2.2 Filter rounds <= 1000
df = df[df['Round'] <= 1000]

# ==========================================
# 3. Data Preprocessing: Calculate Trends
# ==========================================
trend_dfs = []

# Re-calculate trends based on the filtered data
for method in df['Method'].unique():
    sub_df = df[df['Method'] == method].copy()
    
    # Skip if not enough data points for binning
    if len(sub_df) < 2:
        continue
        
    try:
        # Try 3 bins, fallback to fewer if data is sparse
        n_bins = min(3, len(sub_df))
        sub_df['Bin'] = pd.qcut(sub_df['Round'], q=n_bins, duplicates='drop')
    except:
        sub_df['Bin'] = pd.cut(sub_df['Round'], bins=n_bins)
    
    agg_df = sub_df.groupby('Bin', observed=True)[['Round', 'Accuracy']].mean().reset_index()
    agg_df['Method'] = method
    trend_dfs.append(agg_df)

df_trend = pd.concat(trend_dfs)

# ==========================================
# 4. Plotting
# ==========================================
sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
plt.figure(figsize=(10, 7))

# Get unique methods remaining after filter for color consistency
remaining_methods = df['Method'].unique()
palette = sns.color_palette("bright", n_colors=len(remaining_methods))

# A. Background Scatter (Raw Data)
sns.scatterplot(
    data=df, 
    x='Round', 
    y='Accuracy', 
    hue='Method', 
    palette=palette,
    hue_order=remaining_methods,
    alpha=0.3, 
    s=60, 
    legend=False
)

# B. Trend Lines (Aggregated Data)
sns.lineplot(
    data=df_trend, 
    x='Round', 
    y='Accuracy', 
    hue='Method', 
    style='Method',
    hue_order=remaining_methods,
    style_order=remaining_methods,
    palette=palette,
    linewidth=3,
    markers=True,
    markersize=10,
    dashes=False
)

# ==========================================
# 5. Styling & Saving
# ==========================================
plt.title('Scaling Law (Filtered: Round <= 1000)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Round (Cost)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')

# Set X-axis limit explicitly as requested
plt.xlim(right=1050) # Slightly over 1000 to give breathing room for markers

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Method')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig('filtered_comparison_plot.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()