import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_COLS = [f"Theoretical_MW_{m}" for m in MONTHS]

def plot_spatial_potential(df):
    plt.figure(figsize=(10, 6))
    valid_df = df.dropna(subset=['Longitude', 'Latitude'])
    plt.scatter(valid_df['Longitude'], valid_df['Latitude'],
                c=valid_df.get('Actual extractable water volume per year (Qreal=ΣQex_m) \n(m3/s)', valid_df.iloc[:, 0]),
                cmap='plasma', alpha=0.6)
    plt.colorbar(label='Annual Extractable Volume')
    plt.title('Geographical Distribution of Renewable Potential')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('spatial_potential.png')
    plt.show()
    plt.close()

def plot_monthly_variation(df, num_rivers=5):
    plt.figure(figsize=(12, 6))
    subset_df = df.dropna(subset=MONTH_COLS).head(num_rivers)
    for i, row in subset_df.iterrows():
        plt.plot(MONTHS, row[MONTH_COLS].values, marker='o',
                 label=f"River: {row.get('River name', f'ID {i}')}")
    plt.title(f"Monthly Potential Variation for First {num_rivers} Rivers")
    plt.xlabel("Month")
    plt.ylabel("Theoretical Potential (MW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_potential_plot.png')
    plt.show()
    plt.close()

def plot_potential_heatmap(df, num_rivers=20):
    plt.figure(figsize=(14, 8))
    subset = df.dropna(subset=MONTH_COLS).iloc[:num_rivers]
    heatmap_data = subset[MONTH_COLS]
    sns.heatmap(heatmap_data, annot=False, cmap='viridis', xticklabels=MONTHS,
                yticklabels=subset.get('River name', subset.index))
    plt.title(f"Heatmap of Monthly Potentials (Top {num_rivers} Rivers)")
    plt.tight_layout()
    plt.savefig('monthly_potential_heatmap.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv('ARA24_Clean_Master_Enhanced.csv')
    df.columns = df.columns.str.strip()

    plot_spatial_potential(df)
    plot_monthly_variation(df)
    plot_potential_heatmap(df)
    print("✅ Visualizations generated successfully.")