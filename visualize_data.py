import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_spatial_potential(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Longitude'], df['Latitude'],
                c=df['Actual extractable water volume per year (Qreal=ΣQex_m) \n(m3/s)'],
                cmap='plasma', alpha=0.6)
    plt.colorbar(label='Annual Extractable Volume')
    plt.title('Geographical Distribution of Renewable Potential')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('spatial_potential.png')
    plt.show()

def plot_monthly_variation(df, num_rivers=5):
    month_cols = ["Jan_Pot", "Feb_Pot", "Mar_Pot", "Apr_Pot", "May_Pot", "Jun_Pot",
                  "Jul_Pot", "Aug_Pot", "Sep_Pot", "Oct_Pot", "Nov_Pot", "Dec_Pot"]
    plt.figure(figsize=(12, 6))
    for i in range(num_rivers):
        plt.plot(month_cols, df.loc[i, month_cols], marker='o',
                 label=f"River: {df.loc[i, 'River name']}")
    plt.title(f"Monthly Potential Variation for First {num_rivers} Rivers")
    plt.xlabel("Month")
    plt.ylabel("Potential (MW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_potential_plot.png')
    plt.show()

def plot_potential_heatmap(df, num_rivers=20):
    month_cols = ["Jan_Pot", "Feb_Pot", "Mar_Pot", "Apr_Pot", "May_Pot", "Jun_Pot",
                  "Jul_Pot", "Aug_Pot", "Sep_Pot", "Oct_Pot", "Nov_Pot", "Dec_Pot"]
    plt.figure(figsize=(14, 8))
    subset = df.iloc[:num_rivers][month_cols]
    sns.heatmap(subset, annot=False, cmap='viridis', xticklabels=month_cols,
                yticklabels=df.iloc[:num_rivers]['River name'])
    plt.title(f"Heatmap of Monthly Potentials (Top {num_rivers} Rivers)")
    plt.tight_layout()
    plt.savefig('monthly_potential_heatmap.png')
    plt.show()

# Run all visualizations
if __name__ == "__main__":
    df = pd.read_csv('ARA24_Clean_Master_Enhanced.csv')
    df.columns = df.columns.str.strip()

    plot_spatial_potential(df)
    plot_monthly_variation(df)
    plot_potential_heatmap(df)