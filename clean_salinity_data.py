import pandas as pd


def clean_salinity_data(original_csv, excel_file, output_file):
    # 1. Load your original rich dataset
    df_original = pd.read_csv(original_csv)

    # 2. Load the SGE Theoretical Potential sheet
    df_potential = pd.read_excel(excel_file, sheet_name='Theoretical SGE potential', header=1)

    # Select only River ID and the 12 months (indices 4 to 15)
    # Using 'River ID' to merge ensures we align potential with the correct river
    potentials = df_potential.iloc[:, [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    months = ["Jan_Pot", "Feb_Pot", "Mar_Pot", "Apr_Pot", "May_Pot", "Jun_Pot",
              "Jul_Pot", "Aug_Pot", "Sep_Pot", "Oct_Pot", "Nov_Pot", "Dec_Pot"]
    potentials.columns = ['River ID'] + months

    # 3. Merge based on River ID
    df_merged = pd.merge(df_original, potentials, on='River ID', how='left')

    # 4. Save the new master file
    df_merged.to_csv(output_file, index=False)
    print(f"✅ Created {output_file} with {df_merged.shape[1]} columns.")


# Execute
clean_salinity_data('ARA24_Clean_Master.csv', 'SGE_Global_Database_ARA24.xlsx', 'ARA24_Clean_Master_Enhanced.csv')