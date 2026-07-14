import os
import pandas as pd

EXCEL_FILE = "SGE_Global_Database_ARA24.xlsx"


def build_clean_dataset(file_path):
    if not os.path.exists(file_path):
        print("❌ Database file not found.")
        return

    print("⚡ Cleaning and aggregating ARA24 sheets...")

    # 1. Load tabs from row 0 to inspect real column names dynamically
    try:
        rivers_raw = pd.read_excel(file_path, sheet_name="Rivers", header=0)
        potential_raw = pd.read_excel(file_path, sheet_name="Extractable SGE potential", header=0)
        density_raw = pd.read_excel(file_path, sheet_name="SGE density", header=0)
    except Exception as e:
        print(f"❌ Failed to load sheets from Excel. Error: {e}")
        return

    # Clean up column spaces or linebreaks that Excel sometimes injects
    rivers_raw.columns = [str(c).strip() for c in rivers_raw.columns]
    potential_raw.columns = [str(c).strip() for c in potential_raw.columns]
    density_raw.columns = [str(c).strip() for c in density_raw.columns]

    # Find the exact column name for River ID (handles subtle naming variants)
    id_col = [c for c in rivers_raw.columns if "River ID" in c or "ID" in c][0]
    name_col = [c for c in rivers_raw.columns if "River name" in c or "name" in c][0]

    # 2. Drop rows where River ID is NaN (this instantly wipes out that empty top row!)
    rivers_df = rivers_raw.dropna(subset=[id_col]).copy()
    potential_df = potential_raw.dropna(subset=[potential_raw.columns[0]]).copy()
    density_df = density_raw.dropna(subset=[density_raw.columns[0]]).copy()

    # Standardize the ID column type to integer strings so they match perfectly
    rivers_df[id_col] = rivers_df[id_col].astype(float).astype(int).astype(str)

    # Extract the first column from other sheets to use as the matching key
    pot_id = potential_df.columns[0]
    den_id = density_df.columns[0]
    potential_df[pot_id] = potential_df[pot_id].astype(float).astype(int).astype(str)
    density_df[den_id] = density_df[den_id].astype(float).astype(int).astype(str)

    # 3. Dynamic Monthly Metrics Extraction
    # Look for columns matching months to bypass structural subheaders
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    actual_pot_months = [c for c in potential_df.columns if any(m in str(c) for m in months)]
    actual_den_months = [c for c in density_df.columns if any(m in str(c) for m in months)]

    # Filter down subsets to just the ID and their respective monthly data
    pot_subset = potential_df[[pot_id] + actual_pot_months].copy()
    den_subset = density_df[[den_id] + actual_den_months].copy()

    # Clean up column headers for the master sheet
    pot_subset.columns = [id_col] + [f"{m.split()[0]}_Potential_MW" for m in months[:len(actual_pot_months)]]
    den_subset.columns = [id_col] + [f"{m.split()[0]}_Density_MJ_m3" for m in months[:len(actual_den_months)]]

    # 4. Compile the Data Matrix
    master_df = pd.merge(rivers_df, pot_subset, on=id_col, how="inner")
    master_df = pd.merge(master_df, den_subset, on=id_col, how="inner")

    # Drop any leftover completely nameless columns
    master_df = master_df.loc[:, ~master_df.columns.str.contains('^Unnamed')]

    print(f"✅ Aggregation Complete! Formatted {master_df.shape[0]} unique rivers.")

    # Output path
    output_path = os.path.join(os.path.dirname(file_path), "ARA24_Clean_Master.csv")
    master_df.to_csv(output_path, index=False)
    print(f"💾 Saved clean master dataset to: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(base_dir, EXCEL_FILE)
    build_clean_dataset(full_excel_path)