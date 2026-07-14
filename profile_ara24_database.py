import os
import pandas as pd

# Dataset filename
EXCEL_FILE = "SGE_Global_Database_ARA24.xlsx"

# The exact names of the tabs inside the Excel file
TABS_TO_ANALYZE = [
    "Rivers",
    "SGE density",
    "Theoretical SGE potential",
    "Extractable SGE potential"
]


def profile_excel_tabs(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ Could not find the Excel file at: {file_path}")
        print("Please ensure 'SGE_Global_Database_ARA24.xlsx' is spelled exactly right in your folder.")
        return

    print("============================================================")
    print(f"📖 OPENING EXCEL WORKBOOK: {os.path.basename(file_path)}")
    print("============================================================\n")

    for sheet in TABS_TO_ANALYZE:
        print("-" * 50)
        print(f"📊 PROFILING TAB: {sheet}")
        print("-" * 50)

        try:
            # Tell pandas to read the specific tab name, skipping structural header rows if needed
            # For these sheets, header=[0,1] or header=0 works depending on the tab setup
            df = pd.read_excel(file_path, sheet_name=sheet, header=0)

            print(f"▪ Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            print("▪ Core Columns Found:")
            # Display first 5 columns as a preview
            for col in df.columns[:5]:
                print(f"  - {col}")

            print(f"▪ Total Missing Values: {df.isnull().sum().sum()}")
            print("\n▪ Data Preview (Top 2 Rows):")
            print(df.iloc[:2, :5].to_string(index=False))
            print("\n")

        except Exception as e:
            print(f"❌ Could not read tab '{sheet}'. Error: {e}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(base_dir, EXCEL_FILE)
    profile_excel_tabs(full_excel_path)