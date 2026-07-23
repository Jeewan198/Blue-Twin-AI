import os
import pandas as pd

EXCEL_FILE = "SGE_Global_Database_ARA24.xlsx"
OUTPUT_FILE = "ARA24_Clean_Master_Enhanced.csv"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


def extract_monthly_sheet(file_path, sheet_name, prefix):
    """
    Reads a sheet with a two-row Excel header (merged label + month sub-header)
    and returns a dataframe with columns: [River ID, <prefix>_January, ..., <prefix>_December].
    Works for sheets that contain exactly ONE monthly block (Extractable SGE potential,
    SGE density, Theoretical SGE potential).
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
    raw.columns = [str(c).strip() for c in raw.columns]

    id_col = raw.columns[0]  # River ID sits in position 0 with no sub-header, so it becomes 'Unnamed: 0'
    month_cols = [c for c in raw.columns if any(m in c for m in MONTHS)]

    if len(month_cols) != 12:
        raise ValueError(
            f"Expected 12 monthly columns in '{sheet_name}', found {len(month_cols)}. "
            f"Sheet structure may have changed — check manually before proceeding."
        )

    out = raw[[id_col] + month_cols].copy()
    out.columns = ["River ID"] + [f"{prefix}_{m}" for m in MONTHS]
    out = out.dropna(subset=["River ID"]).copy()
    out["River ID"] = out["River ID"].astype(float).astype(int).astype(str)
    return out


def extract_rivers_sheet(file_path):
    """
    Reads the Rivers sheet twice: once (header=0) for the named metadata/annual-scalar
    columns, once (header=1) for the two monthly blocks (Qm and Qex_m), which lose their
    names under header=0 and would otherwise be silently dropped as 'Unnamed' columns.
    """
    named = pd.read_excel(file_path, sheet_name="Rivers", header=0)
    named.columns = [str(c).strip() for c in named.columns]

    id_col = [c for c in named.columns if "River ID" in c][0]
    name_col = [c for c in named.columns if "River name" in c][0]
    scalar_labels = {
        "Qa": "interannual mean discharge",
        "Qe": "enviromental discharge",
        "EF": "Extraction Factor",
        "Qd": "Design flow",
        "Qreal": "Actual extractable water volume",
        "CF": "Capacity factor",
    }
    scalar_cols = {key: [c for c in named.columns if label in c][0] for key, label in scalar_labels.items()}

    metadata = named[[id_col, name_col, "Latitude", "Longitude", "Country", "Region",
                       "Continent", "Ocean Basin"] + list(scalar_cols.values())].copy()
    metadata = metadata.dropna(subset=[id_col]).reset_index(drop=True)
    metadata[id_col] = metadata[id_col].astype(float).astype(int).astype(str)
    metadata = metadata.rename(columns={id_col: "River ID", name_col: "River name"})

    # Second read: header=1 exposes the two monthly blocks (Qm, then Qex_m) as
    # literal month names ('January'...'December', 'January.1'...'December.1').
    monthly_raw = pd.read_excel(file_path, sheet_name="Rivers", header=1)
    monthly_raw.columns = [str(c).strip() for c in monthly_raw.columns]
    month_matched = [c for c in monthly_raw.columns if any(m in c for m in MONTHS)]

    if len(month_matched) != 24:
        raise ValueError(
            f"Expected 24 monthly columns (Qm x12 + Qex_m x12) in 'Rivers', found "
            f"{len(month_matched)}. Sheet structure may have changed — check manually."
        )

    qm_cols, qex_cols = month_matched[:12], month_matched[12:24]
    qm_df = monthly_raw[qm_cols].copy()
    qm_df.columns = [f"Qm_{m}" for m in MONTHS]
    qex_df = monthly_raw[qex_cols].copy()
    qex_df.columns = [f"Qex_{m}" for m in MONTHS]

    # The header=0 read includes one leading blank sub-header row (dropped above via
    # dropna on id_col); the header=1 read never had that row, since it was consumed
    # as the header itself. Both therefore already align 1:1 in the same row order
    # once reset — no further offset/reindexing is needed.
    if len(metadata) != len(qm_df) or len(metadata) != len(qex_df):
        raise ValueError(
            f"Row count mismatch after alignment: metadata={len(metadata)}, "
            f"Qm={len(qm_df)}, Qex_m={len(qex_df)}. Check sheet structure manually."
        )

    combined = pd.concat([metadata.reset_index(drop=True),
                           qm_df.reset_index(drop=True),
                           qex_df.reset_index(drop=True)], axis=1)
    return combined


def build_clean_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Database file not found: {file_path}")
        return

    print("⚡ Building ARA24 master dataset from scratch (single consolidated script)...")

    rivers = extract_rivers_sheet(file_path)
    print(f"  Rivers sheet: {len(rivers)} rivers, with Qm & Qex_m monthly data intact.")

    extractable = extract_monthly_sheet(file_path, "Extractable SGE potential", "Extractable_MW")
    density = extract_monthly_sheet(file_path, "SGE density", "Density_MJ_m3")
    theoretical = extract_monthly_sheet(file_path, "Theoretical SGE potential", "Theoretical_MW")

    master = rivers.copy()
    for name, df in [("Extractable SGE potential", extractable),
                      ("SGE density", density),
                      ("Theoretical SGE potential", theoretical)]:
        before = len(master)
        master = pd.merge(master, df, on="River ID", how="left")
        missing = master[df.columns[1]].isna().sum()
        print(f"  Merged {name}: {before} -> {len(master)} rows, {missing} rivers unmatched "
              f"({'OK' if missing == 0 else 'CHECK THIS'})")

    print(f"✅ Final dataset: {master.shape[0]} rivers, {master.shape[1]} columns.")

    output_path = os.path.join(os.path.dirname(file_path) or ".", OUTPUT_FILE)
    master.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_excel_path = os.path.join(base_dir, EXCEL_FILE)
    build_clean_dataset(full_excel_path)
