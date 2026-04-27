import pandas as pd

def transform_to_tidy_data(input_file, output_file, sheet_name="2024"):
    try:
        raw_df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)

        # Find header row
        header_row_index = None
        for i, row in raw_df.iterrows():
            if any("parameter" in str(v).lower() for v in row.values):
                header_row_index = i
                break

        if header_row_index is None:
            print("❌ Could not find a row containing 'Parameters'.")
            return

        df = pd.read_excel(input_file, sheet_name=sheet_name, header=header_row_index)
        df.columns = [str(c).strip() for c in df.columns]

        id_col  = next((c for c in df.columns if "parameter" in c.lower()), None)
        uom_col = next((c for c in df.columns if "uom" in c.lower()), None)

        if not id_col:
            print("❌ No 'Parameters' column found after header detection.")
            return
        if not uom_col:
            print("⚠️  No 'UOM' column found — adding blank UOM.")
            df["UOM"] = ""
            uom_col = "UOM"

        # Detect month columns dynamically (any MMM'YY pattern)
        import re
        month_pattern = re.compile(r"^[A-Za-z]{3}'?\d{2}$")
        available_months = [c for c in df.columns if month_pattern.match(c)]

        if not available_months:
            print("❌ No month columns detected. Expected format: Jan'24, Feb'24 …")
            return

        print(f"Found {len(available_months)} month columns: {available_months}")

        # Print unique parameter names to help debug PARAMS_MAP in functions.py
        print("\n--- Unique Parameters in your file ---")
        for p in sorted(df[id_col].dropna().unique()):
            print(f"  {p}")
        print("--------------------------------------\n")

        tidy_df = pd.melt(
            df,
            id_vars=[id_col, uom_col],
            value_vars=available_months,
            var_name="Month",
            value_name="Metric_Value",
        )
        tidy_df = tidy_df.rename(columns={id_col: "Parameters", uom_col: "UOM"})
        tidy_df["Metric_Value"] = pd.to_numeric(tidy_df["Metric_Value"], errors="coerce")
        tidy_df = tidy_df.dropna(subset=["Metric_Value"])

        tidy_df.to_excel(output_file, index=False)
        print(f"✅ SUCCESS — {len(tidy_df)} rows written to: {output_file}")
        print(f"   Rows × Columns: {tidy_df.shape}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        raise

transform_to_tidy_data("Steel_Plant_Process_Master.xlsx", "Audit_Ready_Plant_Data.xlsx")