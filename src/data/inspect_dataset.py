from pathlib import Path
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/dataset.yaml")


def read_csv_safely(path):
    """
    Try multiple encodings because the IO-VNBD smartphone
    CSV files are not necessarily UTF-8 encoded.
    """

    encodings = [
        "utf-8",
        "cp1252",
        "latin1",
    ]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding

        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def inspect_file(path):

    try:
        df, encoding = read_csv_safely(path)

        return {
            "success": True,
            "encoding": encoding,
            "rows": len(df),
            "columns": list(df.columns),
            "error": None,
        }

    except Exception as e:

        return {
            "success": False,
            "encoding": None,
            "rows": 0,
            "columns": [],
            "error": str(e),
        }


def main():

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    root = Path(config["dataset"]["root"])

    categorized = (
        root /
        config["dataset"]["categorized"]["path"]
    )

    print("=" * 80)
    print("IO-VNBD COMPLETE DATASET INSPECTION")
    print("=" * 80)

    print(f"\nDataset root:")
    print(root.resolve())

    print(f"\nCategorized dataset:")
    print(categorized.resolve())

    if not categorized.exists():
        print("\nERROR: Categorized dataset does not exist.")
        return

    # ---------------------------------------------------------
    # Find all CSVs
    # ---------------------------------------------------------

    csv_files = sorted(categorized.rglob("*.csv"))

    print(f"\nTotal CSV files: {len(csv_files)}")

    # ---------------------------------------------------------
    # Separate Smartphone and Vehicle files
    # ---------------------------------------------------------

    smartphone_files = []
    vehicle_files = []

    for file in csv_files:

        name = file.name.lower()

        if name.startswith("s-"):
            smartphone_files.append(file)

        elif name.startswith("v-"):
            vehicle_files.append(file)

    print(f"Smartphone files: {len(smartphone_files)}")
    print(f"Vehicle files:    {len(vehicle_files)}")

    # ---------------------------------------------------------
    # Inspect Smartphone files
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("SMARTPHONE DATASET")
    print("=" * 80)

    smartphone_success = 0
    smartphone_failed = 0

    smartphone_schema = None

    for file in smartphone_files:

        result = inspect_file(file)

        relative = file.relative_to(categorized)

        if result["success"]:

            smartphone_success += 1

            if smartphone_schema is None:
                smartphone_schema = result["columns"]

            print(
                f"\n[OK] {relative}"
            )

            print(
                f"     Rows: {result['rows']}"
            )

            print(
                f"     Encoding: {result['encoding']}"
            )

        else:

            smartphone_failed += 1

            print(
                f"\n[ERROR] {relative}"
            )

            print(
                f"        {result['error']}"
            )

    print("\nSmartphone summary:")
    print(f"  Successful: {smartphone_success}")
    print(f"  Failed:     {smartphone_failed}")

    # ---------------------------------------------------------
    # Print smartphone schema
    # ---------------------------------------------------------

    if smartphone_schema:

        print("\n" + "-" * 80)
        print("SMARTPHONE COLUMN SCHEMA")
        print("-" * 80)

        for i, column in enumerate(smartphone_schema):

            print(f"{i:02d}. {column}")

    # ---------------------------------------------------------
    # Inspect Vehicle files
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("VEHICLE / REFERENCE DATASET")
    print("=" * 80)

    vehicle_success = 0
    vehicle_failed = 0

    vehicle_schema = None

    for file in vehicle_files:

        result = inspect_file(file)

        relative = file.relative_to(categorized)

        if result["success"]:

            vehicle_success += 1

            if vehicle_schema is None:
                vehicle_schema = result["columns"]

            print(
                f"\n[OK] {relative}"
            )

            print(
                f"     Rows: {result['rows']}"
            )

            print(
                f"     Encoding: {result['encoding']}"
            )

        else:

            vehicle_failed += 1

            print(
                f"\n[ERROR] {relative}"
            )

            print(
                f"        {result['error']}"
            )

    print("\nVehicle summary:")
    print(f"  Successful: {vehicle_success}")
    print(f"  Failed:     {vehicle_failed}")

    # ---------------------------------------------------------
    # Print vehicle schema
    # ---------------------------------------------------------

    if vehicle_schema:

        print("\n" + "-" * 80)
        print("VEHICLE COLUMN SCHEMA")
        print("-" * 80)

        for i, column in enumerate(vehicle_schema):

            print(f"{i:02d}. {column}")

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Total CSVs:       {len(csv_files)}")
    print(f"Smartphone CSVs:  {len(smartphone_files)}")
    print(f"Vehicle CSVs:     {len(vehicle_files)}")

    print(
        f"\nSmartphone readable: "
        f"{smartphone_success}/{len(smartphone_files)}"
    )

    print(
        f"Vehicle readable: "
        f"{vehicle_success}/{len(vehicle_files)}"
    )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()