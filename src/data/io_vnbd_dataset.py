from pathlib import Path
import pandas as pd


class IOVNBDSynchronizedDataset:
    """
    Loader for the synchronized IO-VNBD dataset.

    Expected structure:

    synchronized/
    ├── Categorised IOVNB Dataset/
    │   ├── M (Driver B)/
    │   ├── S (Driver A)/
    │   ├── Vf (Driver E)/
    │   ├── Vta (Driver E)/
    │   ├── Vtb (Driver E)/
    │   ├── Vw (Driver E)/
    │   └── Y (Driver D)/
    │
    └── Uncategorised IOVNB Dataset/
    """

    def __init__(self, dataset_root, use_uncategorised=False):
        self.dataset_root = Path(dataset_root)

        self.categorised_root = (
            self.dataset_root / "Categorised IOVNB Dataset"
        )

        self.uncategorised_root = (
            self.dataset_root / "Uncategorised IOVNB Dataset"
        )

        self.use_uncategorised = use_uncategorised

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_root}"
            )

        if not self.categorised_root.exists():
            raise FileNotFoundError(
                f"Categorised dataset not found: "
                f"{self.categorised_root}"
            )

    @staticmethod
    def read_csv(path):
        """
        Read IO-VNBD CSV while handling the encoding
        used by the smartphone files.
        """

        encodings = [
            "utf-8",
            "cp1252",
            "latin1",
        ]

        last_error = None

        for encoding in encodings:
            try:
                return pd.read_csv(
                    path,
                    encoding=encoding
                )
            except UnicodeDecodeError as error:
                last_error = error

        raise RuntimeError(
            f"Unable to decode CSV: {path}"
        ) from last_error

    def find_sessions(self):
        """
        Find synchronized S/V pairs.

        Example:

        S-S1.csv
        V-S1.csv

        becomes:

        session_id = S1
        """

        smartphone_files = {}
        vehicle_files = {}

        csv_files = self.categorised_root.rglob("*.csv")

        for path in csv_files:

            filename = path.name

            if filename.startswith("S-"):
                session_id = path.stem[2:]
                smartphone_files[session_id] = path

            elif filename.startswith("V-"):
                session_id = path.stem[2:]
                vehicle_files[session_id] = path

        session_ids = sorted(
            set(smartphone_files.keys())
            & set(vehicle_files.keys())
        )

        sessions = []

        for session_id in session_ids:

            smartphone_path = smartphone_files[session_id]
            vehicle_path = vehicle_files[session_id]

            relative = smartphone_path.relative_to(
                self.categorised_root
            )

            category = (
                relative.parts[0]
                if len(relative.parts) > 1
                else "Unknown"
            )

            sessions.append(
                {
                    "session_id": session_id,
                    "category": category,
                    "smartphone": smartphone_path,
                    "vehicle": vehicle_path,
                }
            )

        return sessions

    def load_session(self, session):
        """
        Load one synchronized S/V session.
        """

        smartphone = self.read_csv(
            session["smartphone"]
        )

        vehicle = self.read_csv(
            session["vehicle"]
        )

        return smartphone, vehicle

    def get_sessions(self):
        """
        Return all valid synchronized sessions.
        """

        sessions = self.find_sessions()

        if not sessions:
            raise RuntimeError(
                "No synchronized S/V sessions were found."
            )

        return sessions


if __name__ == "__main__":

    dataset = IOVNBDSynchronizedDataset(
        "data/IO-VNBD/synchronized"
    )

    sessions = dataset.get_sessions()

    print("=" * 70)
    print("IO-VNBD DATASET LOADER TEST")
    print("=" * 70)

    print(f"\nSynchronized sessions found: {len(sessions)}")

    for session in sessions[:10]:

        print(
            f"\n{session['session_id']}"
        )

        print(
            f"Category: {session['category']}"
        )

        print(
            f"Smartphone: {session['smartphone'].name}"
        )

        print(
            f"Vehicle: {session['vehicle'].name}"
        )

    print("\nLoader test completed.")