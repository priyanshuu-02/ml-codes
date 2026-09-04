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

        # Pairing is case-insensitive on purpose.
        #
        # The dataset is not internally consistent about filename case: the
        # smartphone file "S-Vta2.csv" is paired with the vehicle file
        # "V-vta2.csv". Exact string matching therefore silently discarded every
        # Vta and Vtb session, 40 of the 72 available, which is why split.py
        # warns that it expected 32 sessions. Those 40 sessions load fine; only
        # their filename case differed.
        smartphone_files = {}
        vehicle_files = {}
        display_names = {}

        csv_files = self.categorised_root.rglob("*.csv")

        for path in csv_files:

            prefix = path.name[:2].upper()
            key = path.stem[2:].casefold()

            if prefix == "S-":
                smartphone_files[key] = path
                # Prefer the smartphone spelling for the session identifier so
                # ids stay stable and human readable.
                display_names[key] = path.stem[2:]

            elif prefix == "V-":
                vehicle_files[key] = path
                display_names.setdefault(key, path.stem[2:])

        session_keys = sorted(
            set(smartphone_files.keys())
            & set(vehicle_files.keys())
        )

        sessions = []

        for key in session_keys:

            session_id = display_names[key]
            smartphone_path = smartphone_files[key]
            vehicle_path = vehicle_files[key]

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