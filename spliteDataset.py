
import io
import json
import re
import zipfile

from collections import Counter
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "InTAS_highway_2_constantPositionOffset.zip"
)

SPLITS = ("Train", "Validation", "Test")

VALID_LABELS = {0, 1}

WRAPPER_KEYS = (
    "data",
    "records",
    "messages",
    "items",
)


# ============================================================
# Split detection
# ============================================================

def detect_split(path, inherited_split=None):
    """
    Detect the official split from a ZIP/member path.

    Supports names such as:
        Train/
        Validation/
        Val/
        Test/

    If a nested ZIP does not specify its own split, the split
    inherited from its parent archive is retained.
    """

    normalized = str(path).replace("\\", "/")
    parts = normalized.split("/")

    for part in reversed(parts):
        name = re.sub(r"[^a-z]", "", part.lower())

        if name == "train":
            return "Train"

        if name in {"validation", "valid", "val", "verify"}:
            return "Validation"

        if name == "test":
            return "Test"

    return inherited_split


# ============================================================
# Receiver ID extraction
# ============================================================

def extract_receiver_id(source_name):
    """
    Extract receiver identity from a JSON filename.

    Examples recognized:
        veh_1521.json       -> 1521
        veh_1521.jsonl      -> 1521
        receiver_1521.json  -> 1521
        receiver-1521.json  -> 1521
        rx_1521.json        -> 1521
        rx-1521.json        -> 1521

    Returns None if the filename does not match a known pattern.

    The original source path is preserved separately, so no
    information about the file's origin is discarded.
    """

    normalized = str(source_name).replace("\\", "/")
    filename = normalized.rsplit("/", 1)[-1]

    extensions = r"(?:json|jsonl|ndjson)"

    patterns = (
        rf"^veh_(.+?)\.{extensions}$",
        rf"^receiver[_-](.+?)\.{extensions}$",
        rf"^rx[_-](.+?)\.{extensions}$",
    )

    for pattern in patterns:
        match = re.match(
            pattern,
            filename,
            flags=re.IGNORECASE,
        )

        if match:
            receiver_id = match.group(1).strip()

            if receiver_id:
                return receiver_id

    return None


# ============================================================
# JSON parsing helpers
# ============================================================

def _unwrap_records(obj):
    """
    Convert supported JSON structures into a list of dictionaries.

    Supported forms:
        {"sender_id": ..., ...}
        [{"sender_id": ...}, ...]
        {"records": [...]}
        {"data": [...]}
        {"messages": [...]}
        {"items": [...]}

    Non-dictionary entries are ignored.
    """

    if isinstance(obj, dict):
        # A record containing a label or sender information is
        # treated as a record rather than a wrapper.
        record_markers = {
            "attacker",
            "sender_id",
            "sender",
            "receiver",
            "rcvTime",
            "sendTime",
            "messageID",
        }

        if record_markers.intersection(obj.keys()):
            return [obj]

        # Handle wrapper objects.
        for key in WRAPPER_KEYS:
            value = obj.get(key)

            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

            if isinstance(value, dict):
                nested = _unwrap_records(value)

                if nested:
                    return nested

        # Unknown dictionary structure: do not silently interpret
        # its contents as records.
        return []

    if isinstance(obj, list):
        return [
            item
            for item in obj
            if isinstance(item, dict)
        ]

    return []


def _parse_json_lines(text):
    """
    Parse JSON Lines / NDJSON content.

    Blank lines are ignored. Invalid lines are counted and
    reported to the caller.
    """

    records = []
    invalid_lines = 0

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue

        if isinstance(item, dict):
            records.append(item)
        elif isinstance(item, list):
            records.extend(
                value
                for value in item
                if isinstance(value, dict)
            )

    return records, invalid_lines


def parse_json_bytes(data, source_name):
    """
    Parse a JSON/JSONL member and preserve its provenance.

    Each record retains its original fields and gains:
        receiver_id: extracted from the source filename, if known
        source_file: original member path inside the ZIP hierarchy

    Existing receiver_id/source_file values are not overwritten.
    """

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    stripped = text.strip()

    if not stripped:
        return []

    records = []

    # First attempt: standard JSON.
    try:
        parsed = json.loads(stripped)
        records = _unwrap_records(parsed)

    except json.JSONDecodeError:
        # Fallback: JSON Lines / NDJSON.
        records, invalid_lines = _parse_json_lines(stripped)

        if invalid_lines:
            print(
                "[WARNING] "
                f"{source_name}: skipped "
                f"{invalid_lines} invalid JSON line(s)"
            )

    filename_receiver_id = extract_receiver_id(source_name)

    output = []

    for record in records:
        # Copy to avoid modifying a nested object returned by parsing.
        item = dict(record)

        # Keep existing values, if present.
        if item.get("receiver_id") is None and filename_receiver_id is not None:
            item["receiver_id"] = filename_receiver_id

        item.setdefault("source_file", str(source_name))

        output.append(item)

    return output


# ============================================================
# Recursive ZIP reader
# ============================================================

def _is_json_member(name):
    return name.lower().endswith(
        (".json", ".jsonl", ".ndjson")
    )


def _is_zip_member(name):
    return name.lower().endswith(".zip")


def read_zip_recursive(
    zip_bytes,
    archive_name,
    inherited_split=None,
):
    """
    Recursively read nested ZIP archives without extracting them.

    Returns a dictionary:
        {
            "Train": [...],
            "Validation": [...],
            "Test": [...]
        }

    Official split membership is inherited from parent ZIP paths
    when nested ZIP filenames do not contain a split name.
    """

    result = {
        split: []
        for split in SPLITS
    }

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:

            for member in zf.infolist():
                if member.is_dir():
                    continue

                member_name = member.filename

                full_name = (
                    f"{archive_name.rstrip('/')}/{member_name}"
                    if archive_name
                    else member_name
                )

                current_split = detect_split(
                    full_name,
                    inherited_split=inherited_split,
                )

                try:
                    member_bytes = zf.read(member)
                except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                    print(
                        "[WARNING] Cannot read archive member "
                        f"{full_name}: {exc}"
                    )
                    continue

                if _is_zip_member(member_name):
                    nested = read_zip_recursive(
                        zip_bytes=member_bytes,
                        archive_name=full_name,
                        inherited_split=current_split,
                    )

                    for split in SPLITS:
                        result[split].extend(nested[split])

                    continue

                if not _is_json_member(member_name):
                    continue

                if current_split not in result:
                    print(
                        "[WARNING] JSON file has no identifiable "
                        f"official split and was skipped: {full_name}"
                    )
                    continue

                records = parse_json_bytes(
                    data=member_bytes,
                    source_name=full_name,
                )

                result[current_split].extend(records)

    except zipfile.BadZipFile as exc:
        print(
            f"[WARNING] Invalid ZIP archive {archive_name}: {exc}"
        )

    return result


# ============================================================
# Record field helpers
# ============================================================

def get_label(record):
    """
    Return a normalized binary attacker label.

    Expected values:
        0 -> benign
        1 -> malicious

    Missing, invalid, or unsupported labels return None.
    Boolean values are accepted as binary labels.
    """

    if not isinstance(record, dict):
        return None

    value = record.get("attacker")

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if value in VALID_LABELS:
            return int(value)
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"0", "false", "benign", "normal"}:
            return 0

        if normalized in {"1", "true", "malicious", "attack"}:
            return 1

    return None


def get_sender(record):
    """
    Return sender_id as a string, if available.

    String conversion makes values consistent when the dataset
    contains IDs represented by different JSON types.
    """

    value = record.get("sender_id")

    if value is not None:
        return str(value)

    sender = record.get("sender")

    if isinstance(sender, dict):
        value = sender.get("sender_id")
        if value is not None:
            return str(value)

    return None


def get_receiver_id(record):
    """
    Return receiver identity preserved from the filename or record.
    """

    value = record.get("receiver_id")

    if value is not None and str(value).strip():
        return str(value).strip()

    return None


def get_rcv_time(record):
    """
    Return rcvTime as a float when possible.
    """

    value = record.get("rcvTime")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Reporting
# ============================================================

def print_split_report(split_name, records):
    """
    Print label counts, ID coverage, time range and basic quality
    checks without modifying or dropping any loaded records.
    """

    labels = Counter()
    senders = set()
    receivers = set()
    source_files = set()

    missing_sender_id = 0
    missing_receiver_id = 0
    missing_rcv_time = 0
    missing_source_file = 0

    valid_times = []

    for record in records:
        label = get_label(record)

        if label == 0:
            labels["benign"] += 1
        elif label == 1:
            labels["malicious"] += 1
        elif record.get("attacker") is None:
            labels["missing_label"] += 1
        else:
            labels["invalid_label"] += 1

        sender_id = get_sender(record)

        if sender_id is None:
            missing_sender_id += 1
        else:
            senders.add(sender_id)

        receiver_id = get_receiver_id(record)

        if receiver_id is None:
            missing_receiver_id += 1
        else:
            receivers.add(receiver_id)

        source_file = record.get("source_file")

        if source_file is None:
            missing_source_file += 1
        else:
            source_files.add(str(source_file))

        rcv_time = get_rcv_time(record)

        if rcv_time is None:
            missing_rcv_time += 1
        else:
            valid_times.append(rcv_time)

    print()
    print("=" * 72)
    print(f"Split: {split_name}")
    print("=" * 72)

    print(f"Total records:          {len(records):,}")
    print(f"Benign (attacker=0):    {labels['benign']:,}")
    print(f"Malicious (attacker=1): {labels['malicious']:,}")
    print(f"Missing labels:         {labels['missing_label']:,}")
    print(f"Invalid labels:         {labels['invalid_label']:,}")

    print("-" * 72)

    print(f"Unique sender IDs:      {len(senders):,}")
    print(f"Unique receiver IDs:    {len(receivers):,}")
    print(f"Unique source files:    {len(source_files):,}")

    print(f"Missing sender IDs:     {missing_sender_id:,}")
    print(f"Missing receiver IDs:   {missing_receiver_id:,}")
    print(f"Missing source files:   {missing_source_file:,}")
    print(f"Missing rcvTime:        {missing_rcv_time:,}")

    if valid_times:
        valid_times.sort()

        print(f"rcvTime minimum:        {valid_times[0]}")
        print(f"rcvTime median:         {valid_times[len(valid_times) // 2]}")
        print(f"rcvTime maximum:        {valid_times[-1]}")

    if missing_receiver_id:
        print(
            "[WARNING] Some records do not have receiver_id. "
            "Check the actual JSON filenames and update "
            "extract_receiver_id() if needed."
        )

    if labels["missing_label"] or labels["invalid_label"]:
        print(
            "[WARNING] Some records do not have valid binary labels."
        )


# ============================================================
# Dataset loader
# ============================================================

def load_dataset(dataset_path=None):
    """
    Load the official Train / Validation / Test splits.

    No random splitting is performed.
    Nested ZIP files are read in memory.
    """

    if dataset_path is None:
        dataset_path = DATASET_PATH
    else:
        dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset ZIP file was not found:\n"
            f"{dataset_path}\n\n"
            "Check DATASET_PATH and ensure the archive exists."
        )

    if not zipfile.is_zipfile(dataset_path):
        raise zipfile.BadZipFile(
            f"Dataset file is not a valid ZIP archive: {dataset_path}"
        )

    print("=" * 72)
    print("InTAS official split loader")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print("Nested ZIP extraction: disabled")
    print("Split policy: official Train / Validation / Test")
    print("Label field: attacker")
    print("Receiver identity: extracted from source filename when possible")

    archive_bytes = dataset_path.read_bytes()

    splits = read_zip_recursive(
        zip_bytes=archive_bytes,
        archive_name=dataset_path.name,
        inherited_split=None,
    )

    # Report each official split independently.
    for split_name in SPLITS:
        print_split_report(
            split_name=split_name,
            records=splits[split_name],
        )

    # Combined totals.
    all_records = [
        record
        for split_name in SPLITS
        for record in splits[split_name]
    ]

    total_labels = Counter()

    for record in all_records:
        label = get_label(record)

        if label == 0:
            total_labels["benign"] += 1
        elif label == 1:
            total_labels["malicious"] += 1
        elif record.get("attacker") is None:
            total_labels["missing_label"] += 1
        else:
            total_labels["invalid_label"] += 1

    print()
    print("=" * 72)
    print("Combined dataset summary")
    print("=" * 72)

    print(f"Total records:          {len(all_records):,}")
    print(f"Train records:          {len(splits['Train']):,}")
    print(f"Validation records:     {len(splits['Validation']):,}")
    print(f"Test records:            {len(splits['Test']):,}")
    print(f"Benign records:         {total_labels['benign']:,}")
    print(f"Malicious records:      {total_labels['malicious']:,}")
    print(f"Missing labels:         {total_labels['missing_label']:,}")
    print(f"Invalid labels:         {total_labels['invalid_label']:,}")

    expected_counts = {
        "Train": 37701,
        "Validation": 17977,
        "Test": 44259,
    }

    print()
    print("Official split count checks:")

    for split_name, expected in expected_counts.items():
        actual = len(splits[split_name])
        status = "OK" if actual == expected else "CHECK"

        print(
            f"  {split_name:12s}: "
            f"expected={expected:,}, actual={actual:,} [{status}]"
        )

    if len(all_records) != sum(expected_counts.values()):
        print(
            "[WARNING] Total record count differs from the expected "
            "official split counts. Check split detection and archive layout."
        )

    if (
        total_labels["missing_label"] > 0
        or total_labels["invalid_label"] > 0
    ):
        print(
            "[WARNING] Some records have missing or invalid labels. "
            "Inspect them before training."
        )

    return (
        splits["Train"],
        splits["Validation"],
        splits["Test"],
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_records, validation_records, test_records = load_dataset()

    print()
    print("=" * 72)
    print("Loader finished")
    print("=" * 72)

    print(f"Train records:      {len(train_records):,}")
    print(f"Validation records: {len(validation_records):,}")
    print(f"Test records:       {len(test_records):,}")

    # Show one record's provenance for verification.
    for split_name, records in (
        ("Train", train_records),
        ("Validation", validation_records),
        ("Test", test_records),
    ):
        if records:
            sample = records[0]

            print()
            print(f"Sample metadata from {split_name}:")
            print(f"  receiver_id: {get_receiver_id(sample)}")
            print(f"  sender_id:   {get_sender(sample)}")
            print(f"  source_file: {sample.get('source_file')}")
            print(f"  attacker:    {get_label(sample)}")
