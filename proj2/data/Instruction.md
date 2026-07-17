# Copenhagen Networks Study: data download instructions

Follow this guide to download an unchanged local copy of **The Copenhagen
Networks Study interaction data** into `data/raw/`. No Figshare account or API
token is required.

## 1. Dataset source

- **Repository:** Figshare
- **Dataset:** [The Copenhagen Networks Study interaction data](https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1)
- **Version:** 1
- **DOI:** [10.6084/m9.figshare.7267433.v1](https://doi.org/10.6084/m9.figshare.7267433.v1)
- **Authors:** Piotr Sapiezynski, Arkadiusz Stopczynski, David Dreyer Lassen,
  and Sune Lehmann Jørgensen
- **License:** MIT
- **Download size:** approximately 99.2 MB (94.6 MiB)

The dataset describes a multi-layer temporal network of more than 700
university students. It includes Bluetooth proximity observations, phone
calls, text messages, Facebook friendships, and participant gender labels.

## 2. Requirements

You need:

- an internet connection;
- about 100 MB of free disk space; and
- `curl`, which is included with macOS and is available on most Linux systems.

Run the commands below **from the `data/` directory that contains this
`Instruction.md` file**. For this repository, that directory is:

```text
26-the-deep-learners-analysis/proj2/data/
```

For example, from the repository root:

```bash
cd 26-the-deep-learners-analysis/proj2/data
```

## 3. Download the raw files

Copy and run this complete command block:

```bash
mkdir -p raw

curl --fail --location --retry 3 --output raw/fb_friends.csv \
  https://ndownloader.figshare.com/files/13389320
curl --fail --location --retry 3 --output raw/genders.csv \
  https://ndownloader.figshare.com/files/13389440
curl --fail --location --retry 3 --output raw/fb_friends.README \
  https://ndownloader.figshare.com/files/13389839
curl --fail --location --retry 3 --output raw/bt_symmetric.csv \
  https://ndownloader.figshare.com/files/14000795
curl --fail --location --retry 3 --output raw/bt_symmetric.README \
  https://ndownloader.figshare.com/files/14039048
curl --fail --location --retry 3 --output raw/sms.README \
  https://ndownloader.figshare.com/files/14127308
curl --fail --location --retry 3 --output raw/calls.README \
  https://ndownloader.figshare.com/files/14127311
curl --fail --location --retry 3 --output raw/Copenhagen_Networks_Study_Notebook.ipynb \
  https://ndownloader.figshare.com/files/14574485
curl --fail --location --retry 3 --output raw/calls.csv \
  https://ndownloader.figshare.com/files/14579972
curl --fail --location --retry 3 --output raw/sms.csv \
  https://ndownloader.figshare.com/files/14579975
```

The files are downloaded individually and are not compressed, so there is no
archive to unzip.

### What the commands do

- `mkdir -p raw` creates the destination directory if it does not already
  exist. It does not remove existing files.
- `curl` downloads a file from its permanent Figshare file URL.
- `--fail` makes HTTP errors cause the command to fail instead of saving an
  error page as data.
- `--location` follows redirects from Figshare's download service.
- `--retry 3` retries a download up to three times after a temporary failure.
- `--output raw/<filename>` saves the file under its original name in `raw/`.

Running the block again replaces files with fresh copies from Figshare.

## 4. Verify the download

First, confirm that all ten files are present:

```bash
ls -lh raw
```

The expected layout is:

```text
data/
├── Instruction.md
└── raw/
    ├── Copenhagen_Networks_Study_Notebook.ipynb
    ├── bt_symmetric.README
    ├── bt_symmetric.csv
    ├── calls.README
    ├── calls.csv
    ├── fb_friends.README
    ├── fb_friends.csv
    ├── genders.csv
    ├── sms.README
    └── sms.csv
```

For a stronger integrity check, run the following from `data/`. It works on
macOS and Linux and compares each file with the MD5 checksum published by
Figshare:

```bash
cd raw

check_md5() {
  expected="$1"
  file="$2"

  if command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "$file" | awk '{print $1}')"
  else
    actual="$(md5 -q "$file")"
  fi

  if [ "$actual" = "$expected" ]; then
    printf '%s: OK\n' "$file"
  else
    printf '%s: FAILED\n' "$file"
  fi
}

check_md5 b1f96bae3e79949ed4278e90d0753843 fb_friends.csv
check_md5 eb033aae23fff859684a82d00ffd8099 genders.csv
check_md5 86dcef05caae3c0eca1dd99f75ee6960 fb_friends.README
check_md5 98892459f73e774cf79e7977edfeee3e bt_symmetric.csv
check_md5 291ddeee8c92b73134946abe375302c2 bt_symmetric.README
check_md5 02257591786710c11769b61d7748e389 sms.README
check_md5 669bd1df29e91b0dea7c50760336d911 calls.README
check_md5 c60cb026ad888e54a7e98ff19d4b1e1a Copenhagen_Networks_Study_Notebook.ipynb
check_md5 50a7c5be85d7e3eaf5fabd267a87008d calls.csv
check_md5 3d24cef3717881b3d9086584332dd740 sms.csv

cd ..
```

Every line should report `OK`. If a file fails, rerun its `curl` command and
then repeat the integrity check.

## 5. Raw-file reference

| Raw file | Contents |
|---|---|
| `fb_friends.csv` | Facebook friendship edge list with `user_a` and `user_b`. |
| `genders.csv` | Participant ID and binary `female` indicator. |
| `fb_friends.README` | Author-provided column and Facebook-edge description. |
| `bt_symmetric.csv` | Timestamped Bluetooth proximity observations with participant IDs and received signal strength (`rssi`). The accompanying README explains the `-1` and `-2` sentinel IDs. |
| `bt_symmetric.README` | Author-provided Bluetooth column descriptions and sentinel-value notes. |
| `sms.README` | Author-provided SMS column descriptions. |
| `calls.README` | Author-provided call column descriptions; a duration of `-1` denotes a missed call. |
| `Copenhagen_Networks_Study_Notebook.ipynb` | Author-provided Jupyter notebook demonstrating the dataset. |
| `calls.csv` | Timestamped caller, callee, and call-duration records. No call content is included. |
| `sms.csv` | Timestamped sender and recipient records. No message content is included. |

Some CSV files begin with a header line prefixed by `#`. The Python example
below handles both prefixed and ordinary headers.

## 6. Preview the first five observations in Python

Install pandas if it is not already available:

```bash
python3 -m pip install pandas
```

Then, while still in the `data/` directory, run:

```bash
python3 - <<'PY'
from pathlib import Path

import pandas as pd

raw_dir = Path("raw")

for filename in (
    "fb_friends.csv",
    "genders.csv",
    "bt_symmetric.csv",
    "calls.csv",
    "sms.csv",
):
    with (raw_dir / filename).open(encoding="utf-8") as file:
        columns = file.readline().lstrip("# ").strip().split(",")
        data = pd.read_csv(file, names=columns)
    print(f"\n{filename}")
    print(data.head(5).to_string(index=False))
PY
```

This reads and displays the data only; it does not modify the raw files.

## 7. Raw-data rule

Keep every downloaded file unchanged in `data/raw/`. At this stage:

- do not rename columns or files;
- do not clean, filter, or remove rows;
- do not create `train.csv` or `test.csv`;
- do not write transformed data into `raw/`;
- do not train a model or calculate accuracy; and
- do not create visualizations.

Any later intermediate or processed outputs should go in the project's
`data/interim/` or `data/processed/` directories, never in `data/raw/`.
