# Copenhagen Networks Study: Physical vs. Digital Connections

This project examines how Facebook friendship, calls, and SMS align with
repeated Bluetooth proximity. Bluetooth is treated as evidence of co-presence,
not direct proof of friendship.

## Run the EDA

Open `notebooks/EDA-RC.ipynb` and run all cells. The first run reads the raw
files and builds a dyad-level analysis table; it takes roughly one minute on
the current workspace.

`notebooks/EDA-relationship-types.ipynb` contains a separate temporal EDA. It
compares class-hour and after-hours proximity, tests whether schedule patterns
align with digital connections, and explores unlabeled temporal archetypes.

The notebook uses these default physical-tie criteria:

- RSSI greater than or equal to -80 dBm;
- at least 12 close five-minute observations; and
- observations on at least 3 distinct days.

These values can be changed in the `load_and_prepare` call. The notebook also
includes a threshold-sensitivity figure.

Generated tables are written to `data/processed/`, and the nine EDA figures
are saved to `results/figures/`.
