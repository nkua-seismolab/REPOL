# REPOL

[![lint](https://github.com/nkua-seismolab/REPOL/actions/workflows/lint.yml/badge.svg)](https://github.com/nkua-seismolab/REPOL/actions/workflows/lint.yml)
[![tests](https://github.com/nkua-seismolab/REPOL/actions/workflows/tests.yml/badge.svg)](https://github.com/nkua-seismolab/REPOL/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Near real-time P-wave first-motion polarity, onset, and S/P amplitude-ratio determination for SeisComP.

REPOL is part of a four-application workflow for automatic weak-event source characterization:

**[REPOL](https://github.com/nkua-seismolab/REPOL)** → [RESS](https://github.com/nkua-seismolab/RESS) → [REHASH](https://github.com/nkua-seismolab/REHASH) → [REBayFM](https://github.com/nkua-seismolab/REBayFM)

## How it works

REPOL connects to the SeisComP messaging system and listens for new events. For every P pick of an event it:

1. Fetches waveforms from a local SDS archive.
2. Classifies the first-motion polarity (up/down) and onset
   (impulsive/emergent) with the [DiTingMotion deep-learning model](https://github.com/mingzhaochina/DiTing-FOCALFLOW).
3. Measures the S/P amplitude ratio (observed or theoretical S arrival).

Polarity and onset are written back onto the pick in the SeisComP database. The
S/P ratio travels as a JSON comment on the pick (`<pickID>/comment/repol`), where
downstream applications such as [REHASH](https://github.com/nkua-seismolab/REHASH) pick it up.

## Requirements

- A running SeisComP system and a local SDS waveform archive.
- Docker Engine with the Compose plugin.
- Resources: ~3 GB disk for the image; roughly 2 GB RAM during processing.
  No GPU required - the default image uses `tensorflow-cpu` (see below).

Tested with SeisComP 7.3.0 on Ubuntu 24.04.1.

> **Optional GPU inference:** replace `tensorflow-cpu` with `tensorflow` in
> `repol/requirements.txt`, rebuild the image, install the NVIDIA Container
> Toolkit, and add a `gpus` entry to `docker-compose.yml`. The model is small,
> so CPU inference is usually sufficient.

## Installation

```bash
git clone https://github.com/nkua-seismolab/REPOL.git
cd REPOL
cp config.example.yaml config.yaml
```

Then:

1. Edit `config.yaml`: set `seiscomp.host` and `seiscomp.database` to your
   SeisComP messaging host and database URL.
2. Edit `docker-compose.yml`: point the SDS archive volume at your archive
   (the container path `/data/archive` must match `sds.archive` in
   `config.yaml`).
3. Start:

   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```

## Configuration

All options live in `config.yaml` and are documented inline in
[config.example.yaml](config.example.yaml). The main sections:

| Section | Purpose |
| ------- | ------- |
| `seiscomp` | Messaging host, database URL, wait time, reprocessing policy |
| `sds` | Path of the SDS waveform archive inside the container |
| `model` | Path of the DiTingMotion model file |
| `waveform` | Acquisition window, sampling rate, optional pre-filter |
| `spratio` | S/P amplitude-ratio measurement windows and filters |
| `processing` | Parallel waveform fetch workers |
| `logging` | Level and optional log file |

## Output

For each processed pick, REPOL updates the SeisComP database with:

- `pick.polarity` (positive/negative) and `pick.onset`
  (impulsive/emergent);
- a JSON comment (id `<pickID>/comment/repol`) carrying the S/P ratio and
  provenance, e.g.:

  ```json
  {"provenance": {"software": "REPOL", "version": "1.0.0", "stored_at": "..."},
   "sp_ratio": "3.500"}
  ```

## Getting started

See [GETTING_STARTED.md](GETTING_STARTED.md) for a step-by-step demo run with
the example event `nkua2020abcd` from the demo dataset
([doi:10.5281/zenodo.23105466](https://doi.org/10.5281/zenodo.23105466)).

## License

[GPL-3.0](LICENSE)

## Funding

This work is part of the [TRANSFORM²](https://www.transform2-project.eu/) project which aims to improve physical and digital infrastructure across Near-Fault Observatories (NFOs) in Europe.

TRANSFORM² is funded by the European Union under project number 101188365 within the HORIZON-INFRA-2024-DEV-01-01 call.

<div align="center">
  <img src="https://www.transform2-project.eu/wp-content/uploads/2022/08/Logo_TRANSFORM2-round-logo-100x100-1.png" alt="TRANSFORM² logo">
</div>
