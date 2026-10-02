# Getting started with REPOL

This walkthrough runs REPOL on one demo earthquake using the demo dataset
([doi:10.5281/zenodo.23105466](https://doi.org/10.5281/zenodo.23105466)).

REPOL is the first component of the weak-motion focal mechanism workflow
(REPOL / RESS -> REHASH / REBayFM). It runs fine on its own; start the other
containers too if you want the full workflow from the same dispatch.

## Prerequisites

- A running SeisComP system with messaging and database.
- Docker Engine with the Compose plugin.
- The demo dataset files.

## 1. Prepare SeisComP

Import the demo inventory into your SeisComP test system.

## 2. Prepare REPOL

```bash
git clone https://github.com/nkua-seismolab/REPOL.git
cd REPOL
cp config.example.yaml config.yaml
```

- In `config.yaml`, set `seiscomp.host` and `seiscomp.database` (defaults may suit
  a default SeisComP installation).
- Unzip `archive.zip` and point the archive volume in `docker-compose.yml` at it, e.g.:

  ```yaml
      - /path/to/unzipped/archive:/data/archive:ro
  ```

Start and watch the logs:

```bash
docker compose up -d --build
docker compose logs -f
```

Wait until you see that REPOL connected and is watching for events.

## 3. Dispatch the demo event

On the SeisComP host:

```bash
scdispatch -i nkua2020abcd_crl_scml.xml -v
```

REPOL picks up the new event, waits `seiscomp.wait_time`, fetches the
waveforms, and processes every P pick. Progress appears in the container
logs.

## 4. Check the results

- Open the event in `scolv`: picks now carry first-motion polarities and
  onsets.
- Each processed pick has a JSON comment (id `<pickID>/comment/repol`) with
  the S/P ratio, visible in the database or via `scxmldump`.

## 5. Rerun

To remove the dispatched objects and run the demo again:

```bash
scdispatch -i nkua2020abcd_crl_scml.xml -v -O remove
```

Set `seiscomp.reprocess: true` in `config.yaml` (and restart the container)
if you want REPOL to reprocess events it has already handled.
