"""Execute one registered training row, retaining the package's single training path."""

import argparse
import fcntl
from pathlib import Path

from kgr4vrd.artifacts import read_json, verify_run, write_json
from kgr4vrd.campaign import load_campaign
from kgr4vrd.config import load_config
from kgr4vrd.training import train


def run(campaign_path, key, resume=False):
    campaign_path = Path(campaign_path).resolve()
    campaign = load_campaign(campaign_path)
    rows = campaign["development_runs"] + campaign["runs"]
    row = next(r for r in rows if r["key"] == key)
    if row in campaign["runs"] and campaign["status"] != "frozen":
        raise ValueError("Final training requires completed development and frozen choices")
    inventory_path = campaign_path.parent / "status.json"
    inventory = read_json(inventory_path)
    existing = next(r for r in inventory["runs"] if r["key"] == key)
    if existing["status"] == "completed":
        verify_run(Path(existing["selection"]).parent)
        return Path(existing["selection"])
    config = load_config(campaign_path.parent / row["config"])
    directory = train(config, resume=resume)
    verify_run(directory)
    with inventory_path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inventory = read_json(inventory_path)
        item = next(r for r in inventory["runs"] if r["key"] == key)
        item.update(status="completed", selection=str(directory / "selected.json"))
        write_json(inventory_path, inventory)
    return directory / "selected.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(args.campaign, args.key, args.resume))
