"""The single supported command-line entry point."""

import argparse
import json
import logging
from pathlib import Path

from .artifacts import verify_run
from .config import data_path, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kgr4vrd", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--artifact-root", type=Path)
        command.add_argument("--device", choices=("cpu", "cuda", "mps"))
        if name == "prepare":
            command.add_argument(
                "--stage", choices=("data", "reasoning", "features"), required=True
            )
        if name == "train":
            command.add_argument("--resume", action="store_true")
    for name in ("plan", "freeze", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--group")
        else:
            command.add_argument("--output", type=Path, required=True)
    command = subparsers.add_parser("predict")
    command.add_argument("--selection", type=Path, required=True)
    command.add_argument("--split", choices=("train", "validation", "test"), default="test")
    command.add_argument("--campaign", type=Path)
    command.add_argument("--data-root", type=Path)
    command = subparsers.add_parser("evaluate")
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--campaign", type=Path)
    command.add_argument(
        "--filter", choices=("none", "ontology", "simple_asserted"), default="none"
    )
    command.add_argument("--data-root", type=Path)
    subparsers.add_parser("verify").add_argument("--run", type=Path, required=True)
    subparsers.add_parser("smoke").add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = _dispatch(args)
    except FileNotFoundError as error:
        parser.exit(3, f"Missing prerequisite: {error}. Prepare or supply this input first.\n")
    except (ValueError, KeyError, TypeError, FileExistsError) as error:
        parser.exit(2, f"Invalid configuration or artifact: {error}\n")
    except (RuntimeError, OSError, FloatingPointError) as error:
        parser.exit(4, f"Execution failed: {error}\n")
    print(json.dumps(result, default=str, indent=2))
    return 0


def _dispatch(args: argparse.Namespace):
    if args.command in {"inspect", "prepare", "train"}:
        config = load_config(
            args.config,
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            device=args.device,
        )
        if args.command == "inspect":
            path = data_path(config, config["dataset"]["manifest"])
            if not path.exists():
                return {"dataset": str(path), "status": "missing", "next": "prepare --stage data"}
            from .data import load_objects, load_vocabulary

            vocabulary = load_vocabulary(path)
            counts = {
                split: len(load_objects(path, split)) for split in ("train", "validation", "test")
            }
            knowledge = data_path(config, config["intervention"]["knowledge_manifest"])
            return {
                "status": "prepared",
                "partitions": counts,
                "classes": len(vocabulary.object_classes),
                "predicates": len(vocabulary.predicates),
                "knowledge_available": knowledge.exists(),
                "expected_final_runs": 50,
                "expected_primary_views": 110,
            }
        if args.command == "train":
            from .training import train

            return train(config, resume=args.resume)
        if args.stage == "data":
            from .adapters.nesy4vrd import prepare_dataset

            return prepare_dataset(config)
        if args.stage == "reasoning":
            from .reasoning import prepare_reasoning

            return prepare_reasoning(config)
        from .adapters.vision import prepare_features

        return prepare_features(config)
    if args.command in {"plan", "freeze"}:
        from .campaign import freeze, plan

        return (
            plan(args.campaign, args.output)
            if args.command == "plan"
            else freeze(args.campaign, args.group)
        )
    if args.command == "predict":
        from .prediction import predict

        return predict(
            args.selection, split=args.split, campaign=args.campaign, data_root=args.data_root
        )
    if args.command == "evaluate":
        from .prediction import evaluate_predictions

        return evaluate_predictions(
            args.predictions,
            campaign=args.campaign,
            filter_mode=args.filter,
            data_root=args.data_root,
        )
    if args.command == "report":
        from .reporting import summarise

        return summarise(args.campaign, args.output)
    if args.command == "verify":
        return verify_run(args.run)
    from .smoke import smoke

    return smoke(args.output)
