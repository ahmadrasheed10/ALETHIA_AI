"""Compare a candidate against a reference or audit and compress a full run."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from run import NAMES


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def compare(reference, candidate):
    def index(path):
        rows = {}
        errors = set()
        for record in records(path):
            video = record["video"]
            if "error" in record:
                if video in errors:
                    raise ValueError(f"duplicate error in {path}: {video}")
                errors.add(video)
                continue
            key = video, record["second"]
            if key in rows:
                raise ValueError(f"duplicate second in {path}: {key}")
            rows[key] = set(record["objects"]), record["people"]
        return rows, errors

    expected, expected_errors = index(reference)
    actual, actual_errors = index(candidate)
    if expected.keys() != actual.keys() or expected_errors != actual_errors:
        raise ValueError("video/second coverage or error video IDs differ")
    if not expected:
        raise ValueError("reference contains no scored seconds")

    tp = fp = fn = exact = within_one = 0
    for key, (wanted_objects, wanted_people) in expected.items():
        objects, people = actual[key]
        tp += len(objects & wanted_objects)
        fp += len(objects - wanted_objects)
        fn += len(wanted_objects - objects)
        exact += people == wanted_people
        within_one += abs(people - wanted_people) <= 1
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 1.0
    exact_rate = exact / len(expected)
    within_one_rate = within_one / len(expected)
    print(json.dumps({"seconds": len(expected), "micro_object_f1": f1,
                      "people_exact": exact_rate, "people_within_one": within_one_rate}))
    return f1 >= 0.95 and exact_rate >= 0.85 and within_one_rate >= 0.95


def package(folder, result, destination):
    expected = {}
    required = {f"train-{i:05d}-of-01357.parquet" for i in range(50)}
    found = {path.name for path in folder.glob("*.parquet")}
    if found != required:
        raise ValueError(f"expected exactly shards 0–49; missing={sorted(required - found)}, "
                         f"extra={sorted(found - required)}")
    for name in sorted(required):
        with pq.ParquetFile(folder / name) as shard:
            for row in range(shard.metadata.num_rows):
                expected[f"{name}:{row}"] = 0

    errors = set()
    seconds = 0
    for record in records(result):
        video = record["video"]
        if video not in expected or video in errors:
            raise ValueError(f"unexpected video or record following an error: {video}")
        if set(record) == {"video", "error"} and isinstance(record["error"], str) and record["error"]:
            errors.add(video)
            continue
        if set(record) != {"video", "second", "objects", "people"}:
            raise ValueError(f"invalid record fields: {record}")
        second, objects, people = record["second"], record["objects"], record["people"]
        if (type(second) is not int or second != expected[video] or type(people) is not int
                or people < 0 or not isinstance(objects, list)
                or any(name not in NAMES for name in objects) or len(objects) != len(set(objects))):
            raise ValueError(f"invalid second record: {record}")
        expected[video] += 1
        seconds += 1

    missing = [video for video, count in expected.items() if not count and video not in errors]
    if missing or not seconds:
        raise ValueError(f"incomplete results: {len(missing)} videos missing, {seconds} seconds")
    if result.suffix == ".gz" or destination.suffix != ".gz":
        raise ValueError("provide an uncompressed input and a .gz output")
    temporary = destination.with_name(destination.name + ".incomplete")
    with result.open("rb") as source, temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    temporary.replace(destination)
    print(json.dumps({"shards": len(required), "videos": len(expected),
                      "seconds": seconds, "videos_with_errors": len(errors)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    packaging = commands.add_parser("package")
    packaging.add_argument("shards", type=Path)
    packaging.add_argument("results", type=Path)
    packaging.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        if not compare(args.reference, args.candidate):
            raise SystemExit(1)
    else:
        package(args.shards, args.results, args.destination)


if __name__ == "__main__":
    main()
