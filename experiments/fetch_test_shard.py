"""Download only the first FineVideo shard with resumable HTTP ranges."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path
from threading import Lock, local
from time import sleep

import requests
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url


REPO = "HuggingFaceFV/finevideo"
NAME = "train-00000-of-01357.parquet"
STRIDE = 4 * 1024 * 1024


def download(target):
    token = get_token()
    if not token:
        raise SystemExit("Missing saved Hugging Face login")
    original_url = hf_hub_url(REPO, f"data/{NAME}", repo_type="dataset")
    info = get_hf_file_metadata(original_url, token=token, timeout=45)
    size, digest = info.size, info.etag
    if not size or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit("Hugging Face did not provide the expected size and checksum")
    url = hf_hub_url(REPO, f"data/{NAME}", repo_type="dataset", revision=info.commit_hash)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{NAME}.{digest}.partial")
    journal = partial.with_suffix(".chunks")
    offsets = set(range(0, size, STRIDE))
    if not partial.exists() or not journal.exists() or partial.stat().st_size != size:
        with partial.open("wb") as out:
            out.truncate(size)
        journal.write_text("", encoding="ascii")
    done = offsets.intersection(int(line) for line in journal.read_text(encoding="ascii").splitlines())
    print(f"Resuming {len(done)}/{len(offsets)} ranges", flush=True)
    threads = local()
    lock = Lock()

    def transfer(start):
        if not hasattr(threads, "session"):
            threads.session = requests.Session()
        end = min(start + STRIDE, size) - 1
        headers = {"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}",
                   "Accept-Encoding": "identity"}
        for attempt in range(4):
            try:
                with threads.session.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                    status = response.status_code
                    if status == 206 and response.headers.get("Content-Range") == f"bytes {start}-{end}/{size}":
                        data = response.content
                        if len(data) == end - start + 1:
                            with partial.open("r+b") as out:
                                out.seek(start)
                                out.write(data)
                            with lock, journal.open("a", encoding="ascii") as log:
                                print(start, file=log)
                                done.add(start)
                                if len(done) % 5 == 0 or done == offsets:
                                    print(f"Received {len(done)}/{len(offsets)} ranges", flush=True)
                            return True
            except requests.RequestException as error:
                status = type(error).__name__
            sleep(attempt + 1)
        print(f"Range {start // STRIDE + 1} needs another attempt ({status})", flush=True)
        return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(transfer, sorted(offsets - done)))
    if done != offsets:
        raise SystemExit(f"Download waiting on {len(offsets - done)} ranges")
    with partial.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != digest:
            raise SystemExit("Shard checksum did not match")
    partial.replace(target)
    journal.unlink()
    print(f"Verified {target.name} ({size:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("finevideo/one/data"))
    args = parser.parse_args()
    download(args.output_dir / NAME)
