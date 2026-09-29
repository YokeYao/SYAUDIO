"""Download and verify the SYAUDIO files from the verified Hugging Face repository (no model needed)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from huggingface_hub import hf_hub_download

DEFAULT_REPO = 'YokyYao/ALM-Sycophancy'
CORE = ('mmau', 'mmar', 'gsm8k', 'mmlu')
HUMAN = ('human_annie', 'human_danielle', 'human_junchi')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def extract_verified(archive: Path, destination: Path, records: list[dict]) -> None:
    """Extract only manifest-listed regular files; preserve differing local data."""
    expected = {x['path']: x for x in records}
    seen = set()
    destination = destination.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            rel = PurePosixPath(name)
            if name not in expected or not member.isfile() or rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Unexpected or unsafe archive member: {name}')
            if name in seen:
                raise ValueError(f'Duplicate archive member: {name}')
            seen.add(name)
            target = destination.joinpath(*rel.parts)
            try:
                target.resolve().relative_to(destination)
            except ValueError:
                raise ValueError(f'Path escapes destination: {name}') from None
            record = expected[name]
            if member.size != record['bytes']:
                raise ValueError(f'Size mismatch: {name}')
            if target.exists():
                if target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']:
                    continue
                raise FileExistsError(f'Preserving differing local file: {target}; use a fresh --destination.')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.syaudio-part')
            try:
                with tar.extractfile(member) as src, temporary.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                if sha256(temporary) != record['sha256']:
                    raise ValueError(f'Checksum mismatch: {name}')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError(f'Missing archive members: {sorted(set(expected) - seen)}')

def install_direct_file(record: dict, destination: Path, **download_options) -> None:
    rel = PurePosixPath(record['path'])
    destination = destination.resolve()
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe manifest path: {rel}')
    target = destination.joinpath(*rel.parts)
    try:
        target.resolve().relative_to(destination)
    except ValueError:
        raise ValueError(f'Path escapes destination: {rel}') from None
    if target.exists():
        if target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']:
            return
        raise FileExistsError(f'Preserving differing local file: {target}; use a fresh --destination.')
    source = Path(hf_hub_download(filename=record['remote_path'], **download_options))
    if source.stat().st_size != record['bytes'] or sha256(source) != record['sha256']:
        raise ValueError(f'Remote file differs from the verified release: {record["remote_path"]}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.syaudio-part')
    try:
        with source.open('rb') as src, temporary.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    index = json.loads(Path(__file__).with_name('syaudio-files.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', default=DEFAULT_REPO)
    p.add_argument('--revision', default=index['revision'], help='Verified dataset commit by default.')
    p.add_argument('--groups', nargs='+', default=list(CORE), choices=[*CORE, *HUMAN, 'all'])
    p.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--cache-dir', type=Path, default=None)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    kwargs = dict(repo_id=args.repo_id, repo_type='dataset', revision=args.revision)
    if args.cache_dir is not None:
        kwargs['cache_dir'] = str(args.cache_dir)
    groups = set([*CORE, *HUMAN] if 'all' in args.groups else args.groups)
    records = [x for x in index['files'] if x['group'] in groups]
    # The human supplement is already tracked in GitHub, but not yet on HF.
    for record in records:
        if record['group'] in HUMAN and not (args.destination / record['path']).exists():
            raise FileNotFoundError('Human validation files currently come with the GitHub clone. '
                                    'Use that clone as --destination; HF supplemental upload is pending.')
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    install = partial(install_direct_file, destination=args.destination, **kwargs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(install, records), 1):
            if i % 100 == 0 or i == len(records):
                print(f'Verified {i}/{len(records)} files', flush=True)
    print(f'SYAUDIO ready at {args.destination.resolve() / "benchmark"}')

if __name__ == '__main__':
    main()
