#!/usr/bin/env python3
"""ARM validation for shared .gt files and current sparse dataset prefixes.

Run after sourcing the configured Conan runtime environment. Never modifies
source datasets. Prefixes have their own CSR files and float64 exhaustive truth
inside the benchmark; these timings are correctness smoke evidence only.
"""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csr(path, rows, dim=16):
    offsets, ids, values = [0], [], []
    for row in rows:
        ids.extend(x for x, _ in row)
        values.extend(v for _, v in row)
        offsets.append(len(ids))
    path.write_bytes(struct.pack('<QQQ', len(rows), dim, len(ids))
                     + struct.pack(f'<{len(offsets)}Q', *offsets)
                     + struct.pack(f'<{len(ids)}I', *ids)
                     + struct.pack(f'<{len(values)}f', *values))


def prefix_csr(source, target, limit):
    with source.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        rows, dim, nnz = struct.unpack_from('<QQQ', data)
        count = min(rows, limit)
        end = struct.unpack_from('<Q', data, 24 + count * 8)[0]
        ids = 24 + (rows + 1) * 8
        vals = ids + nnz * 4
        assert len(data) == vals + nnz * 4
        with target.open('wb') as out:
            out.write(struct.pack('<QQQ', count, dim, end))
            out.write(data[24:24 + (count + 1) * 8])
            out.write(data[ids:ids + end * 4])
            out.write(data[vals:vals + end * 4])


def metadata(root, metric, params=None):
    value = {'metric': metric, 'input_sha256': {p.name: sha(p) for p in root.glob('*.csr')},
             'output_sha256': {p.name: sha(p) for p in (root / 'gt').glob('*.gt')}}
    if params is not None:
        value['bm25'] = params
    (root / 'gt/metadata.json').write_text(json.dumps(value, indent=2) + '\n')


def truth(path, ids, scores, queries=1):
    assert len(ids) == len(scores) and len(ids) % queries == 0
    path.write_bytes(struct.pack('<II', queries, len(ids) // queries)
                     + struct.pack(f'<{len(ids)}I', *ids)
                     + struct.pack(f'<{len(scores)}f', *scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/volume/data_sparse2'))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    runs = []

    def run(name, root, metric, quant='fp16', h2=0, k=10, extra=(), error=None):
        target = output / (name + '.csv')
        command = [str(binary), '--data-dir', str(root), '--base-file', str(root / 'corpus.csr'),
                   '--query-file', str(root / 'queries.csr'), '--metric', metric,
                   '--quant-type', quant, '--h2', str(h2), '--k', str(k), '--window', '1024',
                   '--threads', str(args.threads), '--warmup', '1', '--repeats', '1',
                   '--output', str(target), *map(str, extra)]
        with (output / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record = {'name': name, 'command': command, 'exit_code': result.returncode, 'expected_error': error}
        runs.append(record)
        (output / 'commands.json').write_text(json.dumps(runs, indent=2) + '\n')
        log = (output / (name + '.log')).read_text()
        if error:
            assert result.returncode != 0 and error in log, (name, log[-3000:])
            return None
        assert result.returncode == 0, (name, log[-3000:])
        value = json.loads(Path(str(target) + '.json').read_text())
        assert value['independent_hit_scores_passed']
        record.update(hits_sha256=sha(Path(str(target) + '.hits.csv')),
                      index_bytes=value['index_bytes'], recall=value['measurements'][0]['id_recall'])
        print(name + ' passed', flush=True)
        return record

    # Rounded-equal scores must keep the float64 ranking, not resort by ID.
    # The query's tiny second term makes doc 1 rank ahead of doc 0.
    for metric in ('IP', 'BM25'):
        root = output / ('fixture_' + metric)
        (root / 'gt').mkdir(parents=True)
        rows = [[(1, 1)], [(1, 1), (2, 1)], []]
        write_csr(root / 'corpus.csr', rows)
        write_csr(root / 'queries.csr', [[(1, 1), (2, 1e-9)]])
        params = {'k1': 0, 'b': 0.75, 'avgdl': 1} if metric == 'BM25' else None
        gt = root / 'gt/queries.k10.gt'
        truth(gt, [1, 0, 2] + [0xffffffff] * 7, [1, 1, 0] + [0] * 7)
        metadata(root, metric.lower(), params)
        quant = 'u8' if metric == 'BM25' else 'fp16'
        for mode in (0, 1):
            run(metric + '.format.' + str(mode), root, metric, quant, mode,
                extra=['--truth', gt, '--reload-check', '1'])
        # Legacy int64-ID format, truncation, invalid padding, duplicate IDs.
        cases = {
            'legacy': struct.pack('<Q', 10) + struct.pack('<10q', *range(10)) + bytes(40),
            'truncated': gt.read_bytes()[:-1],
            'interleaved': struct.pack('<II', 1, 10) + b''.join(struct.pack('<If', i, 1.) for i in range(10)),
        }
        for name, data in cases.items():
            bad = root / 'gt' / (name + '.gt')
            bad.write_bytes(data)
            metadata(root, metric.lower(), params)
            run(metric + '.reject.' + name, root, metric, quant,
                extra=['--truth', bad], error='ground-truth')
        for name, ids, scores in [
            ('padding', [1, 0, 2] + [0xffffffff] * 7, [1, 1, 0, 1] + [0] * 6),
            ('duplicate', [1, 1, 2] + [0xffffffff] * 7, [1, 1, 0] + [0] * 7),
        ]:
            bad = root / 'gt' / (name + '.gt')
            truth(bad, ids, scores)
            metadata(root, metric.lower(), params)
            run(metric + '.reject.' + name, root, metric, quant,
                extra=['--truth', bad], error='ground-truth')
        manifest_path = root / 'gt/metadata.json'
        original = manifest_path.read_text()
        manifest = json.loads(original)
        manifest['output_sha256'][gt.name] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
        run(metric + '.reject.hash', root, metric, quant, extra=['--truth', gt], error='Truth SHA256')
        manifest_path.write_text(original)

    inventory = []
    for source_meta in sorted(args.data_root.glob('*/*/gt/metadata.json')):
        source = source_meta.parent.parent
        source_manifest = json.loads(source_meta.read_text())
        metric = source_manifest['metric'].upper()
        for name, digest in source_manifest['input_sha256'].items():
            assert sha(source / name) == digest, (source, name, 'input hash mismatch')
        for name, digest in source_manifest['output_sha256'].items():
            assert sha(source / 'gt' / name) == digest, (source, name, 'truth hash mismatch')
        for query_name in source_manifest['input_sha256']:
            if query_name == 'corpus.csr':
                continue
            stem = Path(query_name).stem
            small = (source / 'gt' / (stem + '.k10.gt')).read_bytes()
            large = (source / 'gt' / (stem + '.k100.gt')).read_bytes()
            nq, k10 = struct.unpack_from('<II', small)
            rows100, k100 = struct.unpack_from('<II', large)
            assert nq == rows100 and k10 == 10 and k100 == 100
            assert len(small) == 8 + 8 * nq * k10 and len(large) == 8 + 8 * nq * k100
            for q in range(nq):
                for offset10, offset100 in ((8, 8), (8 + 4 * nq * k10, 8 + 4 * nq * k100)):
                    assert (small[offset10 + q * 40:offset10 + q * 40 + 40] ==
                            large[offset100 + q * 400:offset100 + q * 400 + 40]), (source, stem, q)
        root = output / 'prefixes' / source.parent.name / source.name
        (root / 'gt').mkdir(parents=True)
        prefix_csr(source / 'corpus.csr', root / 'corpus.csr', 4097)
        query_names = sorted(name for name in source_manifest['input_sha256'] if name != 'corpus.csr')
        inventory.append({'dataset': str(source), 'metric': metric, 'queries': query_names,
                          'metadata': source_manifest})
        # Test a natural query file where available; otherwise nnz10 synthetic.
        query = next((name for name in query_names if 'synthetic' not in name), query_names[0])
        prefix_csr(source / query, root / 'queries.csr', 8)
        metadata(root, metric.lower(), source_manifest.get('bm25'))
        codecs = ('u8', 'u16', 'auto') if metric == 'BM25' else ('fp16',)
        for quant in codecs:
            index = root / (quant + '.index')
            for k in (10, 100):
                pair = []
                for mode in (0, 1):
                    extra = ['--brute-force', '1', '--reload-check', '1']
                    if k == 10 and mode == 0:
                        extra += ['--index-save', index]
                    else:
                        extra += ['--index-load', index]
                    name = f'{source.parent.name}.{source.name}.{quant}.k{k}.h2_{mode}'
                    pair.append(run(name, root, metric, quant, mode, k, extra))
                assert pair[0]['hits_sha256'] == pair[1]['hits_sha256'], (source, quant, k, pair)
                assert pair[1]['index_bytes'] >= pair[0]['index_bytes']
    assert inventory, 'No current datasets found'
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (output / 'summary.json').write_text(json.dumps({'passed': True, 'binary_sha256': sha(binary),
        'datasets': len(inventory), 'runs': runs, 'scope': 'format fixtures and 4097-doc/8-query prefixes; no full timings'},
        indent=2) + '\n')
    print(f'{len(runs)} invocations passed across {len(inventory)} dataset prefixes', flush=True)


if __name__ == '__main__':
    main()
