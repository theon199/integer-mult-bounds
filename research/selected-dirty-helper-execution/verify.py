#!/usr/bin/env python3
"""Independent exact execution of one selected dirty-helper operator column.

Sricharan Vagvala, with substantial OpenAI Codex assistance. Apache-2.0.
No predecessor executable code is imported. NumPy supplies checked integer arrays.
"""
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIM, HELPER = 512, 557
PREFIX = 'candidate/strict/global/base/providers/gauge44/'
ARCHIVE = 'research/community-frontier1094-audit/selected-replay.tar.gz'
MEMBERS = {
    'CANONICAL-WORD.json.gz': '16bc0ebb04b4be6c0ade74badc6a0c0cdba4215eda86e4ff9f13abf8ecebd219',
    'CLIFFORD-RECIPES.json': 'd0768f2164d5f64a2fdbec422f1cfde4e400818792f1904ca541958377243731',
    'LOCAL-PROGRAMS.json.gz': '241fde27d4e43f5a9cb6b128f9f76c5131a53d09f0371676163f914d0b0fef3a',
    'RESIDUAL-ADAPTERS.json': '383f40aadd00a4ac700dfd0d4c22d6c6c0be9a9863116e4d69c7acd23bafae0c',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def reject_constant(value):
        raise ValueError('non-JSON numeric constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def load(root=ROOT):
    sources = strict_json((HERE / 'SOURCES.json').read_bytes())
    for row in sources['files']:
        if row['usage'] == 'input':
            need(hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256'],
                 'source pin: ' + row['path'])
    objects = []
    with tarfile.open(root / ARCHIVE) as archive:
        for basename, expected in MEMBERS.items():
            matches = [m for m in archive.getmembers() if m.name == PREFIX + basename]
            need(len(matches) == 1 and matches[0].isfile(), 'unique regular archive member')
            raw = archive.extractfile(matches[0]).read()
            need(hashlib.sha256(raw).hexdigest() == expected, 'archive member pin: ' + basename)
            objects.append(strict_json(gzip.decompress(raw) if basename.endswith('.gz') else raw))
    return objects


def guard(array, multiplier=1):
    # The sum of at most four operands below 2^60 is inside signed int64.
    need(array.dtype == np.int64 and array.size > 0, 'integer array')
    maximum = max(int(array.max()), -int(array.min()))
    need(maximum * abs(multiplier) < 2**60, 'exact int64 arithmetic bound')


def reduced(array, exponent):
    need(type(exponent) is int and exponent >= 0, 'nonnegative dyadic exponent')
    guard(array)
    combined = int(np.bitwise_or.reduce(np.abs(array).ravel(), initial=np.int64(0)))
    if combined == 0:
        return None
    shift = min(exponent, (combined & -combined).bit_length() - 1)
    return (array >> shift if shift else array), exponent - shift


def scale(value, coefficient):
    q = Fraction(coefficient)
    need(q.denominator & (q.denominator - 1) == 0, 'dyadic scalar')
    if value is None or q == 0:
        return None
    a, e = value
    guard(a, q.numerator)
    return reduced(a * q.numerator, e + q.denominator.bit_length() - 1)


def plus(left, right):
    if left is None:
        return right
    if right is None:
        return left
    a, ea = left
    b, eb = right
    e = max(ea, eb)
    guard(a, 1 << (e - ea))
    guard(b, 1 << (e - eb))
    return reduced((a << (e - ea)) + (b << (e - eb)), e)


class Engine:
    """Forward push on independent Gaussian-integer payload columns."""
    def __init__(self):
        self.tables = {}

    def table(self, record, adjoint=False):
        key = json.dumps(record, sort_keys=True), adjoint
        if key in self.tables:
            return self.tables[key]
        need(set(record) == {'matrix_rows', 'quadratic_rows', 'linear_Z4', 'offset', 'phase'},
             'monomial schema')
        rows, quad, linear = [record[k] for k in ('matrix_rows', 'quadratic_rows', 'linear_Z4')]
        need(len(rows) == len(quad) == len(linear) == 9, 'nine active axes')
        need(all(type(v) is int and 0 <= v < DIM for v in rows + quad), 'binary masks')
        need(all(type(v) is int and 0 <= v < 4 for v in linear), 'linear phase')
        need(type(record['offset']) is int and 0 <= record['offset'] < DIM
             and type(record['phase']) is int and 0 <= record['phase'] < 4, 'monomial constants')
        columns = [sum(((rows[j] >> i) & 1) << j for j in range(9)) for i in range(9)]
        destinations, angles = [], []
        for x in range(DIM):
            support = [j for j in range(9) if x & (1 << j)]
            y, p = record['offset'], record['phase']
            for j in support:
                y ^= columns[j]
                p += linear[j]
            for j in support:
                for k in support:
                    if j < k:
                        p += 2 * ((quad[j] >> k) & 1)
            destinations.append(y)
            angles.append(p % 4)
        need(sorted(destinations) == list(range(DIM)), 'monomial is a bijection')
        if adjoint:
            inverse, phases = [0] * DIM, [0] * DIM
            for x, y in enumerate(destinations):
                inverse[y], phases[y] = x, -angles[x] % 4
            destinations, angles = inverse, phases
        out = np.array(destinations), np.array(angles)
        self.tables[key] = out
        return out

    def mono(self, value, record, adjoint=False):
        if value is None:
            return None
        dest, phase = self.table(record, adjoint)
        a, e = value
        result = np.empty_like(a)
        for p in range(4):
            source = np.flatnonzero(phase == p)
            v = a[source]
            if p == 0:
                result[dest[source]] = v
            elif p == 1:
                result[dest[source], :, 0] = -v[:, :, 1]
                result[dest[source], :, 1] = v[:, :, 0]
            elif p == 2:
                result[dest[source]] = -v
            else:
                result[dest[source], :, 0] = v[:, :, 1]
                result[dest[source], :, 1] = -v[:, :, 0]
        return result, e

    def line(self, value, axis):
        if value is None:
            return None
        a, e = value
        guard(a)
        low = np.array([x for x in range(DIM) if not x & axis])
        high = low ^ axis
        total, difference = a[low] + a[high], a[low] - a[high]
        result = np.empty_like(a)
        result[low, :, 0] = total[:, :, 0] - difference[:, :, 1]
        result[low, :, 1] = total[:, :, 1] + difference[:, :, 0]
        result[high, :, 0] = total[:, :, 0] + difference[:, :, 1]
        result[high, :, 1] = total[:, :, 1] - difference[:, :, 0]
        return reduced(result, e + 1)

    def child(self, value, record):
        rank = record['rank']
        need(type(rank) is int and 0 <= rank <= 9, 'child rank')
        if rank:
            value = self.mono(value, record['right'])
            for j in range(rank):
                value = self.line(value, 1 << j)
        else:
            need(record['right'] is None, 'rank-zero child')
        return self.mono(value, record['left'])


def register(token):
    need(isinstance(token, list) and len(token) == 2, 'register syntax')
    role, index = token
    need(role in ('L', 'C') and type(index) is int
         and 0 <= index < (704 if role == 'L' else 8), 'register range')
    return index + (704 if role == 'C' else 0)


def equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return left[1] == right[1] and np.array_equal(left[0], right[0])


def initial_columns(columns):
    a = np.zeros((DIM, len(columns), 2), dtype=np.int64)
    for j, x in enumerate(columns):
        a[x, j, 0] = 1
    return a, 0


def execute(events, recipes, residuals, columns, mutation=None):
    engine = Engine()
    children = {(r['old'], r['new']): r for r in recipes['children']}
    adapters = {(r['role'], r['port']): r for r in recipes['endpoint_adapters']}
    state = [None] * 712
    state[HELPER] = initial_columns(columns)
    nonzero_boundaries, mutation_done = [], False
    active = Counter()
    touched, child_keys = set(), set()
    for at, event in enumerate(events):
        tag = event[0]
        if tag == 'phase':
            continue
        if tag == 'scalar':
            gate = event[1]
            op, target = gate[0], register(gate[1])
            if state[target] is not None or (op in ('add', 'copy') and state[register(gate[2])] is not None):
                active['scalar_' + op] += 1
                touched.add(target)
            if op == 'clear':
                need(target >= 704, 'CLEAR only owned fields')
                state[target] = None
            elif op == 'copy':
                need(target >= 704, 'COPY only owned fields')
                state[target] = state[register(gate[2])]
            elif op == 'scale':
                state[target] = scale(state[target], Fraction(gate[2], gate[3]))
            else:
                need(op == 'add', 'known scalar operation')
                source = register(gate[2])
                if mutation == 'omit_add' and source == HELPER and state[source] is not None and not mutation_done:
                    mutation_done = True
                else:
                    state[target] = plus(state[target], scale(state[source], Fraction(gate[3], gate[4])))
        elif tag in ('climb', 'center'):
            if tag == 'climb':
                _, _, target, old, new, rank = event
            else:
                _, target, old, new, rank = event
                target += 704
            rec = children[old, new]
            need(rec['rank'] == rank, 'event/recipe rank')
            if state[target] is not None:
                active[tag] += 1
                child_keys.add((old, new))
                touched.add(target)
            if mutation == 'omit_child' and target == HELPER and state[target] is not None and not mutation_done:
                mutation_done = True
            else:
                state[target] = engine.child(state[target], rec)
        elif tag == 'adapter':
            _, direction, target, _ = event
            if state[target] is not None:
                active['adapter'] += 1
                touched.add(target)
            if direction == 'old_to_new':
                if target < 112:
                    state[target] = engine.mono(state[target], adapters['source', target]['left'])
            else:
                need(direction == 'new_to_old', 'known adapter direction')
                if 112 <= target < 224:
                    state[target] = engine.mono(state[target], adapters['target', target - 112]['left'], True)
        else:
            need(tag == 'residual_adapter', 'known event')
            direction, target, old = event[1:4]
            new = event[4] if len(event) == 5 else 1
            key = str(old) if new == 1 else f'{old}:{new}'
            rec = residuals['recipes'][key]
            need(direction in ('pre', 'post'), 'known wrapper direction')
            if state[target] is not None:
                active['residual_' + direction] += 1
                touched.add(target)
            if target == HELPER and state[target] is not None:
                nonzero_boundaries.append([at, direction, old, new])
            if mutation == 'omit_pre' and target == HELPER and direction == 'pre':
                mutation_done = True
            else:
                part = 'right' if direction == 'pre' else 'left'
                adjoint = not (mutation == 'forward_post' and target == HELPER and direction == 'post')
                if not adjoint:
                    mutation_done = True
                state[target] = engine.mono(state[target], rec[part], adjoint)
    need(mutation is None or mutation_done, 'mutation exercised')
    return state, nonzero_boundaries, {'active_events': dict(active),
                                      'distinct_active_children': len(child_keys),
                                      'touched_fields': sorted(touched)}


def expected_columns(columns, rank=7):
    # Closed-form tensor factors, independent of recipes, tables and line executor.
    numerator = np.zeros((DIM, len(columns), 2), dtype=np.int64)
    for j, x in enumerate(columns):
        for delta in range(1 << rank):
            real, imag = 1, 0
            for bit in range(rank):
                sign = -1 if delta & (1 << bit) else 1
                real, imag = real - sign * imag, imag + sign * real
            numerator[x ^ delta, j] = real, imag
    return reduced(numerator, rank)


def check_endpoint(state, expected):
    need(equal(state[HELPER], expected), 'residual endpoint equals C_[7]')
    need(all(v is None for i, v in enumerate(state) if i != HELPER),
         'zero leakage into data, other helpers and private fields')


def failure_controls(programs, recipes, residuals):
    """Each mutation has an exact differing coefficient at input column 79."""
    records = []
    expected = expected_columns([79])
    for side in ('forward', 'backward'):
        for mutation in ('omit_pre', 'forward_post', 'omit_child', 'omit_add'):
            state, _, _ = execute(programs[side], recipes, residuals, [79], mutation)
            helper_matches = equal(state[HELPER], expected)
            fields = [i for i, value in enumerate(state)
                      if not equal(value, expected if i == HELPER else None)]
            need(fields, 'semantic mutation was detected: ' + mutation)
            field = fields[0]
            got = state[field]
            want = expected if field == HELPER else None
            exponent = max(got[1] if got else 0, want[1] if want else 0)
            if got:
                guard(got[0], 1 << (exponent - got[1]))
            if want:
                guard(want[0], 1 << (exponent - want[1]))
            a = got[0] << (exponent - got[1]) if got else np.zeros((DIM, 1, 2), dtype=np.int64)
            b = want[0] << (exponent - want[1]) if want else np.zeros((DIM, 1, 2), dtype=np.int64)
            address = int(np.flatnonzero(np.any(a[:, 0] != b[:, 0], axis=1))[0])
            records.append({'orientation': side, 'mutation': mutation, 'input_column': 79,
                            'helper_endpoint_still_exact': helper_matches,
                            'differing_fields': fields, 'witness_field': field, 'output_address': address,
                            'actual_gaussian_numerator': a[address, 0].tolist(),
                            'expected_gaussian_numerator': b[address, 0].tolist(),
                            'common_denominator': 2**exponent})
    return records


def run(columns, orientations, batch=16):
    word, recipes, programs, residuals = load()
    eligible = [i for i, old in enumerate(word['start']) if old and word['final'][i] != 1]
    need(eligible[79 % len(eligible)] == HELPER and (word['start'][HELPER], word['final'][HELPER]) == (3, 998),
         'assignment selection')
    reports = []
    for side in orientations:
        events = programs[side]
        need(len(events) == 24767, 'complete selected orientation')
        digest = hashlib.sha256()
        first_activity = None
        for offset in range(0, len(columns), batch):
            chosen = columns[offset:offset + batch]
            state, cuts, activity = execute(events, recipes, residuals, chosen)
            check_endpoint(state, expected_columns(chosen))
            need(len(cuts) == 2 and {x[1] for x in cuts} == {'pre', 'post'}, 'both nonzero wrapper boundaries')
            if first_activity is None:
                first_activity = activity
            need(activity == first_activity, 'identical active-event coverage across column batches')
            for j, column in enumerate(chosen):
                digest.update(json.dumps([column, state[HELPER][1], state[HELPER][0][:, j, :].tolist()],
                                         separators=(',', ':')).encode())
        reports.append({'orientation': side, 'events': len(events), 'event_counts': dict(Counter(e[0] for e in events)),
                        'payload_columns': len(columns), 'output_sha256': digest.hexdigest(),
                        'nonzero_wrapper_boundaries': cuts, 'other_fields_zero': 711,
                        'activity': activity})
    return {'schema_version': 1, 'status': 'PASS_SELECTED_DIRTY_HELPER_EXECUTION',
            'helper': HELPER, 'entrance_frame': 3, 'terminal_frame': 998, 'rank': 7,
            'dimension': DIM, 'orientations': reports,
            'scope': 'finite serialized local executor; no all-size, machine, numeric or supplier admission'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--columns', type=int, default=512, help='partial basis domain; partial runs never certify receipt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--orientation', choices=['forward', 'backward', 'both'], default='both')
    ap.add_argument('--write', type=Path)
    args = ap.parse_args()
    need(1 <= args.columns <= 512 and 1 <= args.batch <= 64, 'bounded domain')
    sides = ['forward', 'backward'] if args.orientation == 'both' else [args.orientation]
    full = args.columns == 512 and len(sides) == 2
    need(not args.write or full, 'partial replay cannot publish full receipt')
    result = run(list(range(args.columns)), sides, args.batch)
    if not full:
        result['status'] = 'PASS_PARTIAL_SELECTED_DIRTY_HELPER_EXECUTION'
    else:
        _, recipes, programs, residuals = load()
        result['failure_controls'] = failure_controls(programs, recipes, residuals)
    if args.write:
        with args.write.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    elif full:
        need(json.dumps(strict_json((HERE / 'RESULT.json').read_bytes()), sort_keys=True) ==
             json.dumps(result, sort_keys=True), 'result receipt mismatch')
    print(result['status'], flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
