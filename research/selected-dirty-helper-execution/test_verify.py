"""Semantic controls for the finite dirty-helper executor; Apache-2.0."""
import copy
from fractions import Fraction as Q
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('current_dirty_executor', HERE / 'verify.py')
V = importlib.util.module_from_spec(spec)
# Compile the current authenticated source, without reading import bytecode.
exec(compile((HERE / 'verify.py').read_bytes(), str(HERE / 'verify.py'), 'exec'), V.__dict__)


def fraction_vector(value):
    if value is None:
        return [(Q(0), Q(0))] * V.DIM
    return [(Q(int(a), 2**value[1]), Q(int(b), 2**value[1])) for a, b in value[0][:, 0, :]]


class Behavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word, cls.recipes, cls.programs, cls.residuals = V.load()

    def test_literal_word_mutations_reject(self):
        self.checked_mutants = 0
        for side in ('forward', 'backward'):
            for mutation in ('omit_pre', 'forward_post', 'omit_child', 'omit_add'):
                with self.subTest(side=side, mutation=mutation):
                    state, _, _ = V.execute(self.programs[side], self.recipes, self.residuals,
                                             [0, 79, 511], mutation)
                    with self.assertRaises(ValueError):
                        V.check_endpoint(state, V.expected_columns([0, 79, 511]))
                    self.checked_mutants += 1
        self.assertEqual(self.checked_mutants, 8)

    def test_basis_columns_and_two_spectators(self):
        cols = [0, 79, 128, 256, 511]
        for side in ('forward', 'backward'):
            state, _, _ = V.execute(self.programs[side], self.recipes, self.residuals, cols)
            V.check_endpoint(state, V.expected_columns(cols))
            # C_[7] cannot change either of the two high bits.
            a = state[V.HELPER][0]
            for j, x in enumerate(cols):
                for y in range(512):
                    if (x >> 7) != (y >> 7):
                        self.assertTrue(np.all(a[y, j] == 0))
            re, im = a[:, :, 0], a[:, :, 1]
            # ||column||^2 == 1, exactly, including imaginary components.
            self.assertTrue(np.all(np.sum(re * re + im * im, axis=0) == 2**(2*state[V.HELPER][1])))

    def test_batch_partition_does_not_change_operator(self):
        columns = list(range(17))
        self.assertEqual(V.run(columns, ['forward', 'backward'], 1),
                         V.run(columns, ['forward', 'backward'], 16))

    def test_gaussian_line_against_fraction_formula(self):
        a = np.zeros((512, 1, 2), dtype=np.int64)
        for x in range(512):
            a[x, 0] = (x * 79 % 41) - 20, (x * 17 % 37) - 18
        value = V.reduced(a, 3)
        original = fraction_vector(value)
        engine = V.Engine()
        for j in range(9):
            got = fraction_vector(engine.line(value, 1 << j))
            want = []
            for x, (ar, ai) in enumerate(original):
                br, bi = original[x ^ (1 << j)]
                want.append(((ar-ai+br+bi)/2, (ai+ar+bi-br)/2))
            self.assertEqual(got, want)

    def test_monomial_phase_permutation_and_adjoint(self):
        rec = {'matrix_rows': [2, 1] + [1 << j for j in range(2, 9)],
               'quadratic_rows': [2] + [0] * 8,
               'linear_Z4': [1, 3] + [0] * 7, 'offset': 5, 'phase': 1}
        engine = V.Engine()
        value = V.initial_columns(list(range(512)))
        got = engine.mono(value, rec)
        for x in range(512):
            y = (x & ~3) | ((x & 1) << 1) | ((x & 2) >> 1)
            y ^= 5
            phase = (1 + (x & 1) + 3*((x >> 1) & 1) + 2*((x & 3) == 3)) % 4
            pair = ((1, 0), (0, 1), (-1, 0), (0, -1))[phase]
            self.assertEqual(tuple(got[0][y, x]), pair)
        self.assertTrue(V.equal(engine.mono(got, rec, True), value))
        bad = copy.deepcopy(rec)
        bad['matrix_rows'][0] = bad['matrix_rows'][1]
        with self.assertRaisesRegex(ValueError, 'bijection'):
            engine.table(bad)

    def test_integer_arithmetic_refuses_overflow_before_operations(self):
        a = np.array([[[2**59, -2**59]]], dtype=np.int64)
        value = V.reduced(a, 3)
        self.assertEqual(fraction_vector(value)[0], (Q(2**56), Q(-2**56)))
        with self.assertRaisesRegex(ValueError, 'int64'):
            V.scale((a, 0), 2)
        with self.assertRaisesRegex(ValueError, 'int64'):
            V.plus((a, 0), (a, 2))
        with self.assertRaisesRegex(ValueError, 'dyadic'):
            V.scale(value, Q(1, 3))

    def test_zero_leakage_check_is_independent_of_residual_check(self):
        state = [None] * 712
        state[V.HELPER] = V.expected_columns([79])
        V.check_endpoint(state, V.expected_columns([79]))
        state[192] = V.initial_columns([79])
        with self.assertRaisesRegex(ValueError, 'zero leakage'):
            V.check_endpoint(state, V.expected_columns([79]))

    def test_source_pin_and_strict_receipt_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = root / V.ARCHIVE
            name.parent.mkdir(parents=True)
            name.write_bytes(b'incorrect scientific source')
            with self.assertRaisesRegex(ValueError, 'source pin'):
                V.load(root)
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                V.strict_json(raw)

    def test_partial_output_cannot_publish_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'receipt.json'
            proc = subprocess.run([sys.executable, '-I', '-O', '-B', str(HERE/'verify.py'),
                                   '--columns', '1', '--write', str(target)], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('partial replay cannot publish', proc.stderr)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
