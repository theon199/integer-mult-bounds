# Execute one nonzero selected residual helper

**Independent finite verification; no new supplier or κ.** In both complete
selected #1094 local words, an arbitrary 512-dimensional complex input in
helper **557**, with every other field initially zero, ends as

```text
helper 557 = (C_[7] ⊗ I_4) input
all other 711 fields = 0.
```

Here `C_[7]` is the product of the positive Gaussian lines on coordinate bits
0 through 6; bits 7 and 8 are spectators. The helper enters at frame 3 and
ends at frame 998. Its physical residual endpoint is a rank-seven transform,
not the unchanged input. No defect in the accepted construction was found.

The executable checks all 512 basis inputs against a closed-form Gaussian
integer kernel. Complex linearity extends this checked operator block to
arbitrary complex payloads in this helper. This is one block of the finite
local executor, not the full multiplication theorem or an all-input check of
the complete 712-field operator.

## What this adds

Sagar Thakrar's [#1781 recipe audit](../frontier1094-recipes-exact-seed76/README.md)
checks the child and data-adapter records. His
[#2029 residual audit](../frontier1094-residual-wrappers-seed62/README.md)
checks all 36 residual records on every column, and explicitly requests a
literal helper trace through its entrance wrapper, climbs and exit wrapper.
His [#1964 extended-label audit](../selected1094-extended-labels-seed65/README.md)
separately checks the whole-word conjugation hypotheses. These are predecessor
results, not new identities here.

This package supplies the requested **complete emitted-word execution** for
one nonzero residual input block, including the scalar interactions with other
fields. No predecessor executable code, frame constructor, label replay,
tableau or packed evaluator is imported. The NumPy integer-array interpreter
and its event dispatcher are new. The expected endpoint is generated directly
from the seven factors `(1+i)/2` and `(1-i)/2`, without reading the wrapper
record or using the executor's line routine.

Recent QQ16 #3017/#3067/#3101 witnesses concern different producer/numerical
claims and have zero payloads at these residual-wrapper boundaries. They are
credited context, not assertions of new numerical coverage here. Both wrapper
boundaries in this experiment carry nonzero payloads.

## Exact coverage and controls

Each orientation contains 24,767 events: 19,693 scalar, 3,486 climb, eight
centre, 1,090 data-adapter, 480 residual-adapter and ten phase events.
All are dispatched, with exact zero arithmetic skipped where applicable.
This input slice activates only the following small part of the program:

| Orientation | Active child records | Active ADDs | Touched ordinary fields |
| --- | ---: | ---: | --- |
| forward | 6 | 7 | 192, 557 |
| backward | 9 | 9 | 5, 455, 557 |

The nonzero helper-557 wrapper events are at zero-based positions **7959**
(`pre`) and **24623** (`post`) in each orientation. All 512 columns pass;
each residual matrix has 65,536 nonzero complex entries and reduced common
denominator 16. Both high-coordinate spectators and every other field's zero
endpoint are included. This is not fresh coverage of every child recipe.

[RESULT.json](RESULT.json) records the exact inventory, activity, output hash
and eight deliberate faulty-executor controls. Per orientation, omitting the
entrance wrapper, using the forward exit monomial, skipping a live child,
and omitting a live ADD each gives an explicit differing coefficient at
input column 79. The omitted-ADD controls leave the helper endpoint correct
while leaking into other fields: forward fields 192/200; backward fields
4/5/68/69. Thus checking only the residual output would miss those faults.
These are mutations of this reference executor, not defects attributed to the
producer or its existing audit.

Nine behavioral groups additionally compare Gaussian lines with a separate
`Fraction` oracle, check monomial phases/permutations and adjoints, vary the
column batch partition, test the two spectators, reject integer overflow,
check leakage separately, and reject source/JSON and partial-publication errors.
The test command also runs under `-O`; acceptance does not use assertions.

## Arithmetic and source boundary

Each value is a Gaussian-integer array divided by a common power of two.
The executor applies right monomial, positive coordinate lines, left
monomial; entrance/exit residual wrappers use the appropriate adjoints.
The monomial phase reads each strict-upper quadratic coefficient once,
retaining global phase and the stored permutation direction. Scalar ADD,
SCALE, private COPY and private CLEAR follow the literal event order.

Every addition, multiplication and shift is bounded before execution;
the largest permitted sum of four operands is below `2^62`. Values outside
the stricter `2^60` guard are rejected. Common factors of two are removed
exactly. NumPy `int64` operations cannot silently wrap within these guards;
there is no floating-point arithmetic, packed digit encoding or tolerance.

Base: **`18aacb215fc3717b06be8ceb9fca62aeca0d3e99`**. The selected source PR1094
head is `0fc329d0bec93ca7cd288eaea1295172e8789810`; accepted κ remains
`836330098538459/500000000000000000`. The consumed reviewed capsule is
`research/community-frontier1094-audit/selected-replay.tar.gz`, SHA256
`f1da80610bf33c58527f18ad63b4350d0b864e1d4486141ee17ac9bed4d529d6`.
All four member hashes are in `verify.py`; members are selected by their full
`candidate/strict/global/base/providers/gauge44/` paths and must be unique
regular files. Same-basename `split-centres` members are not selected.
[SOURCES.json](SOURCES.json) binds the capsule's Git blob and provenance.
No frozen source, selected record, shared verifier or `upstream/` file changes.

## Reproduce

Python 3.12 and `numpy==2.3.5`, from the repository root:

```sh
python3 -Werror -I -B scripts/research_package.py --package research/selected-dirty-helper-execution
```

Expected: `PASS_SELECTED_DIRTY_HELPER_EXECUTION`, nine passing test groups,
then `PASS_REGISTERED_RESEARCH_PACKAGE`. The full scientific replay takes
about five seconds on the recorded Linux environment; timings are informational.
The launcher checks exact dependency versions. Direct `verify.py` and
`test_verify.py` also work in an export with the declared capsule. Default
replay is read-only; `--write NEW_PATH` creates an absent receipt and refuses
partial-domain publication. [VERIFY.json](VERIFY.json) registers all inputs.

## Open next questions

This verifies one input block of both serialized finite local words, with
exact arithmetic and direct native matrices. Other dirty/data inputs,
cross-call charts, whole-call hosting, recursive child implementation and
tariffs, physical movement/storage/tapes, finite precision, all-size compiler,
analytic transfer and the bit supplier remain outside this check. No Lean
build, full selected cold replay or new κ is claimed. Those expensive suites
are unnecessary for this isolated reference-executor contribution.

The next finite question is a distinct input block with a different wrapper
key or data endpoint. Start by deriving its complete expected output and
retaining nonzero wrapper and leakage-only controls; first inventory existing
coverage. The remaining 239 wrapped helpers are not checked here. The larger
question is refinement to an actual recursive/physical child call: specify
one invocation/return and charge its numerical, workspace and movement
contract. This interpreter does not supply that implementation.
[TASK.json](TASK.json) records those blockers, candidates, claim and overlap.

## Credit

Sricharan Vagvala (@theon199), with substantial OpenAI Codex assistance in
investigation, implementation, checks and writing. New material is Apache-2.0.
No independent human review or proof-kernel certification is claimed.

Credit Sagar Thakrar for #1781/#1964/#2029 and the explicit executor handoff;
DreamingOfClouds for Lemmas A/G and the residual construction; icekylinx,
jacklightChen and Chafik Boukhalfa for the Clifford/wrapper lineage; Jeremy
Kothe for the rank-seven Lean pairs and selected word lineage; Rohan Arun for
#1094, with Jeremy Kothe's #1087 packing and Hao Yan's controller. The selected
capsule and its notices retain their cumulative attribution. Tekalign,
Arham Tahir, Ross Richardson, Nick Petalas and the QQ16 team retain credit for
the separate native/numerical work used to distinguish this scope.

## Publication handoff

The verified implementation and receipt are public at fork commit
[`3481bc0eeab0e90c738752884a63eec881538cbf`](https://github.com/theon199/integer-mult-bounds/commit/3481bc0eeab0e90c738752884a63eec881538cbf).
GitHub denied PR creation through both CLI and connector with HTTP 403,
`Resource not accessible by integration`. No upstream PR or Discussion post
was created. The local claim is complete and released. A maintainer can use
the [fork comparison](https://github.com/CrocSwap/integer-mult-bounds/compare/main...theon199:sprint/seed79-independent-obligation?expand=1)
to submit the focused contribution. This section adds publication status only.
