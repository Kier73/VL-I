# TIME-008 — Environment-derived rational processor dimension

**Research state:** bounded implementation on a signed, controlled numerical
publisher. This is not an unknown physical process law, continuous neural
intelligence, a general sequence-prediction guarantee, or a proof that all
possible architectures can be discovered from finite evidence.

## Question and predecessor reuse

TIME-007's four classes preselect the processor dimensions 1, 2, or 3.
TIME-008 removes that family selector: a signed univariate sequence chooses
its smallest fully identifiable constant-coefficient rational recurrence
under an explicit order budget. The coefficient count becomes the processor
dimension, including a tested four-register architecture absent from the
previous fixed family list.

The implementation reuses **unchanged** TIME-005's Ed25519 signed,
publisher-head-pinned journal, TIME-001's independently reconstructable
observation Ground, and TIME-007's exact rational matrix navigation,
source-level processor compiler, same-language self-interpreter, and separate
iterative continuation verifier. The only changes to TIME-007 are admitting
a bounded dimension four and permitting the independently admitted TIME-008
schema at the source-reflection gate.

Cross-repository design precedents:

- [VLD TRACE-002](https://github.com/Kier73/Virtual-Layer-Dynamics/blob/main/docs/TRACE_002_ACTIVE_OBSERVER_INQUIRY.md)
  distinguishes rank-increasing observations, held-out challenges, and a
  separately audited inquiry path. Its finite-field *fixed-order* solver is
  **not** misreported as a rational unknown-order dependency used here.
- [Navigable-Compute NCR-006/007](https://github.com/Kier73/Navigable-Compute/blob/main/docs/NCR-007.md)
  demonstrates provably observer-sufficient processors for declared source
  algebraic families; it does **not** certify these unknown external dynamics.
- [E-CPU IPS-008](https://github.com/Kier73/E-CPU/blob/main/docs/RECEIVER_HANDOFF_BARRIER_IPS008.md)
  is an existing authenticated receiver handoff barrier. This TIME-008 change
  preserves mathematical/source Ground succession but does not yet activate
  it across that physical authority protocol.

## Exact discovery

Given canonical rational samples `y_0,...,y_{N-1}` at a uniform sampled
time step, the signer is independently verified against a caller-pinned
publisher head, source identity, and expected count. Every observation is
matched to the complete independently reconstructed TIME-001 snapshot.

A caller declares distinct, chronological **training**, **validation** and
**untouched test** intervals, each of the last two containing at least two
source events. The producer considers increasing orders `k=1,...,4`,
limited by half the training count. For each order, it constructs the
rational Hankel regression:

```
H[i,j] = y[k+i-j-1]             i,j in [0,k)
b[i]   = y[k+i]
H c    = b
```

The first *nonsingular*, training-consistent candidate is selected via
exact Gaussian elimination. Singular systems are **not** assigned arbitrary
coefficients. The model is then propagated using its own predicted values,
not injected withheld true values, and must exactly agree with the training,
validation and untouched test intervals. It refuses on the first mismatch,
including with a genuine valid signature. A signed value is evidence that
the publisher asserted it—not proof of physical truth.

The second verifier independently reconstructs the exact candidate using
the permutation formula for determinants and Cramer's rule. It independently
rechecks the full signed source ledger and snapshot before comparing the
complete produced Ground, so a modified self-consistent digest cannot
authorize a changed law.

The companion construction uses

```
s[k-1] = (y[k-1], ..., y[0])
A[0,:] = (c_0, ..., c_(k-1))
A[i,j] = 1 if j=i-1 else 0      i=1..k-1
y[t]   = (A^(t-k+1) s[k-1])[0]
```

No trajectory table is generated. A request selects a canonical rational
time coordinate and either `current` or `lag:i` within the source-qualified
observation alphabet. The generated source task and bounded self-interpreter
use the same previously trusted exact arithmetic/tuple/integer primitive
seed, and their result is checked by an independent continuation VM.

## Identity and revision

The Ground retains exact source event digests, source public key fingerprint,
publisher-pinned signed head, externally pinned observation Ground head and
identity, chronological sample splits, inferred dimension, source coefficients,
generated matrix, initial coordinate, and an immutable predecessor digest.

For an opt-in, one-successor revision, the new Ground must start **strictly
after** the previous source prefix. The verifier checks the old packet chain
from genesis to the old externally attested signed head, rechecks the old
recurrence with independent Cramer arithmetic, and demands its exact immutable
event prefix and old Ground head. The new epoch receives its own independent
train/validation/test qualification. No claim of old/new **future** equivalence
is inherited. This prototype intentionally refuses multi-generation
succession unless the intermediate proof chain is implemented.

Future mathematical projections require `allow_projection=True`. They are
labeled `UNVERIFIED_FUTURE_MODEL_PROJECTION`, never authenticated physical
events. The signed stream can contradict a model; it does not grant a
self-asserted processor the authority to perform external effects.

## Adversarial coverage and reproducing

```sh
python -m pip install -e .
python -m pip install 'cryptography>=46,<47'
python -m unittest discover -s tests -p test_discovered_state_space.py -v
python -m unittest discover -s tests -p test_intention_state_space.py -v
```

The TIME-008 CI matrix runs Python 3.11 and 3.13 on Linux and Windows, and
also checks the old signed-source recovery tests. Cases include: genuinely
inferred orders 2, 3 and 4; multiple intent outputs; time navigation without
trajectory replay; depth-two reflected task/VM checks; signed-source and
journal-head tampering; an authenticated validation counterexample;
singular/insufficient data; model-future opt-in; source-prefix-preserving
regime revision; and forged result/proof values.

**Next experimental seams:** (1) a genuine independent external instrument
or service (rather than signed synthetic numeric claims), (2) active rank-aware
probe selection and heterogeneous observers with independently qualified
costs, (3) source-derived nonlinear/Koopman or partial-order grammar rather
than the still declared linear-recurrence metaclass, (4) authoritative
E-CPU IPS-008 remote cutover, and (5) measured cold/warm total cost versus
direct native execution. For single-runner work, do not promote the word
"continuity" to physically uninterrupted observation.
