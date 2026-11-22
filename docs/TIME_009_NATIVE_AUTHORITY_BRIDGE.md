# TIME-009 — A signed mathematical Ground meets native E-CPU effect authority

**Status:** bounded cross-repository implementation and failure experiment.
TIME-009 does **not** claim the rational interpreter itself is E-CPU's original
IPS-004 executing stage, an atomic cutover of two independently owned
systems, physically attested remote hosts, or universally correct physical
measurements. It directly tests what happens when two independently
qualified computational histories are required before a real native effect.

## Reused original mechanisms

1. **VL-I TIME-005 / TIME-008:** Ed25519-signed source observations,
   publisher-external head pin, replay-verified TIME-001 checkpoint, a
   discovered exact rational recurrence, chronological source challenge,
   and a separately implemented Cramer-rule mathematical verifier.
2. **E-CPU IPS-005 / IPS-007 / IPS-008:** actual native-generated arithmetic
   stage, majority-supported current publication, exact typed
   `PublishedGroundEffect`, receiver-owned SQLite effect register,
   receiver-durable PREPARE barrier, mutually authenticated TLS service,
   and original physical process hard-kill/recovery fixtures.
3. **No replacement Hypervisor, consensus, authority or transport.** The
   TIME-009 bridge calls the original native E-CPU public service and
   receiver methods. GitHub CI pins
   [E-CPU `9064559d`](https://github.com/Kier73/E-CPU/commit/9064559d6e9b621034953b7135fc991a436d7746)
   and runs its unchanged IPS-008 test fixtures and original regression.

The authority condition is a conjunction:

```
SIGNED SOURCE LEDGER + PINNED SOURCE GROUND + OBSERVED VALUE
                         AND
PINNED CURRENT E-CPU QUORUM ROOT + RECEIVER INSTALLED ROOT
                         AND
MATCHING ORIGINAL E-CPU STAGE VALUE + ATOMIC RECEIVER APPLY
                         =>
           NATIVE EFFECT RECEIPT (not a new capability)
```

The caller must provide an **externally selected pair**:
`expected_space_digest` and `expected_native_root`. A signed source
cannot choose which E-CPU quorum root to activate. Likewise, a self-hashed
matching Ground never grants a resource right. Every attempted effect
revalidates the *entire* original signed source, independently reconstructs
TIME-008 mathematics, requires a **previously observed** rational coordinate,
checks that it is an exact integer matching the independent native stage,
and asks the original native E-CPU receiver to commit. **Future model
projections are categorically ineligible** for this effect path.

The bridge does not first pretend a snapshot or a local read grants
authority. The E-CPU receiver's original `apply()` performs final current
epoch/host/plan/effect checks under its SQLite writer transaction, and the
service authenticates the caller by its existing mTLS identity. If the
receiver PREPARE barrier is acknowledged, all old effects fail closed even
if the mathematical result remains correct.

## Live experiment: two genuinely different mathematical descriptions

The native E-CPU fixture has the original two-stage affine plan. Its first
publication returns exact integer **12**; its second returns **16**.
VL-I supplies a separately signed controlled source:

- Old source events `y(t)=10+t`, signed for t=0..11. TIME-008 discovers a
  minimal order-two recurrence `y(t)=2*y(t-1)-y(t-2)` without a family label.
  Its authentic historical coordinate t=2 is **12**, matching native
  first-stage `PublishedGroundEffect`. The original E-CPU resource accepts
  one effect, and a second identical request returns the same durable
  receipt without repeating it.
- The native service acknowledges PREPARE for a **different qualified
  computational host**. The effect receiver is hard-killed and reopened
  in a different process with the same mTLS/SQLite identity. Its durable
  barrier still rejects the old computation, despite the old signed
  Ground being mathematically sound.
- VL-I then admits another twelve independently signed values, defining
  a new regime whose recurrence `y(t)=y(t-1)+y(t-3)` is dimension **three**
  with new held-out observations. TIME-008 independently checks the
  original source prefix and immutable old Ground, and constructs a
  successor Ground with a different identity. The new epoch's historical
  coordinate t=14 is **16**, but it **still cannot write** until the E-CPU
  receiver witnesses the correct quorum-supported assignment.
- The original E-CPU coordinator opens preserved authority journals,
  seals its exact successor assignment, and publishes its next native
  stage through an independently qualified host. The original E-CPU
  receiver verifies and activates that current root, then accepts the
  observed integer 16 under the **new pinned TIME-008 Ground**. Previous
  epoch and new epoch are distinct signed-source and native-authority
  identities; only native E-CPU can commit the actual receiver effect.

Adversarial controls: mismatched trusted Ground and E-CPU root pins;
self-rehashed falsified law; legitimate source observation that disagrees
with native stage output; any future mathematical projection; a prepared
handoff; a hard-killed/restarted receiver; old quorum publication after
rehome; and a TLS client signed by the same CA but with wrong subject.

The source values are controlled signed mathematical claims. They are
not independent physical truth readings. The two native stages compute
their **own original E-CPU modular-affine operator**. TIME-009 verifies
a declared observer-value match, **not that E-CPU computed the new
rational recurrence**. Importantly, that negative distinction prevents a
false inference from identical output bytes.

## Reproduction and proof gates

The cross-repository CI matrix uses Python 3.11 and 3.13 on Linux and Windows:

```sh
# Set VLI_ECPU_TEST_ROOT to a pinned checkout of E-CPU/tests
python -m pip install -e .
python -m pip install -e <pinned-E-CPU-directory>
python -m pip install 'cryptography>=46,<47' 'pytest>=8'
python -m unittest discover -s tests -p test_temporal_native_effect_bridge.py -v
python -m unittest discover -s tests -p test_discovered_state_space.py -v
python -m pytest -q <pinned-E-CPU-directory>/tests/test_effect_receiver_service.py
```

The E-CPU source checkout is pinned to one immutable commit for repeatability;
no new service can silently replace native authority.

## Development still to solve

The next frontier is **semantic authority binding** rather than equal-output
observation binding. An admitted TIME-008 source Ground should become an
actual receiver-authorized native pure operator/plan whose typed execution
receipt includes its signed source fingerprint, derived coefficients and
full independent oracle. A verified Ground revision would need a native
quorum transition that atomically retires the previous *computational
interpretation* before admitting the new interpretation on a new host,
without pretending the two independent journals share an atomic clock.

Further experiments should measure acquired source bytes, proof effort,
remote network traffic and native computation cost, and use a genuine
external instrument or service with an independently authenticated
measurement semantics. None of these tasks requires replacing E-CPU's
existing Hypervisor, receiver handoff barrier or native fabric.
