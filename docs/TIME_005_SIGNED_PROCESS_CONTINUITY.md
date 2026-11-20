# TIME-005A — Signed observation continuity across actual process death

**Status: reproducible, small-scope, OS-process crash and authentic-source
admission experiment.** This exercise uses real physical clock capture and
two ungraceful process terminations, not continuous sensor coverage or an
independently established quorum of remote machines.

## Motivation

TIME-001/002 established rational-time observation provenance, one live
observer and checkpoint-sink seams. TIME-003 generated changing mathematical
languages, and TIME-004 used actual NCR-002 source-relative, task-generated
self-interpreters to navigate admitted observed positions.

What they did NOT establish was that a *different* program could determine
whether a supposed source observation truly came from the source, or that
the observation state could survive an actual OS-process death during the
publish/commit transition.

This experiment addresses that missing property directly.

## 1. Separate source publication from the observer

The publisher runs in its own parent process and retains an **Ed25519
private signing key**. It signs a fully specified source envelope: identity,
sequence number, logical rational sample time, source-reported value,
publisher-measured wall-clock nanoseconds and publisher-measured monotonic
nanoseconds. The test publisher uses a controlled mathematical value sequence.
The capture clocks, however, are real hardware/OS clock readings.

The worker subprocess is given **only the Ed25519 public key**, the canonical
JSONL journal and an independently supplied trusted journal head and count.
The worker never receives the signing private key.

Before admitting an observation, it independently:

- Checks the Ed25519 signature over canonical source bytes and identity.
- Verifies strict source sequence, nondecreasing measured monotonic clock and
  logical environmental coordinate.
- Reconstructs the source journal hash chain from genesis and refuses
  mismatch with the externally supplied pinned head/count.
- Checks every persisted TIME-001 event against the *exact corresponding
  signed source event*, then independently verifies the TIME-001 causal
  chain before restarting the observer.

A digest alone is NOT source authenticity. An outer digest recomputed by an
attacker is rejected if even one signed value changes. Source time and value
truth is still **attested by the publisher**, not independently proven
against the physical environment.

## 2. Crash boundaries and recovery

The worker accepts a signed source event into its original TIME-001 Ground
and persists the new checkpoint by writing a temporary file, fsyncing it,
atomically renaming over the old checkpoint, and syncing the parent
directory where supported (POSIX). Windows atomic replacement and file fsync
are exercised; different storage/hardware conditions may provide different
power-loss guarantees.

The experiment invokes the process-hard-exit primitive os._exit at two
adversarial points; there is no finally block, destructor or normal Python
cleanup. A third control pauses a **separate child OS process immediately
after its fsynced checkpoint** and has the external parent forcibly terminate
it with subprocess.Popen.kill(). The parent reads the persisted checkpoint,
then verifies the replacement process accepts exactly the one missing source
event, never replays the already committed effect.

- **Before commit:** the worker dies with source index 2 signed externally
  but missing from its durable state. The new worker verifies the complete
  source journal, restores two already-committed observations, and commits
  only the missing source index 2.
- **After commit, before acknowledgment:** the worker dies after committing
  source index 5 but before returning a decision. Another process verifies
  the source history and checkpoint and admits only source index 6. The
  previously committed source event is NEVER replayed as an additional
  semantic observation.

On the same host there are distinct OS process IDs and different declared
observer-host labels; this does NOT establish travel to another physical
machine or the availability of a networked quorum.

The TIME-003 source-qualified interpreter changes from *affine* to
*quadratic* with the same externally signed causal source history; the
migration receipt proves exactly that source prefix, not future behavior
equivalence.

## 3. Concrete experiment and reporting

A controlled source publisher emits 1,3,5,100,101,104,109 while recording
physically observed capture times. The experiment measures source
inter-sample intervals using publisher monotonic nanosecond timestamps. Consecutive
captures may have *equal* clock readings on coarse-resolution OS timers; such
a zero-duration reading means unresolved precision, never an assertion that no
time elapsed. All
these gaps are **unobserved intervals between discrete samples** and are
reported as such. The rational 0,1,2,... source coordinate is a logical
sample index, NOT the physical wall-clock time of the source.

The example's single actual publisher process survives the first two worker
failures. The third worker gets the exact signed journal, restores the
observation identity and activates a newly qualified language. The test
checks the exact number of new observations after each crash, rejecting
an invented continuous timeline.

## 4. Reproduction

~~~bash
python -m pip install -e .
python -m pip install 'cryptography>=46,<47'
python -m unittest discover -s tests -p test_authenticated_temporal_source.py -v
python -m examples.authenticated_temporal_source_demo
~~~

CI runs on Ubuntu/Windows with Python 3.11 and 3.14, alongside TIME-002
live-observer and TIME-003 interpreter migration regressions.

Adversarial tests include signature corruption even after recomputing
the journal chain, publisher-head rollback, truncated/torn journal, source
sequence replay/reordering, wrong signer/identity, checkpoint rehashing after
altering an observed value, and injecting an extra fictitious source event.

## 5. Trust model and what is NOT yet demonstrated

- The publisher's source private key is a test-held secret outside worker
  subprocesses. An adversary controlling the publisher can sign a fabricated
  value; no physical truth guarantee is implied.
- The expected source journal head and count come from an *independent
  authoritative parent process argument*, rather than from the potentially
  compromised worker checkpoint or same journal. If that authority is
  malicious or rollback-prone, this layer cannot detect it.
- The journal writer is **single-authority only**. This is not an E-CPU
  ReliableChannel or VLD QuorumLineageCoordinator. Competing workers,
  adversarial network partitions, lease loss, storage power failure, secure
  remote identity, and exactly-once effectful external actions have not yet
  been demonstrated.
- A source signed monotonic clock certifies what the publisher reported,
  not a globally synchronized physical time. The clock values are
  comparable only within the same running publisher process and clock epoch.
- The observer did not measure the real world between samples, and no
  uninterrupted host execution across a crash is claimed. "Continuity" here
  means a **verifiable source-history succession** across lifetimes, not
  an uninterrupted subjective process.
- Numeric law execution **inside** NCR-002's generated semantic algebra,
  beyond TIME-004's observed-index navigation, remains a separate, active
  target; the source authenticator does not claim to solve it.
- Neural adaptation, new architectures outside supplied grammar families,
  and learning a model of physically evolving systems still require
  independent experiments.

## 6. Follow-on path

The next continuation is to bind the **authenticated signed receipts**
and publisher-head authority to VLD's true quorum lineage and E-CPU's
authenticated ReliableChannel, with fenced concurrent observer incarnations
and real network-induced delays. Then integrate directly sourced numerical
law semantics into a constructed NCR interpreter, measure the cost and test
whether new representation operators actually improve held-out task
performance.

The primary question is testable rather than metaphysical: **Can a later
observer prove which witnessed changes its predecessor encountered and
which unobserved intervals it cannot reconstruct, even if its previous
physical process was killed during the commit protocol?**
