"""TIME-005: a publisher outlives two actual observer process crashes."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import sleep

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import load_json
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage,certify_language_migration,
    verify_language_migration,
)
from vli.authenticated_temporal_source import (
    _genesis,append_signed_observation,public_key_hex,sign_source_observation,
)

SOURCE="time005:publisher-measured-clock"
ENV="time005:source-signed-observer-process"


def main():
    key=Ed25519PrivateKey.generate()
    pub=public_key_hex(key)
    with tempfile.TemporaryDirectory() as directory:
        journal=Path(directory)/"source.jsonl"
        checkpoint=Path(directory)/"observer.json"
        pin=_genesis(SOURCE,pub)
        count=0
        workers=[]

        def emit(value):
            nonlocal pin,count
            signed=sign_source_observation(
                key,source=SOURCE,sequence=count,
                time=f"{count}/1",value=f"{value}/1")
            pin,count=append_signed_observation(
                journal,signed,expected_head=pin,expected_count=count,
                source=SOURCE,public_key=pub)
            sleep(0.004)  # wall-clock gaps are genuinely measured by publisher

        def worker(host,*,before=None,after=None):
            command=[sys.executable,"-m","vli.authenticated_temporal_source",
                     "--journal",str(journal),
                     "--checkpoint",str(checkpoint),
                     "--source",SOURCE,"--public-key",pub,
                     "--head",pin,"--count",str(count),
                     "--environment",ENV,"--host",host]
            if before is not None:
                command+=["--crash-before",str(before)]
            if after is not None:
                command+=["--crash-after",str(after)]
            completed=subprocess.run(
                command,capture_output=True,text=True,timeout=45)
            if completed.returncode not in (0,71,72):
                raise RuntimeError(completed.stderr[-4000:])
            result=(json.loads(completed.stdout)
                    if completed.returncode==0 else None)
            workers.append({
                "host":host,"process_exit_code":completed.returncode,
                "source_seen":count,
                "accepted_in_process":(
                    result["newly_admitted_in_this_process"] if result else None),
                "pid":result["process_id"] if result else None,
            })
            return result

        for x in [1,3,5]:emit(x)
        worker("A",before=2)  # ungraceful process death before source commit
        first=worker("B")
        assert first["active_environment_family"]=="affine"
        earlier=load_json(checkpoint)
        g0=TemporalIntelligenceGround.restore(earlier,
            expected_head=earlier["head"],expected_identity=earlier["identity"])
        c0=EvolvingEnvironmentLanguage(g0).current(expected_head=g0.head)
        for x in [100,101,104,109]:emit(x)
        worker("B-restarted",after=5)  # durable commit but no acknowledgment
        final=worker("C")
        assert final["active_environment_family"]=="quadratic"
        later=load_json(checkpoint)
        g1=TemporalIntelligenceGround.restore(later,
            expected_head=later["head"],expected_identity=later["identity"])
        c1=EvolvingEnvironmentLanguage(g1).current(expected_head=g1.head)
        migration=certify_language_migration(
            earlier,c0,later,c1,before_head=earlier["head"],
            after_head=later["head"],identity=g1.identity)
        assert verify_language_migration(earlier,c0,later,c1,migration,
            before_head=earlier["head"],after_head=later["head"],
            identity=g1.identity)
        print(json.dumps({
            "schema":"vl-i.time005.source-authorized-process-recovery.v1",
            "source_key_exclusive_to_publisher_process":True,
            "source_public_key_fingerprint":sha256(bytes.fromhex(pub)).hexdigest(),
            "externally_pinned_source_journal_head":pin,
            "source_signed_sample_count":count,
            "observer_identity_preserved":first["observer_identity"]==
                                          final["observer_identity"],
            "actual_worker_processes":workers,
            "new_source_effects_after_final_recovery":
                    final["newly_admitted_in_this_process"],
            "old_observation_history_retained":migration["historical_source_prefix_exact"],
            "interpreter_succession":[c0["family"],c1["family"]],
            "publisher_clock_capture_gaps_ns":
                    final["measured_source_inter_sample_gaps_ns"],
            "source_covers_inter_sample_continuum":False,
            "authenticated_physical_values":False,
            "multi_host_hardware_verified":False,
            "quorum_fencing_verified":False,
            "latest_source_ground_head":final["result_ground_head"],
        },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
