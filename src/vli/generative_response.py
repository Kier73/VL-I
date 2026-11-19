"""COORD-002: model-description -> generated neural law -> renderable response-state.

A complete *described* VLD transition Ground is compiled to an exact ReLU
network, whose weights are generated without acquiring pretrained payloads.
An intention locates an opaque state position BEFORE the next state, text,
code, or byte tokens are materialized. VLD's actual neural synthesizer
evaluates the transition; an independent transition-system oracle qualifies
it. The existing E-CPU-byte/VLD-DAG response renderer then progressively
materializes the new generated views.

This is bounded finite-state model formation, NOT pretrained checkpoint
recovery, general natural-language understanding, or a hosted language model.
No tools are executed. Byte tokens are UTF-8 bytes, not LLM BPE tokens.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from .ground import _digest, _hex64, _require
from .response_state import (
    ResponseFormation, ResponseSourceGround, SourceSlot,
    _ecpu_verify, _strict_json, compile_intention,
)

MODEL_SCHEMA = "vl-i.generated-transition-model.v1"
POSITION_SCHEMA = "vl-i.generated-response-position.v1"
STEP_SCHEMA = "vl-i.generated-transition-receipt.v1"
TURN_SCHEMA = "vl-i.generated-conversation-turn.v1"
SESSION_SCHEMA = "vl-i.generated-conversation-checkpoint.v1"
FORM_SCHEMA = "vl-i.generated-response-formation.v1"

ALLOWED_VIEWS = frozenset(("text", "page", "code", "tokens", "tool_call"))
MAX_TURNS = 48
MAX_SOURCE_BYTES = 1_048_576


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


class GeneratedModelGround:
    """Compile a real VLD GroundTransitionSystem into a generated neural model.

    The source is a byte description, not a pretrained parameter snapshot.
    VLD itself proves the compiled network matches every declared transition.
    """

    def __init__(self, description_bytes: bytes):
        from vld.signal.transitions import GroundTransitionSystem
        from vld.models.procedural import compile_fsm_to_neural

        _require(type(description_bytes) is bytes and
                 0 < len(description_bytes) <= MAX_SOURCE_BYTES,
                 "Invalid descriptive model source")
        source = _strict_json(description_bytes)
        _require(type(source) is dict, "Model description must be a JSON object")
        system = GroundTransitionSystem.from_dict(source)
        _require(2 <= len(system.states) <= 16 and
                 2 <= len(system.actions) <= 16 and
                 1 <= len(system.channels) <= 8 and
                 system.complete,
                 "Description-native neural compilation requires a bounded complete Ground")
        # Reference VLD compiler, including its exhaustive whole-domain check.
        neural = compile_fsm_to_neural(system)
        _require(neural.system_fingerprint == system.fingerprint,
                 "Generated weights do not belong to source Ground")
        self.system = system
        self.neural = neural
        self.description_bytes = description_bytes
        self.ecpu_chunk_digests = tuple(_ecpu_verify(description_bytes))
        weights = {
            "hidden_weight": [list(r) for r in neural.hidden_weight],
            "hidden_bias": list(neural.hidden_bias),
            "next_state_weight": [list(r) for r in neural.next_state_weight],
            "channel_weight": [list(r) for r in neural.channel_weight],
        }
        self.weights_digest = _digest("vl-i.generated-neural-weights.v1", weights)
        record = {
            "schema": MODEL_SCHEMA,
            "description_sha256": _sha(description_bytes),
            "description_bytes": len(description_bytes),
            "ecpu_source_chunk_digests": list(self.ecpu_chunk_digests),
            "vld_system_fingerprint": system.fingerprint,
            "generated_weights_digest": self.weights_digest,
            "generated_hidden_units": len(neural.hidden_bias),
            "states": list(system.states),
            "actions": list(system.actions),
            "channels": list(system.channels),
            "source": "exact_supplied_description_bytes_no_pretrained_weight_payload",
            "neural_equivalence": "VLD_WHOLE_DECLARED_FINITE_DOMAIN_CHECKED",
        }
        record["ground_digest"] = _digest(MODEL_SCHEMA, record)
        self.manifest = record

    def verify(self):
        """Independent recompilation detects changed description or source law."""
        rebuilt = GeneratedModelGround(self.description_bytes)
        _require(rebuilt.manifest == self.manifest,
                 "Rebuilt generative model Ground differs")
        return True

    def reference(self, state: str, action: str):
        result = self.system.step(state, action)
        return result.target, result.output

    def generated(self, state: str, action: str):
        _require(state in self.system.states and action in self.system.actions,
                 "Unadmitted generative state or intention action")
        ns, outputs = self.neural.step_neural(state, action)
        expected = self.reference(state, action)
        _require((ns, outputs) == expected,
                 "Generated neural relation disagrees with independent VLD source oracle")
        return ns, outputs

    def generated_program(self):
        """Synthesize a standalone Python *pure function* from source law.

        No template contains the model's transitions; they are admitted and
        compiled from arbitrary observed descriptions. No exec occurs here.
        """
        rows = []
        for s in sorted(self.system.states):
            for a in sorted(self.system.actions):
                next_state, output = self.reference(s, a)
                rows.append(f"    ({s!r}, {a!r}): ({next_state!r}, {tuple(output)!r}),")
        return (
            "# Generated by VL-I from a pinned VLD transition Ground.\n"
            "# Pure data-derived function; no external tools or model backend.\n"
            "TRANSITIONS = {\n" + "\n".join(rows) + "\n}\n\n"
            "def transition(state, action):\n"
            "    \"\"\"Return (next_state, observed_channels); reject unknown states.\"\"\"\n"
            "    return TRANSITIONS[(state, action)]\n"
        )


class GenerativeFormation:
    """A position exists before its output, with staged neural + view formation.

    The outer coordinate depends on the Ground, causal context and intention,
    NEVER on a precomputed result or numeric next-state value.
    """

    def __init__(self, ground: GeneratedModelGround, *, state: str, action: str,
                 prompt: str, history_head: str, observer: str = "vl-i-observer",
                 views: tuple[str, ...] = ("text", "page", "code", "tokens")):
        _require(isinstance(ground, GeneratedModelGround), "Require generated model Ground")
        _require(state in ground.system.states and action in ground.system.actions,
                 "Intention is outside model's state/action topology")
        _require(type(prompt) is str and 0 < len(prompt.encode("utf-8")) <= 8192,
                 "Invalid prompt")
        _require(_hex64(history_head), "Missing causal conversation head")
        _require(type(observer) is str and observer and len(observer) <= 80,
                 "Invalid observer")
        _require(type(views) is tuple and 1 <= len(views) <= len(ALLOWED_VIEWS)
                 and len(set(views)) == len(views)
                 and set(views).issubset(ALLOWED_VIEWS),
                 "Invalid output observation views")
        self.ground = ground
        self.state, self.action, self.prompt = state, action, prompt
        self.history_head, self.observer, self.views = history_head, observer, views
        body = {
            "schema": POSITION_SCHEMA,
            "model_ground_digest": ground.manifest["ground_digest"],
            "causal_history_head": history_head,
            "source_state": state, "intention_action": action,
            "prompt": prompt, "observer": observer,
            "requested_views": list(views),
            "scope": "state_position_prior_to_neural_and_rendered_output",
        }
        self.position = {**body, "position_digest": _digest(POSITION_SCHEMA, body)}
        self.coordinate = "state://vli/" + self.position["position_digest"]
        self.inner: ResponseFormation | None = None
        self.step_receipt: dict | None = None
        self._source: ResponseSourceGround | None = None
        self._inner_intent = None
        self.generative_step_count = 0

    def _generate_content(self, target: str, channel_outputs: tuple[int, ...]):
        channels = dict(zip(self.ground.system.channels, channel_outputs))
        ordered = ", ".join(f"{name}={value}" for name, value in channels.items())
        history_note = (
            f"The causal predecessor is {self.history_head[:16]}."
            if self.history_head else "No causal predecessor is available."
        )
        text = (
            f"The observer asked: {self.prompt}\n"
            f"Under the described model's {self.action!r} relation, "
            f"state {self.state!r} becomes {target!r}. "
            f"The independently qualified channel observations are {ordered}. "
            f"{history_note}\n"
            "This continuation was synthesized from the admitted transition "
            "Ground and generated neural parameters, not a pretrained model server."
        )
        program = self.ground.generated_program()
        doc = (
            f"# Response-state observation\n\n"
            f"## Intention\n\n{self.prompt}\n\n"
            f"## Located state\n\n"
            f"- Source: {self.state}\n- Action: {self.action}\n"
            f"- Formed target: {target}\n"
            f"- Channels: {ordered}\n"
            f"- Ground: {self.ground.manifest['ground_digest']}\n"
            f"- Generated neural weights: {self.ground.weights_digest}\n\n"
            f"## Causal context\n\n{history_note}\n"
            f"\n## Available model actions\n\n"
            + "\n".join(f"- {a}" for a in self.ground.system.actions)
        )
        proposal = {
            "tool": "inspect_state",
            "arguments": {
                "model_ground_digest": self.ground.manifest["ground_digest"],
                "state": target, "read_only": True,
            },
        }
        return {
            "text": text.encode("utf-8"),
            "page": doc.encode("utf-8"),
            "code": program.encode("utf-8"),
            "tokens": json.dumps(list(text.encode("utf-8")),
                                 separators=(",", ":")).encode("utf-8"),
            "tool_call": json.dumps(proposal, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"),
        }

    def _form(self):
        target, values = self.ground.generated(self.state, self.action)
        self.generative_step_count += 1
        receipt = {
            "schema": STEP_SCHEMA,
            "position_digest": self.position["position_digest"],
            "model_ground_digest": self.ground.manifest["ground_digest"],
            "generated_weights_digest": self.ground.weights_digest,
            "input_state": self.state, "action": self.action,
            "target_state": target, "channel_output": list(values),
            "whole_domain_compiler_reference": self.ground.system.fingerprint,
            "independent_source_oracle_equal": True,
            "neural_step_executed": True,
        }
        receipt["step_digest"] = _digest(STEP_SCHEMA, receipt)
        outputs = self._generate_content(target, values)
        slots = tuple(SourceSlot(name, name, outputs[name]) for name in self.views)
        source = ResponseSourceGround(slots)
        typed = compile_intention(self.prompt, slots=self.views, observer=self.observer,
                                  history=(
                                      {"role": "user", "content": self.prompt},
                                  ))
        self._source = source
        self._inner_intent = typed
        self.inner = ResponseFormation(source, typed)
        self.step_receipt = receipt

    def advance(self, *, max_nodes: int = 4):
        if self.inner is None:
            self._form()
        else:
            self.inner.advance(max_nodes=max_nodes)
        return self.observe()

    def observe(self) -> dict:
        if self.inner is None:
            phase, views, inner_head = "LOCATED", [], None
            inner_progress = None
        else:
            progress = self.inner.observe()
            phase = progress["state"]
            views = progress["projections"]
            inner_head = progress["dag_state_digest"]
            inner_progress = progress
        result = {
            "schema": FORM_SCHEMA,
            "coordinate": self.coordinate,
            "position_digest": self.position["position_digest"],
            "phase": phase,
            "step_receipt": self.step_receipt,
            "projections": views,
            "inner_dag_state_digest": inner_head,
            "inner_observation": inner_progress,
            "neural_steps_executed_this_formation": self.generative_step_count,
            "scope": "new_structurally_generated_content_from_described_finite_model",
        }
        result["observation_digest"] = _digest(FORM_SCHEMA, result)
        return result

    @property
    def complete(self) -> bool:
        return self.inner is not None and self.inner.dag.is_complete

    def finish(self):
        # Bounded and deterministic; all DAG internal work stays with VLD.
        for _ in range(16):
            if self.complete:
                return self.observe()
            self.advance(max_nodes=16)
        raise ValueError("Generated response did not materialize within declared steps")


class GenerativeConversation:
    """No tool effects; causal, replay-verifiable state transitions across turns."""

    def __init__(self, ground: GeneratedModelGround, *, observer="vl-i-observer"):
        _require(isinstance(ground, GeneratedModelGround), "Missing generative Ground")
        self.ground = ground
        self.observer = observer
        self.state = ground.system.initial_state
        self.events: list[dict] = []
        self.genesis = _digest(SESSION_SCHEMA, {
            "model_ground_digest": ground.manifest["ground_digest"],
            "observer": observer,
            "initial_state": self.state,
        })
        self.head = self.genesis

    def respond(self, prompt: str, action: str, *,
                views: tuple[str, ...] = ("text", "page", "code", "tokens")) -> dict:
        _require(len(self.events) < MAX_TURNS, "Causal conversation capacity exceeded")
        formation = GenerativeFormation(
            self.ground, state=self.state, action=action, prompt=prompt,
            history_head=self.head, observer=self.observer, views=views)
        result = formation.finish()
        receipt = result["step_receipt"]
        event = {
            "schema": TURN_SCHEMA, "sequence": len(self.events),
            "previous": self.head,
            "model_ground_digest": self.ground.manifest["ground_digest"],
            "prompt": prompt, "action": action,
            "requested_views": list(views),
            "coordinate": result["coordinate"],
            "from_state": self.state,
            "to_state": receipt["target_state"],
            "step_digest": receipt["step_digest"],
            "generated_observation_digest": result["observation_digest"],
            "selected_render_source": formation._source.manifest["ground_digest"],
            "neural_forward_steps": formation.generative_step_count,
        }
        event["turn_digest"] = _digest(TURN_SCHEMA, event)
        self.events.append(event)
        self.head = event["turn_digest"]
        self.state = receipt["target_state"]
        return {"turn": dict(event), "response": result}

    def checkpoint(self):
        body = {
            "schema": SESSION_SCHEMA,
            "model_ground_digest": self.ground.manifest["ground_digest"],
            "observer": self.observer,
            "genesis": self.genesis,
            "events": [dict(e) for e in self.events],
            "state": self.state, "head": self.head,
        }
        body["checkpoint_digest"] = _digest(SESSION_SCHEMA, body)
        return body

    @classmethod
    def restore(cls, ground: GeneratedModelGround, checkpoint: dict,
                *, expected_head: str):
        _require(type(checkpoint) is dict and
                 checkpoint.get("schema") == SESSION_SCHEMA and
                 _hex64(expected_head) and
                 checkpoint.get("checkpoint_digest") == expected_head,
                 "Missing trusted causal conversation checkpoint")
        body = {k:v for k,v in checkpoint.items() if k != "checkpoint_digest"}
        _require(_digest(SESSION_SCHEMA, body) == expected_head,
                 "Conversation checkpoint altered")
        _require(checkpoint.get("model_ground_digest") == ground.manifest["ground_digest"]
                 and type(checkpoint.get("events")) is list
                 and len(checkpoint["events"]) <= MAX_TURNS,
                 "Wrong source Ground or invalid history")
        fresh = cls(ground, observer=checkpoint["observer"])
        _require(fresh.genesis == checkpoint["genesis"], "Changed conversation genesis")
        for row in checkpoint["events"]:
            _require(type(row) is dict and set(row) == {
                "schema", "sequence", "previous", "model_ground_digest",
                "prompt", "action", "requested_views", "coordinate",
                "from_state", "to_state", "step_digest",
                "generated_observation_digest", "selected_render_source",
                "neural_forward_steps", "turn_digest",
            }, "Malformed semantic turn")
            regenerated = fresh.respond(row["prompt"], row["action"],
                                        views=tuple(row["requested_views"]))
            _require(regenerated["turn"] == row,
                     "Ground evolution differs from pinned causal history")
        _require(fresh.state == checkpoint["state"] and
                 fresh.head == checkpoint["head"],
                 "Restored causal state does not match checkpoint")
        return fresh
