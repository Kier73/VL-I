"""ACT-001: cost-aware active inquiry across signed rational observer channels.

This is the rational, authenticated analogue of VLD TRACE-002's rank-aware
finite-field experiment. A candidate *shared order* is explicitly supplied:
neither a future source value nor an unknown true physical law is inferred
from an offer. Real observations enter only via signed publisher journals.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from itertools import combinations

from .ground import _digest, _require
from .temporal_intelligence import _fraction, _encode
from .authenticated_temporal_source import (
    verified_source_journal, _genesis, _extend, _packet_digest,
)
from .discovered_state_space import _gauss, _cramer, _determinant, _predict
from .intention_state_space import navigate_relation, _encode_vec, _bounded

SCHEMA = "vl-i.act001.signed-rational-active-inquiry.v1"
OFFER = "vl-i.act001.rank-probe-offer.v1"
RECEIPT = "vl-i.act001.authenticated-rank-probe.v1"
LAW = "vl-i.act001.observer-shared-rational-law.v1"
MAX_CHANNELS = 4
MAX_ORDER = 4
MAX_EVENTS = 64


def _rank(rows, width):
    """Rank-producing algorithm: exact forward-elimination over rationals."""
    working = [[_bounded(Fraction(value)) for value in row] for row in rows]
    _require(all(len(row) == width for row in working),
             "Malformed rank observation dimensions")
    rank = 0
    for col in range(width):
        pivot = next((i for i in range(rank, len(working))
                      if working[i][col]), None)
        if pivot is None:
            continue
        working[rank], working[pivot] = working[pivot], working[rank]
        scalar = working[rank][col]
        for j in range(col, width):
            working[rank][j] = _bounded(working[rank][j] / scalar)
        for i in range(rank+1, len(working)):
            factor = working[i][col]
            for j in range(col, width):
                working[i][j] = _bounded(
                    working[i][j] - factor * working[rank][j])
        rank += 1
        if rank == width:
            break
    return rank


def _audit_rank(rows, width):
    """Separate exact minor enumeration; no producer's pivot/rank routine."""
    for dim in range(min(width, len(rows)), 0, -1):
        for chosen_rows in combinations(rows, dim):
            for columns in combinations(range(width), dim):
                minor = [[row[j] for j in columns] for row in chosen_rows]
                if _determinant(minor):
                    return dim
    return 0


def _check_spec(spec):
    _require(type(spec) is dict and set(spec) == {
        "id", "source", "public_key", "journal_path", "expected_head",
        "expected_count", "cost"}, "Malformed source channel descriptor")
    _require(type(spec["id"]) is str and spec["id"] and
             type(spec["source"]) is str and spec["source"] and
             type(spec["public_key"]) is str and len(spec["public_key"]) == 64 and
             type(spec["expected_count"]) is int and
             0 <= spec["expected_count"] <= MAX_EVENTS and
             type(spec["cost"]) is str and _fraction(spec["cost"]) > 0,
             "Invalid authenticated channel identity or cost")


def _prefix_description(spec, packets, count):
    _require(count >= 2 and len(packets) >= count,
             "Insufficient known prefix for rational observer")
    source, key = spec["source"], spec["public_key"]
    head = _genesis(source, key)
    for i, packet in enumerate(packets[:count]):
        _require(packet["sequence"] == i and packet["source"] == source,
                 "Replayed signed channel has different source identity")
        head = _extend(head, packet)
    _require(head == spec["expected_head"],
             "Externally pinned initial source prefix differs")
    times = tuple(_fraction(p["time"]) for p in packets[:count])
    values = tuple(_fraction(p["values"][0]) for p in packets[:count])
    delta = times[1]-times[0]
    _require(delta > 0 and
             all(times[i]-times[i-1] == delta for i in range(2, count)),
             "Channel time coordinates are not a regular source grid")
    return {
        "id": spec["id"], "source": source, "public_key": key,
        "source_head": head, "source_count": count,
        "cost": spec["cost"], "time_origin": _encode(times[0]),
        "time_step": _encode(delta),
        "values": [_encode(q) for q in values],
        "packet_digests": [_packet_digest(p) for p in packets[:count]],
    }


def _seal(session):
    body = {k:v for k,v in session.items() if k != "inquiry_digest"}
    return {**body, "inquiry_digest": _digest(SCHEMA, body)}


def _check_session(session):
    _require(type(session) is dict and session.get("schema") == SCHEMA and
             session.get("inquiry_digest") == _digest(
                 SCHEMA, {k:v for k,v in session.items()
                          if k != "inquiry_digest"}),
             "Inquiry transcript does not match its state identity")
    return True


def begin_inquiry(specs, *, order):
    """Authenticate ONLY prefixes available before any prospective offer."""
    _require(type(order) is int and 2 <= order <= MAX_ORDER and
             type(specs) in (list, tuple) and
             2 <= len(specs) <= MAX_CHANNELS,
             "A bounded multi-observer shared-order contract is required")
    seen_ids, seen_sources = set(), set()
    channels = []
    for spec in specs:
        _check_spec(spec)
        _require(spec["id"] not in seen_ids and
                 (spec["source"], spec["public_key"]) not in seen_sources,
                 "Duplicate observer identity or signed publisher")
        seen_ids.add(spec["id"])
        seen_sources.add((spec["source"], spec["public_key"]))
        _require(spec["expected_count"] >= order,
                 "Cannot offer rank probe without preceding state registers")
        packets = verified_source_journal(
            spec["journal_path"], expected_head=spec["expected_head"],
            expected_count=spec["expected_count"], source=spec["source"],
            public_key=spec["public_key"])
        channels.append(_prefix_description(spec, packets, len(packets)))
    channels.sort(key=lambda c: c["id"])
    roots = [{
        "id": c["id"], "source": c["source"], "public_key": c["public_key"],
        "source_head": c["source_head"], "source_count": c["source_count"],
        "cost": c["cost"],
    } for c in channels]
    _require(all(c["time_origin"] == channels[0]["time_origin"] and
                 c["time_step"] == channels[0]["time_step"]
                 for c in channels),
             "Shared recurrence requires aligned source time coordinates")
    return _seal({
        "schema": SCHEMA, "order": order,
        "initial_source_pins": roots,
        "channels": channels, "accepted": [],
        "rank": 0, "oracle_outputs_read_for_probe_selection": 0,
        "true_future_source_law_established": False,
    })


def _offer(session, rank_fn):
    rows = [[_fraction(x) for x in rec["regressor"]]
            for rec in session["accepted"]]
    width = session["order"]
    before = rank_fn(rows, width)
    _require(before == session["rank"],
             "Selected inquiry rank changed without an authenticated event")
    _require(before < width and len(session["accepted"]) < 2*width+4,
             "Shared recurrence identified or bounded probe budget exhausted")
    offers = []
    for channel in session["channels"]:
        values = [_fraction(x) for x in channel["values"]]
        if len(values) >= MAX_EVENTS:
            continue
        regressor = tuple(reversed(values[-width:]))
        gain = rank_fn(rows+[list(regressor)], width)-before
        offers.append((channel, regressor, gain))
    _require(offers, "No permitted observer probe remains")
    # Exact gain/cost first, then source ID for reproducibility. A bridge
    # offers no immediate rank gain and does not promise eventual gain.
    channel, regressor, gain = sorted(
        offers, key=lambda x: (
            -Fraction(x[2])/_fraction(x[0]["cost"]),
            _fraction(x[0]["cost"]), x[0]["id"]))[0]
    body = {
        "schema": OFFER, "previous_inquiry_digest": session["inquiry_digest"],
        "channel": channel["id"], "sequence": channel["source_count"],
        "preceding_signed_head": channel["source_head"],
        "regressor": [_encode(q) for q in regressor],
        "rank_before": before, "guaranteed_rank_gain": gain,
        "selection": "RANK_INCREASE" if gain else "BRIDGE_NO_RANK_GUARANTEE",
        "cost": channel["cost"],
        "selected_using_future_value": False,
    }
    return {**body, "offer_digest": _digest(OFFER, body)}


def choose_probe(session):
    """The selected channel and rank are determined before observing y(t)."""
    _check_session(session)
    return _offer(session, _rank)


def _accept(session, offer, packet, *, rank_fn):
    channels = [dict(c) for c in session["channels"]]
    selected = next(c for c in channels if c["id"] == offer["channel"])
    current_count = selected["source_count"]
    _require(packet["sequence"] == current_count and
             packet["source"] == selected["source"] and
             _extend(selected["source_head"], packet) is not None,
             "New source observation is not the selected next event")
    actual = _fraction(packet["values"][0])
    previous_time = _fraction(selected["time_origin"]) + (
        current_count * _fraction(selected["time_step"]))
    _require(_fraction(packet["time"]) == previous_time,
             "Selected source observation is not at the next sample time")
    expected_regressor = list(reversed(selected["values"][-session["order"]:]))
    _require(offer["regressor"] == expected_regressor,
             "Probe was not selected from known prior samples")
    new_head = _extend(selected["source_head"], packet)
    selected["source_head"] = new_head
    selected["source_count"] += 1
    selected["values"] = selected["values"] + [_encode(actual)]
    selected["packet_digests"] = (selected["packet_digests"] +
                                  [_packet_digest(packet)])
    rows = [[_fraction(q) for q in item["regressor"]]
            for item in session["accepted"]]
    rows.append([_fraction(q) for q in offer["regressor"]])
    next_rank = rank_fn(rows, session["order"])
    body = {
        "schema": RECEIPT, "offer": offer,
        "signed_packet_digest": _packet_digest(packet),
        "new_source_head": new_head,
        "observed_value": _encode(actual),
        "rank_after": next_rank,
        "source_authenticity": "ED25519_AND_EXTERNAL_PUBLISHER_HEAD",
    }
    receipt = {**body, "receipt_digest": _digest(RECEIPT, body)}
    updated = {
        **session, "channels": channels,
        "accepted": session["accepted"]+[{
            "channel": offer["channel"], "regressor": offer["regressor"],
            "value": _encode(actual), "receipt": receipt}],
        "rank": next_rank,
    }
    return _seal(updated)


def accept_signed_probe(session, offer, *, journal_path, expected_next_head):
    """Advance one actual source packet; caller owns the new trusted head."""
    _check_session(session)
    expected_offer = choose_probe(session)
    _require(offer == expected_offer, "Unauthorized rank proposal")
    channel = next(c for c in session["channels"]
                   if c["id"] == offer["channel"])
    packets = verified_source_journal(
        journal_path, expected_head=expected_next_head,
        expected_count=channel["source_count"]+1,
        source=channel["source"], public_key=channel["public_key"])
    old = packets[:-1]
    trusted = _genesis(channel["source"], channel["public_key"])
    _require([_packet_digest(p) for p in old] ==
             channel["packet_digests"],
             "Signed source prefix was replaced")
    for packet in old:
        trusted = _extend(trusted, packet)
    _require(trusted == channel["source_head"] and
             _extend(trusted, packets[-1]) == expected_next_head,
             "Publisher's signed head transition is not independently pinned")
    return _accept(session, offer, packets[-1], rank_fn=_rank)


def audit_inquiry(session, *, journal_paths, final_source_pins):
    """Rebuild every selection from initial pins and signed final journals.

    The auditor computes ranks from square minors rather than reusing the
    planner's row elimination. No recorded observation or self-hash grants
    source authority without the externally specified final publisher heads.
    """
    _check_session(session)
    _require(type(journal_paths) is dict and
             type(final_source_pins) is dict and
             set(journal_paths) == set(final_source_pins) ==
             {c["id"] for c in session["channels"]},
             "External source verification inputs incomplete")
    full = {}
    initial = []
    for root in session["initial_source_pins"]:
        ident = root["id"]
        pin = final_source_pins[ident]
        _require(type(pin) is dict and set(pin) == {"head", "count"} and
                 type(pin["count"]) is int and
                 root["source_count"] <= pin["count"] <= MAX_EVENTS,
                 "Expected an explicit complete external publisher pin")
        packets = verified_source_journal(
            journal_paths[ident], expected_head=pin["head"],
            expected_count=pin["count"], source=root["source"],
            public_key=root["public_key"])
        full[ident] = packets
        spec = {
            "id": ident, "source": root["source"],
            "public_key": root["public_key"],
            "expected_head": root["source_head"],
            "expected_count": root["source_count"], "cost": root["cost"],
            "journal_path": journal_paths[ident],
        }
        initial.append(_prefix_description(spec, packets, root["source_count"]))
    initial.sort(key=lambda c:c["id"])
    state = _seal({
        "schema": SCHEMA, "order": session["order"],
        "initial_source_pins": session["initial_source_pins"],
        "channels": initial, "accepted": [], "rank": 0,
        "oracle_outputs_read_for_probe_selection": 0,
        "true_future_source_law_established": False,
    })
    for recorded in session["accepted"]:
        offer = _offer(state, _audit_rank)
        _require(recorded["receipt"]["offer"] == offer and
                 offer["channel"] == recorded["channel"],
                 "Recorded adaptive inquiry choice was not rank-optimal")
        channel = next(c for c in state["channels"]
                       if c["id"] == offer["channel"])
        packet = full[channel["id"]][channel["source_count"]]
        state = _accept(state, offer, packet, rank_fn=_audit_rank)
        _require(state["accepted"][-1] == recorded,
                 "Independent signed inquiry replay disagrees")
    _require(state == session, "Independent transcript/revision audit failed")
    return True


def qualify_shared_law(session, *, journal_paths, final_source_pins):
    """Independently challenged common recurrence; no universal law claim."""
    audit_inquiry(session, journal_paths=journal_paths,
                  final_source_pins=final_source_pins)
    order = session["order"]
    _require(session["rank"] == order,
             "Shared recurrence not uniquely identified by active probes")
    rows = [[_fraction(x) for x in r["regressor"]]
            for r in session["accepted"]]
    outputs = [_fraction(r["value"]) for r in session["accepted"]]
    coefficients = _gauss(rows[:order], outputs[:order])
    independently = _cramer(rows[:order], outputs[:order])
    _require(coefficients is not None and coefficients == independently and
             all(sum((x*y for x,y in zip(row,coefficients)),
                     Fraction(0)) == answer
                 for row,answer in zip(rows,outputs)),
             "No independently identifiable shared rational law")
    witnesses = []
    for channel in session["channels"]:
        ident = channel["id"]
        pin = final_source_pins[ident]
        _require(pin["count"] >= channel["source_count"]+2,
                 "Need at least two new held-out signed observations per channel")
        packets = verified_source_journal(
            journal_paths[ident], expected_head=pin["head"],
            expected_count=pin["count"], source=channel["source"],
            public_key=channel["public_key"])
        _require([_packet_digest(p) for p in
                  packets[:channel["source_count"]]] ==
                 channel["packet_digests"],
                 "Original inquiry source evidence changed")
        origin, delta = (_fraction(channel["time_origin"]),
                         _fraction(channel["time_step"]))
        _require(all(_fraction(packet["time"]) == origin + i*delta
                     for i, packet in enumerate(packets)),
                 "Source heldout sampling grid changed after inquiry")
        values = tuple(_fraction(p["values"][0]) for p in packets)
        _predict(values, order, coefficients, channel["source_count"],
                 pin["count"]-channel["source_count"])
        witnesses.append({
            "id": ident, "initial_state": [_encode(q) for q in
                                           reversed(values[:order])],
            "observed_prefix_count": channel["source_count"],
            "challenged_holdout_count": pin["count"]-channel["source_count"],
            "trusted_final_publisher_head": pin["head"],
            "final_source_count": pin["count"],
            "final_source_witnesses": [_packet_digest(p) for p in packets],
        })
    body = {
        "schema": LAW, "inquiry_digest": session["inquiry_digest"],
        "order": order, "coefficients": [_encode(c) for c in coefficients],
        "channels": witnesses,
        "status": "EXACT_HELDOUT_QUALIFIED_SHARED_RATIONAL_MODEL",
        "external_world_law_proven": False,
        "all_query_selection_values_were_preobserved": False,
        "future_model_projection_requires_opt_in": True,
    }
    return {**body, "law_digest": _digest(LAW, body)}


def verify_shared_law(session, law, *, journal_paths, final_source_pins):
    """Numerical reference and source-authenticated proof must both match."""
    expected = qualify_shared_law(
        session, journal_paths=journal_paths,
        final_source_pins=final_source_pins)
    _require(expected == law, "Source-qualified shared law was altered")
    return True


def navigate_shared_law(law, channel_id, step, *, allow_projection=False):
    """Reuses TIME-007's bounded exact operator, never invents source samples."""
    _require(type(law) is dict and law.get("schema") == LAW and
             law.get("law_digest") == _digest(
                 LAW, {k:v for k,v in law.items() if k != "law_digest"}) and
             type(step) is int and step >= 0,
             "No independently certified observer law or coordinate")
    channel = next((c for c in law["channels"] if c["id"] == channel_id),
                   None)
    _require(channel is not None and step >= law["order"]-1 and
             (step < channel["final_source_count"] or allow_projection is True),
             "Unknown observer or unauthorized future coordinate")
    order = law["order"]
    c = tuple(_fraction(x) for x in law["coefficients"])
    A = (c,) + tuple(tuple(Fraction(int(j == i-1))
                          for j in range(order)) for i in range(1,order))
    V = tuple(_fraction(x) for x in channel["initial_state"])
    value,_,metrics = navigate_relation(A,V,step-(order-1),0)
    body = {
        "law_digest": law["law_digest"], "channel": channel_id,
        "requested_sample_coordinate": step, "value": _encode(value),
        "evidence_status": ("SIGNED_SOURCE_OBSERVATION" if
                            step < channel["final_source_count"] else
                            "UNVERIFIED_FUTURE_MODEL_PROJECTION"),
        "navigation_metrics": metrics,
        "physical_future_observed": False,
    }
    return body
