# CLT-004 — Nonlinear reachable-state witness refinement

Connect the public CLT-003 residual-space discovery to VL-64's finite nonlinear observer closure and VL-I's TDL-002 row-expression machinery. The former forms the minimal linear space of Boolean observables closed under action pullback; the latter already implements exact GF(2) row expression.

An important distinction: feature rank is the dimension of the span of function truth tables, not the number of independent bits on actual reachable states. In the finite 256-state experiment, adding nonlinear shear changes closure rank 4→12 while the number of observable classes stays 16. If the refined partition has identical fibers, all new features are nonlinear functions of old coordinates, so no additional source bit is needed. Adding a hidden-to-visible coupling changes rank 12→25 and classes 16→31. The linear complement may use 13 auxiliary features, yet if each old class has at most two new classes, a single conditional bit suffices.

For old map q and new map p, let K(z)={p(x):q(x)=z}. The minimum worst-case fixed-width binary witness on the declared domain is ceil(log2(max_z |K(z)|)). Lower bound: pigeonhole within a fiber. Upper bound: enumerate its new coordinates, use the index as a conditional witness. This is a finite source-derived lookup, not a general efficient decoder. The witness measurement may itself be expensive; table construction, proof, and source acquisition are not free. The old/new action alphabet is deliberately expanded; this is **not** a same-law-epoch TDL-002 succession certificate. The bridge imports only its mathematical _express helper, not its admission API.

Tests use actual pinned VL-64 experiment and independent native audit, then reconstruct every source coordinate with the finite witness. Expected rank deficits 8/13/21 and fixed witness widths 0/1/1 are CI-gated. No pretrained model result or external source authentication is claimed.

Run with pinned VL-64 checkout at e4459e05e1676468edd202630b890365480979ef on PYTHONPATH together with src and its experiments/observer_closure directory:

```sh
python -m unittest discover -s experiments/nonlinear_observer_refinement -v
```

Links: [VL-64 proposal](https://github.com/Kier73/VL-64/pull/44), [CLT-003](https://github.com/Kier73/In-Formation-Space/pull/382), [TDL-002](MINIMAL_REFINEMENT_TDL002.md), [SHAPE-010](SHAPE_010_CAUSAL_BASIS_EXPANSION.md). The next research question is whether a compact symbolic law can replace the fiber lookup and provide a cheaper measurement contract, including actual trained-state observations.
