# Evasion variants

Rewrites of M4 attacks aimed at M6's defences rather than at the model. They
exist to answer one question: when a defence blocks an attack, is it detecting
the attack or only the wording the attack happened to use?

They live in a subdirectory because `load_attacks` globs `*.yaml`
non-recursively. The M4 corpus therefore stays at exactly seven attacks and
every "n/7" figure in `reports/m4_attacks.md` and `reports/m6_defenses.md`
continues to mean what it meant. Run these deliberately:

```bash
python -m threatrag.cli attack run --dir attacks/evasion
python -m threatrag.cli attack run --dir attacks/evasion -o configs/experiments/defense_injection_screen.yaml
```

A variant is only evidence if it does **both** things: evades the defence *and*
still lands against the undefended baseline. A rewrite that slips past a regex
but no longer works has defeated nothing, and is reported as such rather than
iterated on until it succeeds — tuning a payload until it proves the point is
the shaping failure this project guards against everywhere else.

Same constraints as the main corpus: `SourceType.SYNTHETIC_ADVERSARIAL`, inert
YAML that nothing executes automatically, and beacons that only ever target the
loopback sink.
