# Phase 1 — implementation status and governance

## Scope

This records code safeguards. It does not claim model acceptance, Independent Test
validation, or completion of later Blockchain, deployment, or MLOps work.

## Implemented safeguards

- Group-aware Train/Validation split validates required fields, dimensions, group
  IDs, partition names, non-empty partitions, and disjointness across all splits.
- Leakage metadata is computed from actual checks, not hard-coded PASS.
- Decision metrics separate automatic invalid recall, invalid referral rate, and
  referral-inclusive invalid interception. PENDING_REVIEW is never automatic detection.
- INVALID membership is an explicit policy input. No implicit y != VALID rule;
  BLANK semantics are not inferred.
- Zero denominators are undefined (null), not silently replaced by one or zero.
- Acceptance and threshold selection fail closed until PM policy approval and
  separate OOF authorization are recorded.
- CNN probability cache includes data/split/group/label arrays, training config,
  checkpoint checksum, seed, folds, and schema version. Writes are atomic.
- Observed Test is descriptive only, not a model/threshold selection or acceptance set.
- Regression tests cover leakage, malformed inputs, referral metrics, zero
  denominators, policy gates, and cache invalidation.

## Decisions still blocking final acceptance

1. PM-03: approve INVALID label semantics. BLANK remains unresolved; do not
   assume it maps to PENDING_REVIEW or INVALID.
2. PM-01 / PM-06: approve numeric targets. Coverage 0.80 and 0.34 are not approved.
   NFR-10 0.98 remains a referenced target pending PM confirmation of its role.
3. PM-04: finalize CI estimands, weighting, minimum groups, bootstrap method and
   replicate count, and higher-level clustering. No unapproved CI is generated.
4. PM-10: Independent Test acceptance remains BLOCKED without independent source
   and governance.
5. WP-05A: OOF is not run by default; separate explicit PM approval is required.
6. Runtime: Python environment must be validated before model training.

## Commands

Run deterministic unit tests from modules/ballot-validation-ai:

    python -m unittest discover -s tests -v

The main training pipeline is intentionally blocked until both business policy
and separate OOF authorization are approved. It does not generate a dataset
automatically. Dataset generation, training, OOF, and Observed Test evaluation
are not part of the unit-test command.

## Status terminology

- PASS: code-level check executed and succeeded.
- BLOCKED: a source, policy, authorization, or runtime prerequisite is missing.
- NOT RUN: no execution evidence exists.
- Observed Test: previously viewed partition for descriptive internal evaluation;
  it is not an Independent Test.
