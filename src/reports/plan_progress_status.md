# Plan Progress Status

- Completed: 4
- Partial: 7
- Missing: 2

- Phase -1 Data bootstrap [completed]: download_sp500.py + outputs under data/raw_market and metadata checksums
- Phase 0 Research contract [completed]: configs/research_contract.yaml created
- Phase 1 Data QA/alignment [completed]: constituent download + qa report + first-valid-date analysis
- Phase 1A Missing/scaling policy [partial]: implemented train-only scaling in runners; policy docs/tests still incomplete
- Phase 2 Features G1-G6 [partial]: implemented G1/G2/G3; G4/G5/G6 still missing
- Phase 3 Sequence dataset builder [completed]: lag-window sequence builder in deep runner
- Phase 4 Model suite [partial]: implemented required deep/baseline families in torch; no paper-faithful external libs yet
- Phase 5 NLP pipeline [missing]: FinBERT sentiment/embeddings not implemented yet
- Phase 6 Multi-modal integration [missing]: feature-level/late/cross-attention multimodal fusion not implemented
- Phase 7 Full ablations G1..G6 [partial]: G1..G3 comparisons done; G4..G6 and full leave-one-group-out missing
- Phase 8 A-E experiment set [partial]: A/B done; C/D/E blocked by missing sentiment features
- Phase 9 Failure analysis [partial]: basic diagnostics done; full red-team leakage suite pending
- Phase 10 Promotion criteria [partial]: artifact snapshots exist; formal go/no-go thresholds not yet codified