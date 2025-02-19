# PISA-derived Code Notice

The code in this `pisa` directory is derived from the PISA project (https://github.com/pisa-engine/pisa), which is licensed under the Apache License, Version 2.0.

Original Project:
- Name: PISA
- Repository: https://github.com/pisa-engine/pisa
- License: Apache License 2.0

The following files contain code adapted from PISA:
- cursor/max_scored_cursor.h
- cursor/scored_cursor.h
- searcher/daat_maxscore.h
- searcher/daat_wand.h
- searcher/taat_naive.h
- searcher/searcher.h
- util/topk_queue.h
- index_scorer.h

Modifications made to the original PISA code:
1. Refactored to work within the Knowhere codebase
2. Simplified and removed unused functionality
3. Modified interfaces to match Knowhere's requirements
4. Adapted code style to match Knowhere's conventions
5. Removed dependencies on PISA-specific components

This notice is provided in compliance with Section 4(b) of the Apache License 2.0.
