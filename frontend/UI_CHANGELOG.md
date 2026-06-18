# UI optimization changelog

- Unified flat and grouped temporal rendering through `TemporalSequence.jsx`.
- 2 and 3 event sequences stretch to fill available width without placeholder gaps.
- More than 3 events use a 3-card viewport, horizontal scroll, arrows, and a visible range indicator.
- Added modal stack ordering for Surrounding, Similarity, and Video modals. The latest modal is always on top; closing closes the complete modal stack.
- Added video submission controls: 1 ms slider, exact millisecond input, configurable seek step, previous/next seek buttons, and submit at selected timestamp.
- Added semantic and temporal mock scenarios plus structural validation script.
- Production build and mock validation passed on 2026-06-18.
