# UI validation scenarios

1. Semantic: 12 independent result cards, responsive 3–6 column grid, no sequence container.
2. Temporal 2 events: exactly two equal-width cards filling the sequence body; no empty third slot.
3. Temporal 3 events: exactly three equal-width cards filling the sequence body.
4. Temporal 4 events: three cards visible at once, horizontal scroll plus previous/next controls and range indicator.
5. Modal nesting: Surround → Similar and Similar → Surround must preserve the previous modal underneath while the latest modal receives the highest z-index. Closing either backdrop closes the complete stack.
6. Video submission: seek with 1 ms slider, exact millisecond input and ±100/250/500/1000/5000 ms controls; submission sends the selected timestamp.
