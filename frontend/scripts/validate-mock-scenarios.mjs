import { SEARCH_UI_SCENARIOS } from "../src/mocks/searchScenarios.js";

const failures = [];
for (const [name, scenario] of Object.entries(SEARCH_UI_SCENARIOS)) {
  if (!scenario.results.length) failures.push(`${name}: no results`);
  for (const result of scenario.results) {
    const seq = result.matched_sequence || [];
    if (scenario.searchMode === "semantic" && seq.length !== 0) failures.push(`${name}: semantic contains sequence`);
    if (scenario.searchMode === "temporal" && seq.length !== scenario.expected.sequenceLength) failures.push(`${name}: expected ${scenario.expected.sequenceLength}, got ${seq.length}`);
    for (let i = 1; i < seq.length; i += 1) {
      if (seq[i].timestamp_sec <= seq[i - 1].timestamp_sec) failures.push(`${name}: timestamps are not strictly increasing`);
    }
    if (scenario.expected.carousel !== undefined && (seq.length > 3) !== scenario.expected.carousel) failures.push(`${name}: carousel rule mismatch`);
  }
}
if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("PASS: semantic, temporal-2, temporal-3 and temporal-4 UI scenarios are structurally valid.");
