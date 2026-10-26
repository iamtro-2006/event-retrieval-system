import test from "node:test";
import assert from "node:assert/strict";
import { selectAgentSettings } from "../src/utils/agentModels.js";

const models = [
  { model: "a", default: true, default_reasoning_effort: "medium", supported_reasoning_efforts: [{ reasoningEffort: "low" }, { reasoningEffort: "medium" }, { reasoningEffort: "high" }] },
  { model: "b", default_reasoning_effort: "low", supported_reasoning_efforts: [{ reasoningEffort: "low" }] },
];
test("initial selection uses the catalog model and effort defaults", () => {
  const settings = selectAgentSettings({ model: "", reasoningEffort: "" }, models);
  assert.equal(settings.model, "a");
  assert.equal(settings.reasoningEffort, "medium");
});
test("model changes keep supported effort and replace incompatible effort", () => {
  assert.equal(selectAgentSettings({ model: "a", reasoningEffort: "high" }, models).reasoningEffort, "high");
  assert.equal(selectAgentSettings({ model: "b", reasoningEffort: "high" }, models).reasoningEffort, "low");
});
