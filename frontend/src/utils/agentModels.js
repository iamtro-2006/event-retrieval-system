export function selectAgentSettings(settings, models) {
  const model = models.find((item) => (item.model || item.id) === settings.model)
    || models.find((item) => item.default) || models[0];
  if (!model) return settings;
  const levels = model.supported_reasoning_efforts || [];
  const effort = levels.some((item) => item.reasoningEffort === settings.reasoningEffort)
    ? settings.reasoningEffort
    : model.default_reasoning_effort || levels[0]?.reasoningEffort || "";
  return { ...settings, model: model.model || model.id, reasoningEffort: effort };
}
