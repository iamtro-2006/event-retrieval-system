export function selectSimilarityModel(result, selectedModels, availableModels, fallbackModel) {
  const available = new Set(availableModels);
  const candidates = [
    ...(Array.isArray(result?.source_models) ? result.source_models : []),
    result?.model_key,
    result?.raw?.model_key,
    ...selectedModels,
    fallbackModel,
  ];
  for (const candidate of candidates) {
    for (const key of String(candidate || "").split(/[\s+/]+/).filter(Boolean)) {
      if (available.has(key)) return key;
    }
  }
  return availableModels[0] || "";
}
