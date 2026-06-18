import { memo, useMemo } from "react";
import ResultCard from "./ResultCard";
import TemporalSequence from "./TemporalSequence";

const ResultGrid = memo(function ResultGrid({ results, columns, selectedId, onSelect, onSubmit, onPlay, onSimilaritySearch, onSurroundingImages }) {
  const hasTemporal = useMemo(() => results.some((r) => Array.isArray(r.matched_sequence) && r.matched_sequence.length), [results]);
  if (hasTemporal) {
    return <div className="temporal-sequences-grid temporal-flat-sequences-grid" style={{ "--sequence-cols": columns }}>{results.map((result, index) =>
      Array.isArray(result.matched_sequence) && result.matched_sequence.length ? (
        <TemporalSequence key={result.id} result={result} sequenceIndex={index} selectedId={selectedId} onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} />
      ) : (
        <ResultCard key={result.id} result={result} selected={result.id === selectedId} onSelect={onSelect} onSubmit={onSubmit} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} />
      )
    )}</div>;
  }
  return <div className="result-grid" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>{results.map((result) => <ResultCard key={result.id} result={result} selected={result.id === selectedId} onSelect={onSelect} onSubmit={onSubmit} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} />)}</div>;
});
export default ResultGrid;
