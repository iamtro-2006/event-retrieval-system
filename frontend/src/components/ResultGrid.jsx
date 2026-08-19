import { memo, useMemo } from "react";
import ResultCard from "./ResultCard";
import TemporalSequence from "./TemporalSequence";

const ResultGrid = memo(function ResultGrid({ results, columns, selectedId, onSelect, onSubmit, onPlay, onSimilaritySearch, onSurroundingImages, query }) {
  const hasTemporal = useMemo(() => results.some((r) => Array.isArray(r.matched_sequence) && r.matched_sequence.length), [results]);
  if (hasTemporal) {
    return <div className="temporal-sequences-grid temporal-flat-sequences-grid" style={{ "--sequence-cols": columns }}>{results.map((result, index) =>
      Array.isArray(result.matched_sequence) && result.matched_sequence.length ? (
        <TemporalSequence key={result.id} result={result} sequenceIndex={index} index={index} selectedId={selectedId} onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} query={query} />
      ) : (
        <ResultCard key={result.id} result={result} index={index} selected={result.id === selectedId} onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} query={query} />
      )
    )}</div>;
  }
  return <div className="result-grid" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>{results.map((result, index) => <ResultCard key={result.id} result={result} index={index} selected={result.id === selectedId} onSelect={onSelect} onSubmit={onSubmit} onPlay={onPlay} onSimilaritySearch={onSimilaritySearch} onSurroundingImages={onSurroundingImages} query={query} />)}</div>;
});
export default ResultGrid;
