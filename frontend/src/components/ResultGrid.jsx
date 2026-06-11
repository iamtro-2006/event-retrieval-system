import { motion } from "framer-motion";
import ResultCard from "./ResultCard";

export default function ResultGrid({ results, columns, selectedId, onSelect }) {
  return (
    <div
      className="result-grid"
      style={{
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      }}
    >
      {results.map((result, index) => (
        <motion.div
          key={result.id}
          initial={{ opacity: 0, rotateX: -60, y: -20 }}
          animate={{ opacity: 1, rotateX: 0, y: 0 }}
          transition={{ delay: index * 0.015, duration: 0.35 }}
        >
          <ResultCard
            result={result}
            selected={result.id === selectedId}
            onSelect={onSelect}
          />
        </motion.div>
      ))}
    </div>
  );
}