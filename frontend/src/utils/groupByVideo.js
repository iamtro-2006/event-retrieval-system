function getResultScore(item) {
  return Number(
    item.temporal?.avg_score ??
      item.temporal?.video_score ??
      item.similarity ??
      0
  );
}

function getResultTime(item) {
  return Number(
    item.temporal?.start_time ??
      item.timestamp ??
      0
  );
}

export function groupByVideo(results = []) {
  return results.reduce((groups, item) => {
    const videoId = item.video_id || "unknown_video";

    if (!groups[videoId]) {
      groups[videoId] = [];
    }

    groups[videoId].push(item);
    return groups;
  }, {});
}

export function groupByVideoSorted(results = []) {
  const groups = groupByVideo(results);

  return Object.entries(groups)
    .map(([videoId, items]) => ({
      videoId,
      items: [...items].sort((a, b) => {
        const scoreDiff = getResultScore(b) - getResultScore(a);

        if (scoreDiff !== 0) {
          return scoreDiff;
        }

        return getResultTime(a) - getResultTime(b);
      }),
    }))
    .sort((a, b) => {
      const bestA = getResultScore(a.items[0]);
      const bestB = getResultScore(b.items[0]);

      return bestB - bestA;
    });
}