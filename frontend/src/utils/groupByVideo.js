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
        const scoreA = Number(a.similarity ?? 0);
        const scoreB = Number(b.similarity ?? 0);
        return scoreB - scoreA;
      }),
    }))
    .sort((a, b) => {
      const bestA = Number(a.items[0]?.similarity ?? 0);
      const bestB = Number(b.items[0]?.similarity ?? 0);
      return bestB - bestA;
    });
}