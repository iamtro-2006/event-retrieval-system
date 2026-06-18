const image = (video, frame) => `/mock/${video}/${String(frame).padStart(6, "0")}.jpg`;

function frame(video, frameId, timestamp, score, eventIndex, query) {
  return {
    id: `${video}_${frameId}_${eventIndex}`,
    video_id: video,
    frame_id: frameId,
    keyframe_id: frameId,
    frame_name: `${String(frameId).padStart(6, "0")}.jpg`,
    image_url: image(video, frameId),
    video_url: `/mock/${video}.mp4`,
    timestamp,
    timestamp_sec: timestamp,
    score,
    sub_query_idx: eventIndex,
    sub_query: query,
  };
}

function temporalResult(video, rank, events) {
  const matched = events.map((item, index) => frame(video, 100 + index * 25, 4.5 + index * 6.25, 0.34 - index * 0.025, index, item));
  return {
    id: `${video}-sequence-${rank}`,
    video_id: video,
    frame_id: matched[Math.floor(matched.length / 2)].frame_id,
    frame_name: matched[Math.floor(matched.length / 2)].frame_name,
    image_url: matched[Math.floor(matched.length / 2)].image_url,
    video_url: `/mock/${video}.mp4`,
    timestamp: matched[0].timestamp,
    similarity: matched.reduce((sum, item) => sum + item.score, 0) / matched.length,
    rank,
    matched_sequence: matched,
    temporal: {
      start_time: matched[0].timestamp,
      end_time: matched.at(-1).timestamp,
      duration_sec: matched.at(-1).timestamp - matched[0].timestamp,
      avg_score: matched.reduce((sum, item) => sum + item.score, 0) / matched.length,
    },
  };
}

export const SEARCH_UI_SCENARIOS = {
  semantic: {
    query: "a red car parked beside a building",
    searchMode: "semantic",
    expected: { layout: "responsive-grid", sequenceLength: 0 },
    results: Array.from({ length: 12 }, (_, index) => ({
      id: `semantic-${index}`,
      video_id: `L21_V${String(index + 1).padStart(3, "0")}`,
      frame_id: 20 + index,
      frame_name: `${String(20 + index).padStart(6, "0")}.jpg`,
      image_url: image(`L21_V${String(index + 1).padStart(3, "0")}`, 20 + index),
      video_url: `/mock/L21_V${String(index + 1).padStart(3, "0")}.mp4`,
      timestamp: index * 3.2,
      similarity: 0.38 - index * 0.01,
      matched_sequence: [],
    })),
  },
  temporal2: {
    query: "a cyclist rides on the road; then turns left",
    searchMode: "temporal",
    expected: { layout: "fluid-sequence", sequenceLength: 2, carousel: false },
    results: [temporalResult("L22_V011", 1, ["cyclist rides on the road", "cyclist turns left"])],
  },
  temporal3: {
    query: "a person enters; picks up a box; leaves",
    searchMode: "temporal",
    expected: { layout: "fluid-sequence", sequenceLength: 3, carousel: false },
    results: [temporalResult("L22_V018", 1, ["person enters", "picks up a box", "person leaves"])],
  },
  temporal4: {
    query: "a woman enters; checks a shelf; takes a product; walks away",
    searchMode: "temporal",
    expected: { layout: "carousel-sequence", sequenceLength: 4, carousel: true },
    results: [temporalResult("L23_V004", 1, ["woman enters", "checks shelf", "takes product", "walks away"])],
  },
};
