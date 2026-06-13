export function playNotifySound(type = "pending") {
  const AudioContext = window.AudioContext || window.webkitAudioContext;

  if (!AudioContext) return;

  const ctx = new AudioContext();

  const notesByType = {
    correct: [660, 880],
    wrong: [220, 180],
    warning: [440, 520],
    pending: [520],
  };

  const notes = notesByType[type] || notesByType.pending;

  notes.forEach((freq, index) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.value = freq;

    const start = ctx.currentTime + index * 0.09;
    const end = start + 0.08;

    gain.gain.setValueAtTime(0.0001, start);
    gain.gain.exponentialRampToValueAtTime(0.065, start + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, end);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(start);
    osc.stop(end);
  });
}
