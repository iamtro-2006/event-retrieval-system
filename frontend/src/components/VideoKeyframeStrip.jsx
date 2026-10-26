import { useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Star } from "lucide-react";
import { centeredKeyframeOffset, KEYFRAME_CARD_WIDTH, KEYFRAME_SLOT_WIDTH } from "../utils/videoKeyframes";
import "./VideoKeyframeStrip.css";

export default function VideoKeyframeStrip({ keyframes, activeIndex, onSeek }) {
  const viewport = useRef(null);
  // Keep this component outside React Compiler memoization, as with the surrounding-frame browser.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: keyframes.length,
    horizontal: true,
    getScrollElement: () => viewport.current,
    estimateSize: () => KEYFRAME_SLOT_WIDTH,
    overscan: 4,
  });

  useEffect(() => {
    const node = viewport.current;
    if (!node || activeIndex < 0) return;
    const center = () => {
      virtualizer.scrollToOffset(centeredKeyframeOffset(activeIndex, node.clientWidth, keyframes.length));
    };
    center();
    const observer = new ResizeObserver(center);
    observer.observe(node);
    return () => observer.disconnect();
  }, [activeIndex, keyframes.length, virtualizer]);

  function seekFrame(index) {
    if (!keyframes[index]) return;
    virtualizer.scrollToOffset(centeredKeyframeOffset(index, viewport.current?.clientWidth || 0, keyframes.length));
    onSeek(keyframes[index].timestamp * 1000);
  }

  return <div className="video-keyframe-viewport" ref={viewport} role="group" aria-label="Video keyframes" tabIndex={0}
    onKeyDown={(event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      event.currentTarget.focus({ preventScroll: true });
      const index = Math.max(0, Math.min(keyframes.length - 1, activeIndex + (event.key === "ArrowRight" ? 1 : -1)));
      seekFrame(index);
    }}>
    <div className="video-keyframe-spacer" style={{ width: Math.max(0, virtualizer.getTotalSize() - 8) }}>
      {virtualizer.getVirtualItems().map(({ index, start }) => {
        const frame = keyframes[index];
        const active = index === activeIndex;
        const label = frame.keyframe_id.padStart(6, "0");
        return <button key={frame.keyframe_id} type="button" className={`video-keyframe-card${active ? " is-active" : ""}`}
          style={{ width: KEYFRAME_CARD_WIDTH, transform: `translateX(${start}px)` }}
          aria-label={`Keyframe ${label} at ${frame.timestamp.toFixed(3)} seconds`}
          aria-current={active ? "true" : undefined} tabIndex={active ? 0 : -1}
          title={`Keyframe ${label} · ${frame.timestamp.toFixed(3)}s`}
          onClick={() => seekFrame(index)}>
          {frame.image_url && frame.image_url !== "#" ? <img src={frame.image_url} alt={`Keyframe ${label}`} loading="lazy" decoding="async" />
            : <div className="video-keyframe-no-image">No image</div>}
          {active && <Star className="video-keyframe-star" size={15} fill="currentColor" aria-hidden="true" />}
          <span className="video-keyframe-id">{label}</span>
        </button>;
      })}
    </div>
  </div>;
}
