import { describe, expect, it } from "vitest";
import { formatDuration, normalizeScores } from "./utils";

describe("formatDuration", () => {
  it("formats minutes and seconds", () => {
    expect(formatDuration(125)).toBe("2:05");
  });

  it("returns null for falsy input", () => {
    expect(formatDuration(0)).toBeNull();
  });
});

describe("normalizeScores", () => {
  it("maps scores to 0–1 range", () => {
    const out = normalizeScores([
      { video_id: "a", score: 1 },
      { video_id: "b", score: 3 },
    ]);
    expect(out[0].norm).toBe(0);
    expect(out[1].norm).toBe(1);
  });

  it("returns empty array for no videos", () => {
    expect(normalizeScores([])).toEqual([]);
  });
});
