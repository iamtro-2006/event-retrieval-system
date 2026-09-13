export const COLOR_PALETTE = [
  ["black", "#000000"], ["blue", "#235ac8"], ["brown", "#82502d"],
  ["grey", "#808080"], ["green", "#1e963c"], ["orange", "#f58c14"],
  ["pink", "#f582b4"], ["purple", "#8246b4"], ["red", "#d22828"],
  ["white", "#f5f5f5"], ["yellow", "#f5dc28"],
];

export const EMPTY_COLOR_GRID = Object.freeze(Array(25).fill(null));

export function selectedColorCells(grid) {
  return grid.flatMap((color, index) => color ? [{ row: Math.floor(index / 5), col: index % 5, color }] : []);
}
