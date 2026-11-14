// btc-cent-forecast-01 entry (Node 22, standard library only): a 4-hour-ahead
// forecaster for BTC/USDC 1-minute closes.
//
// Method, tuned on a walk-forward backtest over the kit's six practice cases
// (318 checkpoints, every 30 minutes across the last 30 hours of each case):
//   - work in log-price space and anchor on the checkpoint's last close;
//   - add a shrunk 12-hour trend: the OLS slope of log price over the last
//     720 closes, extrapolated 240 minutes, shrunk by 0.30;
//   - add mean reversion toward the 24-hour mean log price (last 1440
//     closes), shrunk by 0.35;
//   - clamp the total log adjustment to +/-0.01 so a wild extrapolation can
//     never move the guess far from the last close.
// Backtest mean absolute error: 35,570 cents vs 39,903 for persistence
// (the house baseline), about 11% better, and better than persistence in
// all six practice cases individually. Variants tried and set aside on the
// same backtest: median 24h anchor 35,990; EMA anchor 37,269; two-scale
// trend 36,902-38,584; volatility damping 35,451 (a wash); significance-
// weighted trend 35,376-35,397 (inside noise, extra machinery not worth
// it). The score is MAE, so the guess aims at the median of the price 240
// minutes out; the trend and reversion terms are symmetric shrinks, with
// no systematic up or down bias.
//
// Reads every <id>.in in the input directory (name order) and writes one
// <id>.out per case: {"forecasts":[{"case":"<id>","price":<number>}]}.
// Deterministic: no randomness, no clock, no network. Stdlib only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** The largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;

/** Minutes of history the trend slope is fitted on (12 hours). */
const TREND_WINDOW = 720;
/** Shrinkage applied to the extrapolated 240-minute trend drift. */
const TREND_SHRINK = 0.30;
/** Minutes of history the reversion anchor averages over (24 hours). */
const MEAN_WINDOW = 1440;
/** Shrinkage applied to the pull toward the 24-hour mean log price. */
const REVERSION_SHRINK = 0.35;
/** Largest allowed log-price move away from the last close. */
const MAX_LOG_MOVE = 0.01;
/** The forecast horizon in minutes. */
const HORIZON_MINUTES = 240;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number =>
  Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

/**
 * OLS slope (per minute) of values[start..start+w-1] against x = 0..w-1,
 * computed in mean-centred form. The denominator has a closed form:
 * sum((x - mean(x))^2) = (w - 1) * w * (w + 1) / 12.
 */
function olsSlope(values: number[], start: number, w: number): number {
  if (w < 2) return 0;
  let sum = 0;
  for (let i = start; i < start + w; i++) sum += values[i] as number;
  const mean = sum / w;
  let num = 0;
  for (let i = 0; i < w; i++) num += (i - (w - 1) / 2) * ((values[start + i] as number) - mean);
  const den = ((w - 1) * w * (w + 1)) / 12;
  return num / den;
}

/** The forecast for one market: the median guess of the close 240 minutes out. */
function forecast(m: CaseMarket): number {
  if (!Array.isArray(m.prices) || m.prices.length === 0) {
    throw new Error(`market ${m.id} has no prices`);
  }
  const n = m.prices.length;
  const logs: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const p = m.prices[i] as number;
    if (typeof p !== 'number' || !Number.isFinite(p) || p <= 0) {
      throw new Error(`market ${m.id} has a bad price at index ${i}`);
    }
    logs[i] = Math.log(p);
  }
  const last = logs[n - 1] as number;

  // 12-hour trend, extrapolated to the horizon and shrunk toward zero.
  const trendWindow = Math.min(TREND_WINDOW, n);
  const drift = olsSlope(logs, n - trendWindow, trendWindow) * HORIZON_MINUTES;

  // 24-hour mean log price: the level the price reverts toward.
  const meanWindow = Math.min(MEAN_WINDOW, n);
  let meanSum = 0;
  for (let i = n - meanWindow; i < n; i++) meanSum += logs[i] as number;
  const mean = meanSum / meanWindow;

  let adj = TREND_SHRINK * drift + REVERSION_SHRINK * (mean - last);
  if (adj > MAX_LOG_MOVE) adj = MAX_LOG_MOVE;
  if (adj < -MAX_LOG_MOVE) adj = -MAX_LOG_MOVE;
  return price5(Math.exp(last + adj));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.markets) || c.markets.length !== 1) {
    throw new Error(`${file} does not hold exactly one market`);
  }
  writeFileSync(
    join(OUTPUT_DIR, `${id}.out`),
    `${JSON.stringify({ forecasts: [{ case: id, price: forecast(c.markets[0] as CaseMarket) }] })}\n`,
  );
}
