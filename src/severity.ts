import { z } from "zod";

/** One severity scale for every tool built on llm-kit, least to most severe. */
export const Severity = z.enum(["low", "medium", "high", "critical"]);
export type Severity = z.infer<typeof Severity>;

export const SEVERITY_ORDER: Readonly<Record<Severity, number>> = { low: 0, medium: 1, high: 2, critical: 3 };

/** Sort comparator: most severe first. */
export function compareSeverity(a: Severity, b: Severity): number {
  return SEVERITY_ORDER[b] - SEVERITY_ORDER[a];
}

/** The most severe of `levels`, never below `floor`. */
export function maxSeverity(levels: Iterable<Severity>, floor: Severity = "low"): Severity {
  let max = floor;
  for (const l of levels) if (SEVERITY_ORDER[l] > SEVERITY_ORDER[max]) max = l;
  return max;
}

/** A `--fail-on` threshold: a severity, or "none" to never fail. */
export const FailOn = z.enum(["none", ...Severity.options]);
export type FailOn = z.infer<typeof FailOn>;

/** True when `level` reaches the `failOn` threshold. */
export function meetsThreshold(level: Severity, failOn: FailOn): boolean {
  return failOn !== "none" && SEVERITY_ORDER[level] >= SEVERITY_ORDER[failOn];
}

/** Parse a user-supplied fail-on value (case-insensitive) with an error that lists the valid choices. */
export function parseFailOn(value: string): FailOn {
  const v = value.trim().toLowerCase();
  const parsed = FailOn.safeParse(v);
  if (!parsed.success) throw new Error(`Invalid fail-on value "${value}". Use one of: ${FailOn.options.join(", ")}.`);
  return parsed.data;
}
