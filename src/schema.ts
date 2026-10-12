import { toJsonSchema } from "@langchain/core/utils/json_schema";
import type { z } from "zod";

/**
 * Length limits keyed by path into the schema: `"summary"` for a string property, `"findings"` for an array
 * (maxItems), `"findings[].issue"` for a property of each array item.
 */
export type SchemaLimits = Record<string, number>;

type JsonSchema = Record<string, any>;

/**
 * JSON schema for `schema` with maxLength / maxItems limits added. Constrained decoders (Ollama / llama.cpp
 * grammars) enforce these, which stops small models from writing one string forever.
 *
 * Do not send it to OpenAI strict mode, which rejects maxLength; `boundedOutput` in the client handles that.
 */
export function boundedJsonSchema(schema: z.ZodType, limits: SchemaLimits): JsonSchema {
  // Deep copy: LangChain caches and shares the converted schema object, so it must not be mutated.
  const root = structuredClone(toJsonSchema(schema)) as JsonSchema;
  for (const [path, limit] of Object.entries(limits)) {
    const node = resolvePath(root, path);
    if (node.type === "array") node.maxItems = limit;
    else if (node.type === "string" || hasStringType(node)) node.maxLength = limit;
    else throw new Error(`Schema limit "${path}": expected a string or array, found ${JSON.stringify(node.type)}`);
  }
  return root;
}

function resolvePath(root: JsonSchema, path: string): JsonSchema {
  let node = root;
  for (const raw of path.split(".")) {
    const isItems = raw.endsWith("[]");
    const key = isItems ? raw.slice(0, -2) : raw;
    const next = unwrapNullable(node).properties?.[key];
    if (!next) throw new Error(`Schema limit "${path}": no property "${key}"`);
    node = isItems ? unwrapNullable(next).items : next;
    if (!node) throw new Error(`Schema limit "${path}": "${key}" is not an array`);
  }
  return unwrapNullable(node);
}

/** `.nullable()` becomes anyOf [T, null]; limits belong on T. */
function unwrapNullable(node: JsonSchema): JsonSchema {
  const variants = node.anyOf as JsonSchema[] | undefined;
  return variants?.find((v) => v.type !== "null") ?? node;
}

function hasStringType(node: JsonSchema): boolean {
  return Array.isArray(node.type) && node.type.includes("string");
}
