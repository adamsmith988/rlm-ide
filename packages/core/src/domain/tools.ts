/** `tool.json`-style descriptor. Tool behaviour comes from this, never from prompt text. */
export interface JsonSchema {
  type?: "object" | "string" | "integer" | "number" | "boolean" | "array";
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  description?: string;
  default?: unknown;
}

export interface ToolDescriptor {
  /** Dotted id: `<namespace>.<name>`, e.g. `fs.read_file`. The namespace becomes the kernel object. */
  id: string;
  description: string;
  parameters: JsonSchema;
  /** Short description of the return value shown in the tool surface. */
  returns: string;
  sideEffects: boolean;
  timeoutSec: number;
  /** Tool group used for per-chat toggles (fs, search, git, shell, editor, session). */
  group: string;
  /** True for file-editing tools whose approval card shows a diff preview. */
  editsFiles?: boolean;
  /** Identical back-to-back calls are intentional (clicks, key presses) and must not be deduplicated. */
  repeatable?: boolean;
}

export class ToolArgError extends Error {}

/** Validates and coerces args against the descriptor schema (subset of JSON Schema). */
export function validateArgs(schema: JsonSchema, raw: unknown): Record<string, unknown> {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new ToolArgError("arguments must be an object of named parameters");
  }
  const args: Record<string, unknown> = { ...(raw as Record<string, unknown>) };
  const props = schema.properties ?? {};
  for (const name of schema.required ?? []) {
    if (args[name] === undefined || args[name] === null) throw new ToolArgError(`missing required argument '${name}'`);
  }
  for (const key of Object.keys(args)) {
    if (!(key in props)) throw new ToolArgError(`unknown argument '${key}'. Allowed: ${Object.keys(props).join(", ") || "(none)"}`);
  }
  for (const [name, p] of Object.entries(props)) {
    let v = args[name];
    if (v === undefined) {
      if (p.default !== undefined) args[name] = p.default;
      continue;
    }
    if ((p.type === "integer" || p.type === "number") && typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
      v = Number(v);
      args[name] = v;
    }
    if (p.type === "boolean" && typeof v === "string" && (v === "true" || v === "false")) {
      v = v === "true";
      args[name] = v;
    }
    const ok =
      p.type === undefined ||
      (p.type === "string" && typeof v === "string") ||
      (p.type === "integer" && Number.isInteger(v)) ||
      (p.type === "number" && typeof v === "number") ||
      (p.type === "boolean" && typeof v === "boolean") ||
      (p.type === "array" && Array.isArray(v)) ||
      (p.type === "object" && typeof v === "object" && v !== null && !Array.isArray(v));
    if (!ok) throw new ToolArgError(`argument '${name}' must be ${p.type}`);
    if (p.enum && !p.enum.includes(v)) throw new ToolArgError(`argument '${name}' must be one of ${JSON.stringify(p.enum)}`);
  }
  return args;
}

/** Stable key for per-turn de-duplication. */
export function callKey(toolId: string, args: Record<string, unknown>): string {
  const sort = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(sort);
    if (v && typeof v === "object") {
      return Object.fromEntries(Object.keys(v as object).sort().map((k) => [k, sort((v as Record<string, unknown>)[k])]));
    }
    return v;
  };
  return `${toolId}:${JSON.stringify(sort(args))}`;
}

/** Python signature for the tool surface, e.g. `fs.read_file(path: str, start_line: int = 1)`. */
export function pythonSignature(d: ToolDescriptor): string {
  const props = d.parameters.properties ?? {};
  const required = new Set(d.parameters.required ?? []);
  const pyType = (s: JsonSchema): string =>
    ({ string: "str", integer: "int", number: "float", boolean: "bool", array: "list", object: "dict" })[s.type ?? "string"] ?? "Any";
  const parts = Object.entries(props)
    .sort(([a], [b]) => Number(required.has(b)) - Number(required.has(a)))
    .map(([name, s]) => {
      if (required.has(name)) return `${name}: ${pyType(s)}`;
      const def = s.default === undefined ? "None" : JSON.stringify(s.default).replace(/^true$/, "True").replace(/^false$/, "False");
      return `${name}: ${pyType(s)} = ${def}`;
    });
  return `${d.id}(${parts.join(", ")})`;
}
