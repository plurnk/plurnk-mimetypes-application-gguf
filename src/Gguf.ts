import { BaseHandler } from "@plurnk/plurnk-mimetypes";
import type { HandlerContent, MimeSymbol } from "@plurnk/plurnk-mimetypes";

// application/x-gguf (GGUF local-LLM model) handler — Tier 4 binary, no dep.
//
// GGUF (llama.cpp / Ollama) packs a model's weights behind a structured
// key-value metadata HEADER: architecture, quantization, context length,
// embedding length, tokenizer, chat template, … This handler reads ONLY that
// header — never the gigabytes of tensor data — so an agent in a model folder
// can answer "what model is this, what quant, what context window" for free.
//
// Symbols are the metadata keys (`general.architecture`, `llama.context_length`,
// …), in file order. deepJson is `{ version, tensorCount, metadata }` — a
// jsonpath target. toText renders the metadata as a `key: value` table — the
// readable projection that backs regex/glob and the embedding. Large arrays
// (tokenizer vocab) are summarized as `<type[N]>`, never expanded. Per-tensor
// inventory is out of scope for v1 (the metadata is the "what is this" answer).
export default class Gguf extends BaseHandler {
    override extractRaw(content: HandlerContent): MimeSymbol[] {
        const header = readGguf(toBytes(content));
        if (!header) return [];
        return header.order.map((key, i) => ({ name: key, kind: "field", line: i + 1, endLine: i + 1 }));
    }

    override deepJson(content: HandlerContent): unknown {
        const header = readGguf(toBytes(content));
        if (!header) return null;
        return { version: header.version, tensorCount: header.tensorCount, metadata: header.metadata };
    }

    override extent(content: HandlerContent): number {
        const header = readGguf(toBytes(content));
        return header ? header.order.length : 0;
    }

    override validate(content: HandlerContent): void {
        if (!readGguf(toBytes(content))) throw new Error("not a valid GGUF file (bad magic or truncated header)");
    }

    protected override toText(content: HandlerContent): string {
        const header = readGguf(toBytes(content));
        if (!header) return "";
        const lines = header.order.map((k) => `${k}: ${renderValue(header.metadata[k])}`);
        lines.push(`tensors: ${header.tensorCount}`);
        return lines.join("\n");
    }
}

export type GgufValue = number | boolean | string | { array: string; length: number };

export interface GgufHeader {
    version: number;
    tensorCount: number;
    metadata: Record<string, GgufValue>;
    order: string[];
}

const TYPE_NAMES: Record<number, string> = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
};

// Parse the GGUF metadata header, or null on anything that isn't one (bad
// magic, truncated). Little-endian throughout; uint64 counts/lengths are small
// enough to fold to Number.
export function readGguf(bytes: Uint8Array): GgufHeader | null {
    if (bytes.length < 24) return null;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    // "GGUF" read big-endian == 0x47475546.
    if (dv.getUint32(0, false) !== 0x47475546) return null;

    let off = 4;
    const need = (n: number): void => {
        if (off + n > bytes.length) throw new RangeError("truncated GGUF header");
    };
    const dec = new TextDecoder();

    const readString = (): string => {
        need(8);
        const len = Number(dv.getBigUint64(off, true)); off += 8;
        need(len);
        const s = dec.decode(bytes.subarray(off, off + len)); off += len;
        return s;
    };
    const readScalar = (type: number): number | boolean | string => {
        switch (type) {
            case 0: { need(1); const v = dv.getUint8(off); off += 1; return v; }
            case 1: { need(1); const v = dv.getInt8(off); off += 1; return v; }
            case 2: { need(2); const v = dv.getUint16(off, true); off += 2; return v; }
            case 3: { need(2); const v = dv.getInt16(off, true); off += 2; return v; }
            case 4: { need(4); const v = dv.getUint32(off, true); off += 4; return v; }
            case 5: { need(4); const v = dv.getInt32(off, true); off += 4; return v; }
            case 6: { need(4); const v = dv.getFloat32(off, true); off += 4; return v; }
            case 7: { need(1); const v = dv.getUint8(off) !== 0; off += 1; return v; }
            case 8: return readString();
            case 10: { need(8); const v = Number(dv.getBigUint64(off, true)); off += 8; return v; }
            case 11: { need(8); const v = Number(dv.getBigInt64(off, true)); off += 8; return v; }
            case 12: { need(8); const v = dv.getFloat64(off, true); off += 8; return v; }
            default: throw new RangeError(`unknown GGUF value type ${type}`);
        }
    };
    const readValue = (type: number): GgufValue => {
        if (type !== 9) return readScalar(type);
        need(12);
        const elemType = dv.getUint32(off, true); off += 4;
        const len = Number(dv.getBigUint64(off, true)); off += 8;
        // Walk (advance past) every element but store only a summary — token
        // vocabularies are huge and are not symbols.
        for (let i = 0; i < len; i += 1) readValue(elemType);
        return { array: TYPE_NAMES[elemType] ?? `type${elemType}`, length: len };
    };

    try {
        const version = dv.getUint32(off, true); off += 4;
        const tensorCount = Number(dv.getBigUint64(off, true)); off += 8;
        const kvCount = Number(dv.getBigUint64(off, true)); off += 8;
        const metadata: Record<string, GgufValue> = {};
        const order: string[] = [];
        for (let i = 0; i < kvCount; i += 1) {
            const key = readString();
            need(4);
            const vtype = dv.getUint32(off, true); off += 4;
            const value = readValue(vtype);
            if (!(key in metadata)) order.push(key);
            metadata[key] = value;
        }
        return { version, tensorCount, metadata, order };
    } catch {
        return null;
    }
}

function renderValue(value: GgufValue): string {
    if (typeof value === "object") return `<${value.array}[${value.length}]>`;
    return String(value);
}

function toBytes(content: HandlerContent): Uint8Array {
    return typeof content === "string" ? new TextEncoder().encode(content) : content;
}
