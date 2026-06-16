import { describe, it } from "node:test";
import assert from "node:assert/strict";
import Gguf, { readGguf } from "./Gguf.ts";

const META = { mimetype: "application/x-gguf", glyph: "🧠", extensions: [".gguf"] };
const h = () => new Gguf(META);

// ——— minimal GGUF byte builder (little-endian) ———
const enc = new TextEncoder();
const u32 = (n: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
const u64 = (n: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(n), true); return b; };
const f32 = (n: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, n, true); return b; };
const gstr = (s: string) => { const e = enc.encode(s); return concat(u64(e.length), e); };
function concat(...parts: Uint8Array[]): Uint8Array {
    const total = parts.reduce((n, p) => n + p.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
}

// types: 4=uint32, 6=float32, 7=bool, 8=string, 9=array
function buildGguf(): Uint8Array {
    const parts: Uint8Array[] = [];
    parts.push(new Uint8Array([0x47, 0x47, 0x55, 0x46])); // "GGUF"
    parts.push(u32(3));   // version
    parts.push(u64(291)); // tensor_count
    parts.push(u64(4));   // kv_count
    // general.architecture = "llama"
    parts.push(gstr("general.architecture"), u32(8), gstr("llama"));
    // llama.context_length = 4096 (uint32)
    parts.push(gstr("llama.context_length"), u32(4), u32(4096));
    // general.file_type = 7.0 (float32, stands in for a quant marker)
    parts.push(gstr("general.quantization_version"), u32(6), f32(2));
    // tokenizer.ggml.tokens = array<string>[3]  (must be SUMMARIZED, not expanded)
    parts.push(gstr("tokenizer.ggml.tokens"), u32(9), u32(8), u64(3), gstr("a"), gstr("b"), gstr("c"));
    return concat(...parts);
}

const GGUF = buildGguf();

describe("Gguf — metadata header parse", () => {
    it("reads version, tensor count, and scalar metadata", () => {
        const hdr = readGguf(GGUF)!;
        assert.equal(hdr.version, 3);
        assert.equal(hdr.tensorCount, 291);
        assert.equal(hdr.metadata["general.architecture"], "llama");
        assert.equal(hdr.metadata["llama.context_length"], 4096);
    });

    it("summarizes arrays instead of expanding them", () => {
        const hdr = readGguf(GGUF)!;
        assert.deepEqual(hdr.metadata["tokenizer.ggml.tokens"], { array: "string", length: 3 });
    });

    it("rejects non-GGUF bytes", () => {
        assert.equal(readGguf(new TextEncoder().encode("not a model")), null);
    });
});

describe("Gguf — channels", () => {
    it("symbols are the metadata keys in order", () => {
        assert.deepEqual(h().extractRaw(GGUF).map((s) => s.name), [
            "general.architecture", "llama.context_length", "general.quantization_version", "tokenizer.ggml.tokens",
        ]);
    });

    it("deepJson carries version/tensorCount/metadata", () => {
        const tree = h().deepJson(GGUF) as { tensorCount: number; metadata: Record<string, unknown> };
        assert.equal(tree.tensorCount, 291);
        assert.equal(tree.metadata["general.architecture"], "llama");
    });

    it("toText renders a readable metadata table (embed-source)", async () => {
        // toText is protected; reach it through the regex query path.
        const matches = await h().query(GGUF, "regex", "general\\.architecture: (\\w+)");
        assert.equal((matches[0]?.matched as string[])[0], "llama");
    });

    it("validate throws on bad magic; other channels degrade to empty", () => {
        const bad = new TextEncoder().encode("nope");
        assert.throws(() => h().validate(bad));
        assert.deepEqual(h().extractRaw(bad), []);
        assert.equal(h().deepJson(bad), null);
        assert.doesNotThrow(() => h().validate(GGUF));
    });
});
