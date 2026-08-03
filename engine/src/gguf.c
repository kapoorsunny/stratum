/* See gguf.h. */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u   /* "GGUF" little endian */

/* Block layout for every element type, taken from the ggml format.
 * block_elems is how many values one block holds and block_bytes is what
 * that block costs on disk. Unknown types carry zeroes and are rejected. */
typedef struct { const char *name; uint32_t block_elems, block_bytes; } TypeInfo;

static const TypeInfo TYPES[GG_TYPE_MAX] = {
    [GG_F32]  = { "f32",  1,  4 },
    [GG_F16]  = { "f16",  1,  2 },
    [GG_Q4_0] = { "q4_0", 32, 18 },
    [GG_Q4_1] = { "q4_1", 32, 20 },
    [GG_Q5_0] = { "q5_0", 32, 22 },
    [GG_Q5_1] = { "q5_1", 32, 24 },
    [GG_Q8_0] = { "q8_0", 32, 34 },
    [GG_Q8_1] = { "q8_1", 32, 36 },
    [GG_Q2_K] = { "q2_K", 256, 84 },
    [GG_Q3_K] = { "q3_K", 256, 110 },
    [GG_Q4_K] = { "q4_K", 256, 144 },
    [GG_Q5_K] = { "q5_K", 256, 176 },
    [GG_Q6_K] = { "q6_K", 256, 210 },
    [GG_Q8_K] = { "q8_K", 256, 292 },
};

const char *gguf_type_name(uint32_t type)
{
    if (type < GG_TYPE_MAX && TYPES[type].name) return TYPES[type].name;
    return "unknown";
}

uint64_t gguf_type_nbytes(uint32_t type, uint64_t n)
{
    if (type >= GG_TYPE_MAX || !TYPES[type].name) return 0;
    const uint32_t be = TYPES[type].block_elems;
    const uint32_t bb = TYPES[type].block_bytes;
    if (n % be) return 0;              /* a partial block cannot exist on disk */
    const uint64_t blocks = n / be;
    if (blocks > UINT64_MAX / bb) return 0;
    return blocks * bb;
}

/* A cursor over the mapping that refuses to walk off the end.
 * Every read checks first and sets bad, so one test at the end of parsing
 * covers every field rather than each one needing its own branch. */
typedef struct {
    const unsigned char *p, *end;
    int bad;
} Cur;

static uint32_t rd_u32(Cur *c)
{
    if (c->bad || c->end - c->p < 4) { c->bad = 1; return 0; }
    uint32_t v;
    memcpy(&v, c->p, 4);
    c->p += 4;
    return v;
}

static uint64_t rd_u64(Cur *c)
{
    if (c->bad || c->end - c->p < 8) { c->bad = 1; return 0; }
    uint64_t v;
    memcpy(&v, c->p, 8);
    c->p += 8;
    return v;
}

/* Strings are a length then that many bytes, with no terminator.
 * The bytes stay in the mapping and are never copied. */
static const char *rd_str(Cur *c, uint32_t *len_out)
{
    const uint64_t n = rd_u64(c);
    if (c->bad) return NULL;
    /* A length past the end of the file means the header is corrupt. */
    if (n > (uint64_t)(c->end - c->p)) { c->bad = 1; return NULL; }
    const char *s = (const char *)c->p;
    c->p += n;
    *len_out = (uint32_t)n;
    return s;
}

/* Fixed width of a scalar metadata value, or 0 when it is not scalar. */
static uint32_t scalar_size(uint32_t t)
{
    switch (t) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
    case GGUF_U16: case GGUF_I16: return 2;
    case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
    case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
    default: return 0;
    }
}

/* Step over one metadata value, recording where it started.
 * Arrays are walked rather than parsed because the engine only ever reads
 * scalars and strings, and skipping keeps a tokenizer array from costing
 * anything. */
static void skip_value(Cur *c, uint32_t type, const unsigned char **val,
                       uint64_t *val_len, int depth)
{
    *val = c->p;

    /* An array of arrays of arrays is not something any real checkpoint
     * contains, so a cap here turns a malicious file into an error rather
     * than a blown stack. */
    if (depth > 4) { c->bad = 1; *val_len = 0; return; }

    if (type == GGUF_STRING) {
        uint32_t len = 0;
        const char *s = rd_str(c, &len);
        *val = (const unsigned char *)s;
        *val_len = len;
        return;
    }

    if (type == GGUF_ARRAY) {
        const uint32_t elem_type = rd_u32(c);
        const uint64_t n = rd_u64(c);
        if (c->bad) { *val_len = 0; return; }
        const uint32_t w = scalar_size(elem_type);
        if (w) {
            /* Overflow first, then bounds. Doing it the other way round can
             * wrap and pass a check it should have failed. */
            if (n > (uint64_t)(c->end - c->p) / w) { c->bad = 1; *val_len = 0; return; }
            c->p += n * w;
        } else {
            for (uint64_t i = 0; i < n && !c->bad; i++) {
                const unsigned char *v; uint64_t vl;
                skip_value(c, elem_type, &v, &vl, depth + 1);
            }
        }
        *val_len = (uint64_t)(c->p - *val);
        return;
    }

    const uint32_t w = scalar_size(type);
    if (!w) { c->bad = 1; *val_len = 0; return; }
    if ((uint64_t)(c->end - c->p) < w) { c->bad = 1; *val_len = 0; return; }
    c->p += w;
    *val_len = w;
}

static int fail(GgufFile *g, const char *what)
{
    snprintf(g->err, sizeof g->err, "%s", what);
    free(g->tensors);
    free(g->kv);
    g->tensors = NULL;
    g->kv = NULL;
    sm_map_close(&g->map);
    return 1;
}

int gguf_open(GgufFile *g, const char *path)
{
    memset(g, 0, sizeof *g);

    if (sm_map_open(&g->map, path, g->err, sizeof g->err) != 0) return 1;

    Cur c = { g->map.base, g->map.base + g->map.size, 0 };

    if (rd_u32(&c) != GGUF_MAGIC)
        return fail(g, "not a GGUF file, the magic bytes are wrong");

    g->version = rd_u32(&c);
    if (g->version < 2 || g->version > 3) {
        char m[128];
        snprintf(m, sizeof m, "GGUF version %u is not supported, this reads 2 and 3",
                 g->version);
        return fail(g, m);
    }

    g->n_tensors = rd_u64(&c);
    g->n_kv = rd_u64(&c);
    if (c.bad) return fail(g, "the header is truncated");

    /* Each tensor entry costs at least 32 bytes and each key value at least
     * 16, so a count larger than the file could hold is corrupt. Checking
     * here means the allocations below cannot be tricked into being huge. */
    if (g->n_tensors > g->map.size / 32 || g->n_kv > g->map.size / 16)
        return fail(g, "the header claims more entries than the file can hold");

    if (g->n_kv) {
        g->kv = (GgufKV *)calloc((size_t)g->n_kv, sizeof *g->kv);
        if (!g->kv) return fail(g, "out of memory reading metadata");
    }
    for (uint64_t i = 0; i < g->n_kv; i++) {
        g->kv[i].key = rd_str(&c, &g->kv[i].key_len);
        g->kv[i].type = rd_u32(&c);
        if (c.bad) return fail(g, "the metadata block is truncated");
        skip_value(&c, g->kv[i].type, &g->kv[i].val, &g->kv[i].val_len, 0);
        if (c.bad) return fail(g, "a metadata value is malformed");
    }

    if (g->n_tensors) {
        g->tensors = (GgufTensor *)calloc((size_t)g->n_tensors, sizeof *g->tensors);
        if (!g->tensors) return fail(g, "out of memory reading the tensor directory");
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        GgufTensor *t = &g->tensors[i];
        t->name = rd_str(&c, &t->name_len);
        t->n_dims = rd_u32(&c);
        if (c.bad) return fail(g, "the tensor directory is truncated");
        if (t->n_dims == 0 || t->n_dims > GGUF_MAX_DIMS)
            return fail(g, "a tensor claims an impossible number of dimensions");

        uint64_t n_elem = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = rd_u64(&c);
            if (t->dims[d] == 0) return fail(g, "a tensor has a zero dimension");
            if (t->dims[d] > UINT64_MAX / n_elem)
                return fail(g, "a tensor element count overflows");
            n_elem *= t->dims[d];
        }
        t->type = rd_u32(&c);
        t->offset = rd_u64(&c);
        if (c.bad) return fail(g, "the tensor directory is truncated");

        t->nbytes = gguf_type_nbytes(t->type, n_elem);
        if (!t->nbytes) {
            char m[160];
            snprintf(m, sizeof m,
                     "tensor %u uses type %u (%s), which this engine cannot size",
                     (unsigned)i, t->type, gguf_type_name(t->type));
            return fail(g, m);
        }
    }

    /* Tensor data starts at the next alignment boundary after the directory.
     * The alignment is a metadata key and defaults to 32. */
    uint64_t align = 32;
    (void)gguf_kv_u64(g, "general.alignment", &align);
    if (align == 0 || (align & (align - 1)))
        return fail(g, "general.alignment is not a power of two");

    uint64_t off = (uint64_t)(c.p - g->map.base);
    const uint64_t rem = off % align;
    if (rem) off += align - rem;
    if (off > g->map.size) return fail(g, "the tensor data starts past the end of the file");
    g->data_offset = off;

    /* Every tensor must fit inside the file. Checking now means the rest of
     * the engine can trust a pointer instead of testing every access. */
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const GgufTensor *t = &g->tensors[i];
        if (t->offset > g->map.size - g->data_offset ||
            t->nbytes > g->map.size - g->data_offset - t->offset) {
            char m[192];
            snprintf(m, sizeof m,
                     "tensor %.*s runs past the end of the file, the checkpoint "
                     "is truncated or corrupt",
                     (int)t->name_len, t->name ? t->name : "");
            return fail(g, m);
        }
    }

    return 0;
}

void gguf_close(GgufFile *g)
{
    if (!g) return;
    free(g->tensors);
    free(g->kv);
    g->tensors = NULL;
    g->kv = NULL;
    sm_map_close(&g->map);
}

/* Names in the mapping are not terminated, so comparison is by length. */
static int name_eq(const char *p, uint32_t len, const char *want)
{
    const size_t n = strlen(want);
    return n == len && memcmp(p, want, n) == 0;
}

const GgufTensor *gguf_find(const GgufFile *g, const char *name)
{
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].name && name_eq(g->tensors[i].name, g->tensors[i].name_len, name))
            return &g->tensors[i];
    return NULL;
}

uint64_t gguf_tensor_file_offset(const GgufFile *g, const GgufTensor *t)
{
    return g->data_offset + t->offset;
}

const unsigned char *gguf_tensor_data(const GgufFile *g, const GgufTensor *t)
{
    if (!g || !t) return NULL;
    const uint64_t off = gguf_tensor_file_offset(g, t);
    if (off > g->map.size || t->nbytes > g->map.size - off) return NULL;
    return g->map.base + off;
}

static const GgufKV *find_kv(const GgufFile *g, const char *key)
{
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key && name_eq(g->kv[i].key, g->kv[i].key_len, key))
            return &g->kv[i];
    return NULL;
}

int gguf_kv_u64(const GgufFile *g, const char *key, uint64_t *out)
{
    const GgufKV *k = find_kv(g, key);
    if (!k || !k->val) return 1;

    /* Every width is widened into u64 so callers asking for a model
     * dimension do not care how the writer stored it. */
    switch (k->type) {
    case GGUF_U8:  { uint8_t v;  memcpy(&v, k->val, 1); *out = v; return 0; }
    case GGUF_I8:  { int8_t v;   memcpy(&v, k->val, 1); if (v < 0) return 1; *out = (uint64_t)v; return 0; }
    case GGUF_U16: { uint16_t v; memcpy(&v, k->val, 2); *out = v; return 0; }
    case GGUF_I16: { int16_t v;  memcpy(&v, k->val, 2); if (v < 0) return 1; *out = (uint64_t)v; return 0; }
    case GGUF_U32: { uint32_t v; memcpy(&v, k->val, 4); *out = v; return 0; }
    case GGUF_I32: { int32_t v;  memcpy(&v, k->val, 4); if (v < 0) return 1; *out = (uint64_t)v; return 0; }
    case GGUF_U64: { uint64_t v; memcpy(&v, k->val, 8); *out = v; return 0; }
    case GGUF_I64: { int64_t v;  memcpy(&v, k->val, 8); if (v < 0) return 1; *out = (uint64_t)v; return 0; }
    default: return 1;
    }
}

int gguf_kv_str(const GgufFile *g, const char *key, char *buf, size_t buflen)
{
    const GgufKV *k = find_kv(g, key);
    if (!k || k->type != GGUF_STRING || !k->val || !buflen) return 1;
    size_t n = (size_t)k->val_len;
    if (n > buflen - 1) n = buflen - 1;
    memcpy(buf, k->val, n);
    buf[n] = 0;
    return 0;
}
