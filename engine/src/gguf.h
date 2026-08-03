/* GGUF checkpoint reader.
 *
 * GGUF is the single file format llama.cpp writes, and it is what the rest
 * of the world quantizes models into. A file is a header, a table of key
 * value metadata, a directory of tensors, then the tensor bytes.
 *
 * Nothing here copies weights. Parsing walks the directory and records where
 * each tensor lives, so opening a 1.5 TB checkpoint costs the same as opening
 * a small one.
 *
 * The parser treats the file as hostile. A truncated or corrupt checkpoint
 * must produce an error, never a read past the end of the mapping.
 */
#ifndef STRATUM_GGUF_H
#define STRATUM_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "map.h"

/* Metadata value types, in the order the format defines them. */
enum {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3,
    GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_U64 = 10, GGUF_I64 = 11,
    GGUF_F64 = 12,
    GGUF_TYPE_COUNT
};

/* Tensor element types. Only the ones this engine can read are named, but
 * every type has a size so the directory of any checkpoint parses. */
enum {
    GG_F32 = 0, GG_F16 = 1, GG_Q4_0 = 2, GG_Q4_1 = 3,
    GG_Q5_0 = 6, GG_Q5_1 = 7, GG_Q8_0 = 8, GG_Q8_1 = 9,
    GG_Q2_K = 10, GG_Q3_K = 11, GG_Q4_K = 12, GG_Q5_K = 13,
    GG_Q6_K = 14, GG_Q8_K = 15,
    GG_TYPE_MAX = 40
};

#define GGUF_MAX_DIMS 4

typedef struct {
    const char *name;       /* points into the mapping, not owned, not terminated */
    uint32_t name_len;
    uint32_t n_dims;
    uint64_t dims[GGUF_MAX_DIMS];
    uint32_t type;          /* one of the GG_ values */
    uint64_t offset;        /* from the start of the tensor data block */
    uint64_t nbytes;        /* size on disk, worked out from type and dims */
} GgufTensor;

typedef struct {
    const char *key;
    uint32_t key_len;
    uint32_t type;          /* one of the GGUF_ values */
    const unsigned char *val;   /* raw bytes, meaning depends on type */
    uint64_t val_len;
} GgufKV;

typedef struct {
    SmMap map;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    GgufTensor *tensors;    /* n_tensors entries */
    GgufKV *kv;             /* n_kv entries */
    uint64_t data_offset;   /* where tensor bytes start, from the file start */
    char err[512];
} GgufFile;

/* Open and parse a checkpoint. Returns 0 on success.
 * On failure g->err says what was wrong and nothing needs freeing. */
int gguf_open(GgufFile *g, const char *path);

void gguf_close(GgufFile *g);

/* Find a tensor by exact name, or NULL. */
const GgufTensor *gguf_find(const GgufFile *g, const char *name);

/* Where a tensor's bytes are, measured from the start of the file. */
uint64_t gguf_tensor_file_offset(const GgufFile *g, const GgufTensor *t);

/* Pointer to a tensor's first byte inside the mapping, or NULL when the
 * tensor does not fit the file. Touching this pointer is what pages the
 * weights in. */
const unsigned char *gguf_tensor_data(const GgufFile *g, const GgufTensor *t);

/* Read a metadata value, returning 0 when the key exists and has a type that
 * converts. Integers of any width convert into u64, which is what the model
 * shape keys need. */
int gguf_kv_u64(const GgufFile *g, const char *key, uint64_t *out);

/* Copy a string valued key into buf, always terminated. Returns 0 on
 * success. */
int gguf_kv_str(const GgufFile *g, const char *key, char *buf, size_t buflen);

/* Bytes on disk for a tensor of this type and element count, or 0 when the
 * type is unknown or the count does not divide the block size. */
uint64_t gguf_type_nbytes(uint32_t type, uint64_t n_elements);

/* Human readable type name, never NULL. */
const char *gguf_type_name(uint32_t type);

#endif
