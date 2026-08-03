/* Tests for the GGUF parser.
 *
 * A checkpoint arrives over the network from somewhere else, so the parser
 * has to survive a bad one. Most of these tests build a deliberately broken
 * file and check that the reader says no instead of reading past the end of
 * the mapping.
 *
 * Every test writes a real file and opens it through the real code path,
 * because the bugs worth catching here live in the arithmetic between the
 * header and the mapping, and a mock would not have any.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"

static int failures = 0;
static int checks = 0;

static void check(int cond, const char *what)
{
    checks++;
    if (!cond) {
        failures++;
        printf("  FAIL %s\n", what);
    }
}

/* A growable buffer, so a test can describe a file rather than compute
 * offsets by hand. */
typedef struct { unsigned char *p; size_t len, cap; } Buf;

static void put(Buf *b, const void *src, size_t n)
{
    if (b->len + n > b->cap) {
        b->cap = (b->len + n) * 2 + 256;
        b->p = (unsigned char *)realloc(b->p, b->cap);
    }
    memcpy(b->p + b->len, src, n);
    b->len += n;
}

static void put_u32(Buf *b, uint32_t v) { put(b, &v, 4); }
static void put_u64(Buf *b, uint64_t v) { put(b, &v, 8); }

static void put_str(Buf *b, const char *s)
{
    const uint64_t n = strlen(s);
    put_u64(b, n);
    put(b, s, (size_t)n);
}

static const char *TMP = "test_gguf_tmp.bin";

static void write_file(const Buf *b)
{
    FILE *f = fopen(TMP, "wb");
    if (!f) { printf("  cannot write %s\n", TMP); exit(2); }
    fwrite(b->p, 1, b->len, f);
    fclose(f);
}

/* A minimal but valid file: one f32 tensor of 8 elements and two keys. */
static void build_good(Buf *b, uint64_t n_tensors)
{
    memset(b, 0, sizeof *b);
    put_u32(b, 0x46554747u);       /* GGUF */
    put_u32(b, 3);                 /* version */
    put_u64(b, n_tensors);
    put_u64(b, 2);                 /* metadata keys */

    put_str(b, "general.architecture");
    put_u32(b, GGUF_STRING);
    put_str(b, "testarch");

    put_str(b, "testarch.block_count");
    put_u32(b, GGUF_U32);
    put_u32(b, 12);

    for (uint64_t i = 0; i < n_tensors; i++) {
        char nm[64];
        snprintf(nm, sizeof nm, "blk.%llu.weight", (unsigned long long)i);
        put_str(b, nm);
        put_u32(b, 1);             /* dims */
        put_u64(b, 8);             /* 8 elements */
        put_u32(b, GG_F32);
        put_u64(b, i * 32);        /* offset inside the data block */
    }

    /* Pad to the default 32 byte alignment, then the tensor bytes. */
    while (b->len % 32) { unsigned char z = 0; put(b, &z, 1); }
    for (uint64_t i = 0; i < n_tensors * 32; i++) {
        unsigned char v = (unsigned char)i;
        put(b, &v, 1);
    }
}

static void test_reads_a_good_file(void)
{
    printf("a good file parses\n");
    Buf b;
    build_good(&b, 3);
    write_file(&b);
    free(b.p);

    GgufFile g;
    check(gguf_open(&g, TMP) == 0, "opens");
    if (g.map.base == NULL) return;

    check(g.version == 3, "version is 3");
    check(g.n_tensors == 3, "finds three tensors");
    check(g.n_kv == 2, "finds two metadata keys");

    char arch[64] = "";
    check(gguf_kv_str(&g, "general.architecture", arch, sizeof arch) == 0,
          "reads a string key");
    check(strcmp(arch, "testarch") == 0, "the string key is right");

    uint64_t layers = 0;
    check(gguf_kv_u64(&g, "testarch.block_count", &layers) == 0,
          "reads an integer key");
    check(layers == 12, "the integer key is right");

    const GgufTensor *t = gguf_find(&g, "blk.1.weight");
    check(t != NULL, "finds a tensor by name");
    if (t) {
        check(t->nbytes == 32, "an f32 tensor of 8 is 32 bytes");
        check(gguf_tensor_data(&g, t) != NULL, "the tensor data is reachable");
    }

    check(gguf_find(&g, "no.such.tensor") == NULL, "a missing name is NULL");
    gguf_close(&g);
}

static void test_rejects_a_bad_magic(void)
{
    printf("a file that is not GGUF is rejected\n");
    Buf b;
    build_good(&b, 1);
    b.p[0] = 'X';
    write_file(&b);
    free(b.p);

    GgufFile g;
    check(gguf_open(&g, TMP) != 0, "refuses it");
    check(strstr(g.err, "magic") != NULL, "and says why");
}

static void test_rejects_a_truncated_file(void)
{
    printf("a truncated file is rejected at every length\n");
    Buf good;
    build_good(&good, 3);

    /* Every prefix of a valid file is an invalid file, and none of them may
     * crash or read past the end. This is the test that catches an
     * off by one in the cursor. */
    int rejected = 0, total = 0;
    for (size_t cut = 4; cut < good.len; cut += 3) {
        Buf b = { good.p, cut, good.cap };
        write_file(&b);
        GgufFile g;
        total++;
        if (gguf_open(&g, TMP) != 0) rejected++;
        else gguf_close(&g);
    }
    free(good.p);
    check(rejected == total, "every prefix is refused");
    printf("  checked %d truncations\n", total);
}

static void test_rejects_a_tensor_that_runs_past_the_end(void)
{
    printf("a tensor pointing outside the file is rejected\n");
    Buf b;
    memset(&b, 0, sizeof b);
    put_u32(&b, 0x46554747u);
    put_u32(&b, 3);
    put_u64(&b, 1);
    put_u64(&b, 0);

    put_str(&b, "liar");
    put_u32(&b, 1);
    put_u64(&b, 8);
    put_u32(&b, GG_F32);
    put_u64(&b, 1000000);          /* far beyond anything written */
    while (b.len % 32) { unsigned char z = 0; put(&b, &z, 1); }
    put_u64(&b, 0);

    write_file(&b);
    free(b.p);

    GgufFile g;
    check(gguf_open(&g, TMP) != 0, "refuses it");
    check(strstr(g.err, "past the end") != NULL, "and names the problem");
}

static void test_rejects_an_absurd_tensor_count(void)
{
    printf("a header claiming more entries than the file holds is rejected\n");
    Buf b;
    memset(&b, 0, sizeof b);
    put_u32(&b, 0x46554747u);
    put_u32(&b, 3);
    put_u64(&b, 0xFFFFFFFFull);    /* four billion tensors in a tiny file */
    put_u64(&b, 0);
    write_file(&b);
    free(b.p);

    GgufFile g;
    check(gguf_open(&g, TMP) != 0, "refuses it");
    check(strstr(g.err, "more entries") != NULL, "and says why");
}

static void test_rejects_an_unsupported_version(void)
{
    printf("an unknown format version is rejected\n");
    Buf b;
    build_good(&b, 1);
    b.p[4] = 99;
    write_file(&b);
    free(b.p);

    GgufFile g;
    check(gguf_open(&g, TMP) != 0, "refuses it");
    check(strstr(g.err, "version") != NULL, "and says which version");
}

static void test_block_sizes_are_right(void)
{
    printf("quantized block sizes match the format\n");

    /* These are the numbers a checkpoint is actually laid out with, so a
     * typo here would misplace every tensor after the first. */
    check(gguf_type_nbytes(GG_F32, 100) == 400, "f32 is 4 bytes each");
    check(gguf_type_nbytes(GG_F16, 100) == 200, "f16 is 2 bytes each");
    check(gguf_type_nbytes(GG_Q8_0, 32) == 34, "q8_0 block is 34 bytes");
    check(gguf_type_nbytes(GG_Q4_0, 32) == 18, "q4_0 block is 18 bytes");
    check(gguf_type_nbytes(GG_Q4_K, 256) == 144, "q4_K block is 144 bytes");
    check(gguf_type_nbytes(GG_Q6_K, 256) == 210, "q6_K block is 210 bytes");

    /* A count that does not fill whole blocks cannot be stored, so it must
     * report zero rather than a rounded guess. */
    check(gguf_type_nbytes(GG_Q4_K, 100) == 0, "a partial block is refused");
    check(gguf_type_nbytes(999, 32) == 0, "an unknown type is refused");
}

static void test_a_missing_file_is_an_error_not_a_crash(void)
{
    printf("a missing file is an error\n");
    GgufFile g;
    check(gguf_open(&g, "definitely_not_here_12345.gguf") != 0, "refuses it");
    check(g.err[0] != 0, "and says something");
}

int main(void)
{
    test_reads_a_good_file();
    test_rejects_a_bad_magic();
    test_rejects_a_truncated_file();
    test_rejects_a_tensor_that_runs_past_the_end();
    test_rejects_an_absurd_tensor_count();
    test_rejects_an_unsupported_version();
    test_block_sizes_are_right();
    test_a_missing_file_is_an_error_not_a_crash();

    remove(TMP);

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
