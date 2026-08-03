/* Measure what this machine's storage actually does.
 *
 * The whole argument for scheduling reads by expert rests on one claim, that
 * reading the same bytes in file order is much faster than reading them in
 * scattered order. That claim is either true on your drive or it is not, and
 * arithmetic cannot tell you which.
 *
 * So this measures it. Three patterns over the same file, same total bytes,
 * same block size, so the only thing that differs is the order.
 *
 *   scattered   blocks in random order, which is what per token streaming does
 *   sorted      the same blocks in file order, which is what a batch does
 *   sequential  a straight sweep, the best the drive can do
 *
 * If sorted lands near sequential, the idea is worth pursuing. If it lands
 * near scattered, the idea is dead and this saved you the work of finding
 * out slowly.
 *
 * Caches make this easy to get wrong. On a machine with plenty of memory a
 * second read of the same file comes back fifty times faster than the drive
 * can manage, and the number looks wonderful and means nothing. So the reads
 * here bypass the cache and go to the drive. Where the filesystem will not
 * allow that, the tool refuses to give a verdict rather than reporting a
 * figure it cannot stand behind.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "direct.h"
#include "map.h"

static double now_s(void)
{
#if defined(_WIN32)
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}

/* A small deterministic generator, so a run can be repeated exactly.
 * This is xorshift64, which is more than random enough for shuffling read
 * order and needs no library. */
static uint64_t rng_state = 0x243F6A8885A308D3ull;

static uint64_t next_rand(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

typedef struct {
    const char *name;
    double seconds;
    double gb_per_s;
    uint64_t bytes;
} Result;

static void report(const Result *r)
{
    printf("  %-12s %8.2f GB in %7.2f s   %6.2f GB/s\n",
           r->name, (double)r->bytes / 1e9, r->seconds, r->gb_per_s);
}

/* Read n_blocks blocks of block_bytes each, in the order given by idx.
 *
 * The reads go through the uncached path, so what is timed is the drive
 * rather than the operating system's memory. Every block lands in the same
 * buffer because the data is not wanted, only the time it took to fetch. */
static Result run_pattern(const char *name, SmDirect *d, const uint64_t *idx,
                          size_t n_blocks, size_t block_bytes, void *buf)
{
    uint64_t sink = 0;
    uint64_t got_total = 0;

    const double t0 = now_s();
    for (size_t i = 0; i < n_blocks; i++) {
        const uint64_t off = idx[i] * (uint64_t)block_bytes;
        if (off >= d->size) continue;
        size_t len = block_bytes;
        if (len > d->size - off) {
            /* The tail of the file is almost never a whole block, and an
             * unaligned length is rejected outright by uncached reads, so
             * the last partial block is simply skipped. */
            continue;
        }
        const int64_t got = sm_direct_read(d, off, buf, len);
        if (got <= 0) continue;
        got_total += (uint64_t)got;
        sink += ((const unsigned char *)buf)[0];
    }
    const double dt = now_s() - t0;

    Result r;
    r.name = name;
    r.seconds = dt;
    r.bytes = got_total;
    r.gb_per_s = dt > 0 ? (double)got_total / dt / 1e9 : 0;

    /* The sum is never used for anything, but the compiler does not know
     * that unless it can prove it. Keeping this branch means the reads are
     * really performed instead of being deleted at -O2. */
    if (sink == 0x5EEDF00Dull) printf(" ");
    return r;
}

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void usage(void)
{
    printf(
"stratum-bandwidth - measure whether read ORDER matters on this machine\n"
"\n"
"usage: stratum-bandwidth FILE [--block MB] [--read MB] [--seed N]\n"
"\n"
"  FILE      any large file. A model checkpoint is ideal because it is the\n"
"            real thing, but any file bigger than memory works.\n"
"  --block   size of each read, in MB. Default 16, which is about the size\n"
"            of one expert in a large sparse model.\n"
"  --read    how much to read in total, in MB. Default 4096.\n"
"  --seed    makes the scattered order repeatable.\n"
"\n"
"Reads bypass the operating system cache, so this measures the drive rather\n"
"than how fast the machine copies memory. Where that cannot be arranged the\n"
"tool says so and refuses to give a verdict.\n");
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage();
        return argc < 2 ? 1 : 0;
    }

    const char *path = argv[1];
    size_t block_mb = 16, read_mb = 4096;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc)
            block_mb = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--read") && i + 1 < argc)
            read_mb = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            rng_state = strtoull(argv[++i], NULL, 10) | 1u;
        else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (!block_mb || !read_mb) {
        fprintf(stderr, "--block and --read must be greater than zero\n");
        return 1;
    }

    SmDirect d;
    char err[512];
    if (sm_direct_open(&d, path, err, sizeof err) != 0) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    /* Uncached reads insist on aligned offsets and lengths, so the block
     * size is rounded up to something the drive will accept. */
    size_t block_bytes = block_mb * 1024u * 1024u;
    if (block_bytes % d.alignment)
        block_bytes += d.alignment - (block_bytes % d.alignment);

    const size_t blocks_in_file = (size_t)(d.size / block_bytes);
    if (blocks_in_file < 8) {
        fprintf(stderr,
                "%s is only %.2f GB, which is too small for a %zu MB block "
                "size to say anything.\nUse a bigger file or a smaller "
                "--block.\n",
                path, (double)d.size / 1e9, block_mb);
        sm_direct_close(&d);
        return 1;
    }

    size_t n_blocks = (read_mb * 1024u * 1024u) / block_bytes;
    if (n_blocks > blocks_in_file) n_blocks = blocks_in_file;
    if (n_blocks < 4) n_blocks = 4;

    printf("stratum-bandwidth\n");
    printf("  file       : %s\n", path);
    printf("  file size  : %.2f GB\n", (double)d.size / 1e9);
    printf("  block      : %zu MB\n", block_bytes / (1024u * 1024u));
    printf("  blocks read: %zu of %zu\n", n_blocks, blocks_in_file);
    printf("  reads      : %s\n\n",
           d.uncached ? "uncached, so this measures the drive"
                      : "CACHED, so these numbers are not the drive");
    if (d.note[0]) printf("  note: %s\n\n", d.note);

    /* The same blocks are used for the scattered and the sorted pass. That
     * is the point of the whole exercise, so that the only difference
     * between them is the order they are visited in. */
    uint64_t *idx = (uint64_t *)malloc(n_blocks * sizeof *idx);
    uint64_t *sorted = (uint64_t *)malloc(n_blocks * sizeof *sorted);
    uint64_t *seq = (uint64_t *)malloc(n_blocks * sizeof *seq);
    void *buf = sm_aligned_alloc(block_bytes, d.alignment);
    if (!idx || !sorted || !seq || !buf) {
        fprintf(stderr, "out of memory\n");
        free(idx); free(sorted); free(seq);
        sm_aligned_free(buf);
        sm_direct_close(&d);
        return 1;
    }

    for (size_t i = 0; i < n_blocks; i++) {
        idx[i] = next_rand() % blocks_in_file;
        seq[i] = i % blocks_in_file;
    }
    memcpy(sorted, idx, n_blocks * sizeof *idx);
    qsort(sorted, n_blocks, sizeof *sorted, cmp_u64);

    /* Scattered runs first, because it is the pattern least helped by
     * anything left over from an earlier pass. Sorted runs on the same
     * blocks straight after, which if anything flatters scattered, so a win
     * for sorted here is a real one. */
    Result a = run_pattern("scattered", &d, idx, n_blocks, block_bytes, buf);
    report(&a);
    Result b = run_pattern("sorted", &d, sorted, n_blocks, block_bytes, buf);
    report(&b);
    Result c = run_pattern("sequential", &d, seq, n_blocks, block_bytes, buf);
    report(&c);

    printf("\n");

    /* No consumer drive reads this fast. A number above this did not come
     * from storage at all, and reporting a verdict from it would be worse
     * than reporting nothing. The whole point of this tool is to replace an
     * assumption with a measurement, so a measurement it cannot stand
     * behind has to be refused out loud. */
    const double IMPOSSIBLE_GB_S = 15.0;
    const double fastest = a.gb_per_s > b.gb_per_s
                         ? (a.gb_per_s > c.gb_per_s ? a.gb_per_s : c.gb_per_s)
                         : (b.gb_per_s > c.gb_per_s ? b.gb_per_s : c.gb_per_s);

    if (fastest > IMPOSSIBLE_GB_S || !d.uncached) {
        printf("  NO VERDICT. At %.0f GB/s these reads did not come from the "
               "drive.\n", fastest);
        printf("  Something served them from memory, so what was measured is "
               "how fast this\n  machine copies memory. That says nothing "
               "about read order.\n\n");
        printf("  Use a file on the drive you actually want to measure, and "
               "one that is not\n  a copy the system is already holding.\n");
        free(idx); free(sorted); free(seq);
        sm_aligned_free(buf);
        sm_direct_close(&d);
        return 2;
    }

    if (a.gb_per_s > 0) {
        const double gain = b.gb_per_s / a.gb_per_s;
        printf("  sorting the same reads into file order: %.2fx\n", gain);
        if (gain >= 1.5)
            printf("  Read order matters on this drive, so scheduling reads "
                   "by file position is worth doing.\n");
        else
            printf("  Read order barely matters on this drive. Scheduling "
                   "reads by file position\n  would not pay for itself here, "
                   "which is worth knowing before building it.\n");
    }

    free(idx); free(sorted); free(seq);
    sm_aligned_free(buf);
    sm_direct_close(&d);
    return 0;
}
