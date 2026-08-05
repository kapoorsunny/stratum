/* Measure what this machine's storage actually does.
 *
 * Any argument about how to lay out an index or stream a checkpoint rests on
 * a claim about the drive underneath it, and arithmetic cannot tell you
 * whether that claim is true on yours.
 *
 * So this measures it. Three patterns over the same file, the same total
 * bytes, the same block size, so the only thing that differs is the order.
 *
 *   scattered   blocks in random order, which is what per token streaming
 *               and a graph walk both do
 *   sorted      the same blocks in file order, which is what a batch can do
 *   sequential  a straight sweep, the best the drive can manage
 *
 * Two things this measures that are easy to get wrong.
 *
 * Caches. On a machine with plenty of memory, a second read of the same file
 * comes back fifty times faster than any drive can manage, and the number
 * looks wonderful and means nothing. The reads here bypass the cache. Where
 * the filesystem will not allow that, the tool refuses to give a verdict
 * rather than reporting a figure it cannot stand behind.
 *
 * Concurrency. A drive answers many reads at once far better than it answers
 * them one at a time, so a measurement taken with a single read outstanding
 * describes a laptop with one user and says nothing about a server with a
 * hundred. --threads is how you see the difference.
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
#include "thread.h"

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

/* One worker's slice. Each has its own buffer and its own run of the index
 * array, so nothing is shared and no locking is needed. */
typedef struct {
    SmDirect *d;
    const uint64_t *idx;
    size_t first, count, block_bytes;
    void *buf;
    uint64_t got_total, asked, sink;
} Slice;

static void read_slice(void *p)
{
    Slice *s = (Slice *)p;
    for (size_t i = s->first; i < s->first + s->count; i++) {
        const uint64_t off = s->idx[i] * (uint64_t)s->block_bytes;
        if (off >= s->d->size) continue;
        /* The tail of a file is almost never a whole block, and an unaligned
         * length is rejected outright by uncached reads, so the last partial
         * block is simply skipped. */
        if (s->block_bytes > s->d->size - off) continue;
        s->asked += s->block_bytes;
        const int64_t got = sm_direct_read(s->d, off, s->buf, s->block_bytes);
        if (got <= 0) continue;
        s->got_total += (uint64_t)got;
        s->sink += ((const unsigned char *)s->buf)[0];
    }
}

static Result run_pattern(const char *name, SmDirect *d, const uint64_t *idx,
                          size_t n_blocks, size_t block_bytes,
                          void **bufs, int threads)
{
    Slice slices[SM_MAX_THREADS];
    void *args[SM_MAX_THREADS];
    if (threads < 1) threads = 1;
    if (threads > SM_MAX_THREADS) threads = SM_MAX_THREADS;
    if ((size_t)threads > n_blocks) threads = (int)n_blocks;

    const size_t per = n_blocks / (size_t)threads;
    for (int t = 0; t < threads; t++) {
        slices[t].d = d;
        slices[t].idx = idx;
        slices[t].first = (size_t)t * per;
        /* The last worker mops up the remainder so no block is dropped. */
        slices[t].count = (t == threads - 1) ? n_blocks - (size_t)t * per : per;
        slices[t].block_bytes = block_bytes;
        slices[t].buf = bufs[t];
        slices[t].got_total = 0;
        slices[t].asked = 0;
        slices[t].sink = 0;
        args[t] = &slices[t];
    }

    const double t0 = now_s();
    sm_run_parallel(read_slice, args, threads);
    const double dt = now_s() - t0;

    uint64_t got_total = 0, asked = 0, sink = 0;
    for (int t = 0; t < threads; t++) {
        got_total += slices[t].got_total;
        asked += slices[t].asked;
        sink += slices[t].sink;
    }

    /* A read that quietly returned nothing would otherwise show up as
     * excellent throughput over a small number of bytes. Saying so is the
     * difference between a measurement and a number. */
    if (asked && got_total < asked)
        printf("  WARNING: %s got %.1f%% of the bytes it asked for, so this "
               "figure is not trustworthy\n", name,
               100.0 * (double)got_total / (double)asked);

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
"stratum-bandwidth - measure how this machine's drive really behaves\n"
"\n"
"usage: stratum-bandwidth FILE [--block MB] [--block-kb KB] [--read MB]\n"
"                              [--threads N] [--seed N]\n"
"\n"
"  FILE        any large file. A model checkpoint is ideal because it is the\n"
"              real thing, but any large file works.\n"
"  --block     size of each read in MB. Default 16, about one expert in a\n"
"              large sparse model.\n"
"  --block-kb  the same in KB, for the small sizes a retrieval index reads\n"
"              in. This is the range where the answer changes most.\n"
"  --read      how much to read in total, in MB. Default 4096.\n"
"  --threads   how many reads to have in flight at once. Default 1, which is\n"
"              one user waiting. A server has many, and a drive answers many\n"
"              at once far better than it answers them one by one.\n"
"  --seed      makes the scattered order repeatable.\n"
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
    size_t block_kb = 16 * 1024, read_mb = 4096;
    int threads = 1;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--block") && i + 1 < argc)
            block_kb = (size_t)strtoull(argv[++i], NULL, 10) * 1024;
        else if (!strcmp(argv[i], "--block-kb") && i + 1 < argc)
            block_kb = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--read") && i + 1 < argc)
            read_mb = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = (int)strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            rng_state = strtoull(argv[++i], NULL, 10) | 1u;
        else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }
    if (!block_kb || !read_mb) {
        fprintf(stderr, "--block and --read must be greater than zero\n");
        return 1;
    }

    SmDirect d;
    char err[512];
    if (sm_direct_open(&d, path, err, sizeof err) != 0) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }

    /* Uncached reads insist on aligned offsets and lengths, so the block size
     * is rounded up to something the drive will accept. */
    size_t block_bytes = block_kb * 1024u;
    if (block_bytes % d.alignment)
        block_bytes += d.alignment - (block_bytes % d.alignment);

    const size_t blocks_in_file = (size_t)(d.size / block_bytes);
    if (blocks_in_file < 8) {
        fprintf(stderr,
                "%s is only %.2f GB, which is too small for a %zu KB block "
                "size to say anything.\nUse a bigger file or a smaller "
                "--block.\n",
                path, (double)d.size / 1e9, block_kb);
        sm_direct_close(&d);
        return 1;
    }

    size_t n_blocks = (read_mb * 1024u * 1024u) / block_bytes;
    if (n_blocks > blocks_in_file) n_blocks = blocks_in_file;
    if (n_blocks < 4) n_blocks = 4;

    printf("stratum-bandwidth\n");
    printf("  file       : %s\n", path);
    printf("  file size  : %.2f GB\n", (double)d.size / 1e9);
    if (block_bytes >= 1024u * 1024u)
        printf("  block      : %zu MB\n", block_bytes / (1024u * 1024u));
    else
        printf("  block      : %zu KB\n", block_bytes / 1024u);
    printf("  blocks read: %zu of %zu\n", n_blocks, blocks_in_file);
    printf("  threads    : %d\n", threads);
    printf("  reads      : %s\n\n",
           d.uncached ? "uncached, so this measures the drive"
                      : "CACHED, so these numbers are not the drive");
    if (d.note[0]) printf("  note: %s\n\n", d.note);

    /* The same blocks are used for the scattered and the sorted pass. That is
     * the point of the exercise, so that the only difference between them is
     * the order they are visited in. */
    uint64_t *idx = (uint64_t *)malloc(n_blocks * sizeof *idx);
    uint64_t *sorted = (uint64_t *)malloc(n_blocks * sizeof *sorted);
    uint64_t *seq = (uint64_t *)malloc(n_blocks * sizeof *seq);

    /* One buffer per thread, because they read at the same time and would
     * otherwise be writing over each other. */
    if (threads < 1) threads = 1;
    if (threads > SM_MAX_THREADS) threads = SM_MAX_THREADS;
    void *bufs[SM_MAX_THREADS];
    int allocated = 0;
    for (; allocated < threads; allocated++) {
        bufs[allocated] = sm_aligned_alloc(block_bytes, d.alignment);
        if (!bufs[allocated]) break;
    }
    if (!idx || !sorted || !seq || allocated < threads) {
        fprintf(stderr, "out of memory\n");
        free(idx); free(sorted); free(seq);
        for (int i = 0; i < allocated; i++) sm_aligned_free(bufs[i]);
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
     * anything left over from an earlier pass. Sorted runs on the same blocks
     * straight after, which if anything flatters scattered, so a win for
     * sorted here is a real one. */
    Result a = run_pattern("scattered", &d, idx, n_blocks, block_bytes, bufs, threads);
    report(&a);
    Result b = run_pattern("sorted", &d, sorted, n_blocks, block_bytes, bufs, threads);
    report(&b);
    Result c = run_pattern("sequential", &d, seq, n_blocks, block_bytes, bufs, threads);
    report(&c);

    printf("\n");

    /* No consumer drive reads this fast. A number above this did not come
     * from storage at all, and reporting a verdict from it would be worse
     * than reporting nothing. The whole point of this tool is to replace an
     * assumption with a measurement, so a measurement it cannot stand behind
     * has to be refused out loud. */
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
        for (int i = 0; i < allocated; i++) sm_aligned_free(bufs[i]);
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

    /* The figure that decides how to lay anything out. Every read carries a
     * fixed cost no matter how small it is, so the thing to minimise is the
     * number of reads that must happen one after another, not their size. */
    if (a.bytes && a.seconds > 0 && block_bytes) {
        const uint64_t reads = a.bytes / block_bytes;
        if (reads) {
            printf("\n  each scattered read cost about %.0f microseconds, "
                   "at %zu KB a time.\n",
                   a.seconds * 1e6 / (double)reads, block_bytes / 1024u);
            printf("  Try --block-kb 4 against --block-kb 256 to see how "
                   "little of that is size,\n  and --threads 16 to see how "
                   "much of it disappears under concurrency.\n");
        }
    }

    free(idx); free(sorted); free(seq);
    for (int i = 0; i < allocated; i++) sm_aligned_free(bufs[i]);
    sm_direct_close(&d);
    return 0;
}
