/* Portable read only file mapping.
 *
 * A checkpoint is far bigger than memory, so it is never read into a buffer.
 * It is mapped, and the operating system pages in only the bytes actually
 * touched. That is the whole reason a 1.5 TB file can be opened on a laptop.
 *
 * POSIX gets mmap, Windows gets MapViewOfFile, and the rest of the engine
 * sees neither.
 */
#ifndef STRATUM_MAP_H
#define STRATUM_MAP_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const unsigned char *base;  /* first mapped byte, NULL when the map failed */
    size_t size;                /* bytes mapped */
    void *handle;               /* platform bookkeeping, do not touch */
} SmMap;

/* Map a whole file read only. Returns 0 on success and fills m.
 * On failure returns non zero and writes a reason into err. */
int sm_map_open(SmMap *m, const char *path, char *err, size_t errlen);

void sm_map_close(SmMap *m);

/* Tell the kernel how these bytes will be used.
 *
 * Sequential means read ahead aggressively, which is what a straight sweep
 * over a tensor wants. Random means do not bother, which is what scattered
 * expert reads want because read ahead there only evicts pages that were
 * about to be used. Both are hints and both are safe to ignore. */
void sm_map_advise_sequential(SmMap *m, size_t offset, size_t len);
void sm_map_advise_random(SmMap *m, size_t offset, size_t len);

/* Ask for these bytes to be brought in without blocking on them.
 *
 * This is how the scheduler overlaps a disk read with arithmetic. It returns
 * as soon as the request is queued, so the caller keeps working while the
 * drive fills the pages. */
void sm_map_prefetch(SmMap *m, size_t offset, size_t len);

/* Drop these bytes from the resident set.
 *
 * Streaming a checkpoint larger than RAM means the page cache fills with
 * weights that will not be needed again for a long time. Releasing an expert
 * once its batch is done keeps the resident set flat instead of letting the
 * kernel decide, badly, under pressure. */
void sm_map_release(SmMap *m, size_t offset, size_t len);

/* Resident set of this process in bytes, or 0 when it cannot be read.
 * The benchmark reports it because a memory claim nobody measured is just
 * a hope. */
uint64_t sm_rss_bytes(void);

/* Page size of this machine. */
size_t sm_page_size(void);

#endif
