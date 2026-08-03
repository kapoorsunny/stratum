/* Reads that go to the drive and not to the page cache.
 *
 * Measuring storage through the normal read path measures the operating
 * system's cache instead, and on a machine with plenty of memory that is
 * off by a factor of fifty. The numbers look wonderful and mean nothing.
 *
 * Every platform has a way to say do not cache this, and every platform
 * spells it differently:
 *
 *   Windows   FILE_FLAG_NO_BUFFERING at open time
 *   Linux     O_DIRECT at open time
 *   macOS     F_NOCACHE after opening, since there is no open flag for it
 *
 * All three demand that reads be aligned. The offset, the length and the
 * destination buffer must all sit on sector boundaries, so this hands out
 * aligned buffers rather than trusting the caller to know that.
 */
#ifndef STRATUM_DIRECT_H
#define STRATUM_DIRECT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    void *handle;       /* platform bookkeeping */
    uint64_t size;      /* file size in bytes */
    size_t alignment;   /* offsets and lengths must be multiples of this */
    int uncached;       /* 1 when the cache really was bypassed */
    char note[256];     /* why, when it could not be */
} SmDirect;

/* Open a file for uncached reading. Returns 0 on success.
 *
 * Succeeding with uncached set to 0 is a real outcome, not a failure. Some
 * filesystems refuse the flag, and the caller needs to know the numbers are
 * not trustworthy rather than being told nothing happened. */
int sm_direct_open(SmDirect *d, const char *path, char *err, size_t errlen);

void sm_direct_close(SmDirect *d);

/* Read len bytes at off into buf. All three must be aligned.
 * Returns bytes read, or a negative value on failure. */
int64_t sm_direct_read(SmDirect *d, uint64_t off, void *buf, size_t len);

/* An aligned buffer, and the matching free. Ordinary malloc does not
 * promise the alignment these reads require. */
void *sm_aligned_alloc(size_t bytes, size_t alignment);
void sm_aligned_free(void *p);

#endif
