/* See direct.h. */
#include "direct.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

/* 4096 covers every drive in normal use. Asking the system for the real
 * figure is possible but the answer is always this or smaller, and reading
 * in larger aligned units is always allowed. */
#define FALLBACK_ALIGN 4096

void *sm_aligned_alloc(size_t bytes, size_t alignment)
{
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, bytes) != 0) return NULL;
    return p;
#endif
}

void sm_aligned_free(void *p)
{
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

#if defined(_WIN32)

int sm_direct_open(SmDirect *d, const char *path, char *err, size_t errlen)
{
    memset(d, 0, sizeof *d);
    d->alignment = FALLBACK_ALIGN;

    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                           NULL);
    if (f != INVALID_HANDLE_VALUE) {
        d->uncached = 1;
    } else {
        /* Some filesystems, network shares in particular, refuse the flag.
         * Falling back is better than failing, as long as the caller is told
         * the numbers now include the cache. */
        f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
        if (f == INVALID_HANDLE_VALUE) {
            snprintf(err, errlen, "cannot open %s (win32 error %lu)",
                     path, (unsigned long)GetLastError());
            return 1;
        }
        snprintf(d->note, sizeof d->note,
                 "this filesystem refused uncached reads, so the cache is "
                 "still in the way");
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
        snprintf(err, errlen, "cannot size %s", path);
        CloseHandle(f);
        return 1;
    }

    d->handle = (void *)f;
    d->size = (uint64_t)sz.QuadPart;
    return 0;
}

void sm_direct_close(SmDirect *d)
{
    if (d && d->handle) CloseHandle((HANDLE)d->handle);
    if (d) memset(d, 0, sizeof *d);
}

int64_t sm_direct_read(SmDirect *d, uint64_t off, void *buf, size_t len)
{
    /* OVERLAPPED carries the offset, so nothing here depends on a shared
     * file position and several of these could run at once. */
    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(off & 0xFFFFFFFFull);
    ov.OffsetHigh = (DWORD)(off >> 32);

    DWORD got = 0;
    if (!ReadFile((HANDLE)d->handle, buf, (DWORD)len, &got, &ov)) {
        if (GetLastError() != ERROR_IO_PENDING) return -1;
        if (!GetOverlappedResult((HANDLE)d->handle, &ov, &got, TRUE)) return -1;
    }
    return (int64_t)got;
}

#else /* POSIX */

int sm_direct_open(SmDirect *d, const char *path, char *err, size_t errlen)
{
    memset(d, 0, sizeof *d);
    d->alignment = FALLBACK_ALIGN;

    int flags = O_RDONLY;
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif

    int fd = open(path, flags);
#ifdef O_DIRECT
    if (fd < 0) {
        /* tmpfs and several network filesystems reject O_DIRECT outright. */
        fd = open(path, O_RDONLY);
        if (fd >= 0)
            snprintf(d->note, sizeof d->note,
                     "this filesystem refused uncached reads, so the cache is "
                     "still in the way");
    } else {
        d->uncached = 1;
    }
#endif
    if (fd < 0) {
        snprintf(err, errlen, "cannot open %s", path);
        return 1;
    }

#if defined(__APPLE__) && defined(F_NOCACHE)
    /* macOS has no open flag for this. The equivalent is set afterwards,
     * and it applies to the descriptor rather than to each read. */
    if (fcntl(fd, F_NOCACHE, 1) == 0) {
        d->uncached = 1;
        d->note[0] = 0;
    } else {
        snprintf(d->note, sizeof d->note,
                 "F_NOCACHE was refused, so the cache is still in the way");
    }
#endif

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        snprintf(err, errlen, "cannot size %s", path);
        close(fd);
        return 1;
    }

    d->handle = (void *)(intptr_t)fd;
    d->size = (uint64_t)st.st_size;
    return 0;
}

void sm_direct_close(SmDirect *d)
{
    if (d && d->handle) close((int)(intptr_t)d->handle);
    if (d) memset(d, 0, sizeof *d);
}

int64_t sm_direct_read(SmDirect *d, uint64_t off, void *buf, size_t len)
{
    /* pread takes the offset as an argument, so it does not disturb the
     * descriptor's own position and is safe to call from several threads. */
    unsigned char *p = (unsigned char *)buf;
    size_t total = 0;
    while (total < len) {
        const ssize_t n = pread((int)(intptr_t)d->handle, p + total,
                                len - total, (off_t)(off + total));
        if (n < 0) return -1;
        if (n == 0) break;              /* end of file */
        total += (size_t)n;
    }
    return (int64_t)total;
}

#endif
