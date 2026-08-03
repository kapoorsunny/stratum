/* See map.h. */
#include "map.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <psapi.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

size_t sm_page_size(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
#else
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t)p : 4096;
#endif
}

/* Round an offset down and a length up to page boundaries.
 * Every hint below has to be page aligned or the platform rejects it. */
static void align_range(size_t off, size_t len, size_t total,
                        size_t *aoff, size_t *alen)
{
    const size_t pg = sm_page_size();
    if (off > total) { *aoff = 0; *alen = 0; return; }
    if (len > total - off) len = total - off;
    const size_t start = off - (off % pg);
    size_t end = off + len;
    const size_t rem = end % pg;
    if (rem) end += pg - rem;
    if (end > total) end = total;
    *aoff = start;
    *alen = end > start ? end - start : 0;
}

#if defined(_WIN32)

typedef struct { HANDLE file; HANDLE mapping; } WinMap;

int sm_map_open(SmMap *m, const char *path, char *err, size_t errlen)
{
    memset(m, 0, sizeof *m);

    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        snprintf(err, errlen, "cannot open %s (win32 error %lu)",
                 path, (unsigned long)GetLastError());
        return 1;
    }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
        snprintf(err, errlen, "cannot size %s", path);
        CloseHandle(f);
        return 1;
    }

    HANDLE mp = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mp) {
        snprintf(err, errlen, "cannot map %s (win32 error %lu)",
                 path, (unsigned long)GetLastError());
        CloseHandle(f);
        return 1;
    }

    const void *base = MapViewOfFile(mp, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        snprintf(err, errlen, "cannot view %s (win32 error %lu)",
                 path, (unsigned long)GetLastError());
        CloseHandle(mp);
        CloseHandle(f);
        return 1;
    }

    WinMap *w = (WinMap *)calloc(1, sizeof *w);
    if (!w) {
        snprintf(err, errlen, "out of memory mapping %s", path);
        UnmapViewOfFile(base);
        CloseHandle(mp);
        CloseHandle(f);
        return 1;
    }
    w->file = f;
    w->mapping = mp;

    m->base = (const unsigned char *)base;
    m->size = (size_t)sz.QuadPart;
    m->handle = w;
    return 0;
}

void sm_map_close(SmMap *m)
{
    if (!m || !m->base) return;
    WinMap *w = (WinMap *)m->handle;
    UnmapViewOfFile((LPCVOID)m->base);
    if (w) {
        if (w->mapping) CloseHandle(w->mapping);
        if (w->file) CloseHandle(w->file);
        free(w);
    }
    memset(m, 0, sizeof *m);
}

/* Windows has no madvise. PrefetchVirtualMemory is the closest thing and it
 * only covers the prefetch case, so the advice calls are honest no ops. */
void sm_map_advise_sequential(SmMap *m, size_t off, size_t len)
{
    (void)m; (void)off; (void)len;
}

void sm_map_advise_random(SmMap *m, size_t off, size_t len)
{
    (void)m; (void)off; (void)len;
}

void sm_map_prefetch(SmMap *m, size_t off, size_t len)
{
    if (!m || !m->base) return;
    size_t aoff, alen;
    align_range(off, len, m->size, &aoff, &alen);
    if (!alen) return;

    /* Resolved at runtime because it needs Windows 8 or newer and the engine
     * should still load on older builds, just without the hint. */
    typedef BOOL (WINAPI *PfnPrefetch)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    /* A union rather than a cast because C does not promise that a function
     * pointer and an object pointer are the same shape. */
    union { FARPROC in; PfnPrefetch out; } conv;
    static PfnPrefetch pfn;
    static int looked_up;
    if (!looked_up) {
        HMODULE k = GetModuleHandleA("kernel32.dll");
        if (k) {
            conv.in = GetProcAddress(k, "PrefetchVirtualMemory");
            pfn = conv.out;
        }
        looked_up = 1;
    }
    if (!pfn) return;

    WIN32_MEMORY_RANGE_ENTRY e;
    e.VirtualAddress = (PVOID)(m->base + aoff);
    e.NumberOfBytes = alen;
    pfn(GetCurrentProcess(), 1, &e, 0);
}

void sm_map_release(SmMap *m, size_t off, size_t len)
{
    if (!m || !m->base) return;
    size_t aoff, alen;
    align_range(off, len, m->size, &aoff, &alen);
    if (!alen) return;
    /* Unmaps the pages from the working set while leaving the view valid,
     * which is what the POSIX side gets from MADV_DONTNEED. */
    VirtualUnlock((LPVOID)(m->base + aoff), alen);
}

uint64_t sm_rss_bytes(void)
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (uint64_t)pmc.WorkingSetSize;
    return 0;
}

#else /* POSIX */

int sm_map_open(SmMap *m, const char *path, char *err, size_t errlen)
{
    memset(m, 0, sizeof *m);

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(err, errlen, "cannot open %s", path);
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        snprintf(err, errlen, "cannot size %s", path);
        close(fd);
        return 1;
    }

    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    /* The descriptor is not needed once the mapping exists. */
    close(fd);
    if (base == MAP_FAILED) {
        snprintf(err, errlen, "cannot map %s", path);
        return 1;
    }

    m->base = (const unsigned char *)base;
    m->size = (size_t)st.st_size;
    m->handle = NULL;
    return 0;
}

void sm_map_close(SmMap *m)
{
    if (!m || !m->base) return;
    munmap((void *)m->base, m->size);
    memset(m, 0, sizeof *m);
}

static void advise(SmMap *m, size_t off, size_t len, int flag)
{
    if (!m || !m->base) return;
    size_t aoff, alen;
    align_range(off, len, m->size, &aoff, &alen);
    if (alen) madvise((void *)(m->base + aoff), alen, flag);
}

void sm_map_advise_sequential(SmMap *m, size_t off, size_t len)
{
    advise(m, off, len, MADV_SEQUENTIAL);
}

void sm_map_advise_random(SmMap *m, size_t off, size_t len)
{
    advise(m, off, len, MADV_RANDOM);
}

void sm_map_prefetch(SmMap *m, size_t off, size_t len)
{
#ifdef MADV_WILLNEED
    advise(m, off, len, MADV_WILLNEED);
#else
    (void)m; (void)off; (void)len;
#endif
}

void sm_map_release(SmMap *m, size_t off, size_t len)
{
#ifdef MADV_DONTNEED
    advise(m, off, len, MADV_DONTNEED);
#else
    (void)m; (void)off; (void)len;
#endif
}

uint64_t sm_rss_bytes(void)
{
    /* Linux reports resident pages in field 2 of statm. Everything else
     * returns 0 and the benchmark says so rather than printing a guess. */
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long total = 0, resident = 0;
    const int got = fscanf(f, "%llu %llu", &total, &resident);
    fclose(f);
    if (got != 2) return 0;
    return (uint64_t)resident * (uint64_t)sm_page_size();
}

#endif
