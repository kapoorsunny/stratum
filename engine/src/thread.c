/* See thread.h. */
#include "thread.h"

#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#endif

typedef struct { SmWorkFn fn; void *arg; } Job;

#if defined(_WIN32)

static DWORD WINAPI trampoline(LPVOID p)
{
    Job *j = (Job *)p;
    j->fn(j->arg);
    return 0;
}

int sm_run_parallel(SmWorkFn fn, void **args, int n)
{
    if (n <= 0) return 0;
    if (n > SM_MAX_THREADS) n = SM_MAX_THREADS;
    if (n == 1) { fn(args[0]); return 0; }

    Job jobs[SM_MAX_THREADS];
    HANDLE th[SM_MAX_THREADS];
    int started = 0;

    for (int i = 0; i < n; i++) {
        jobs[i].fn = fn;
        jobs[i].arg = args[i];
        th[i] = CreateThread(NULL, 0, trampoline, &jobs[i], 0, NULL);
        if (!th[i]) break;
        started++;
    }

    /* Anything that failed to start still has to be run, or the caller
     * silently loses that slice of the work and the numbers come out
     * flattering. */
    for (int i = started; i < n; i++) fn(args[i]);

    if (started) WaitForMultipleObjects((DWORD)started, th, TRUE, INFINITE);
    for (int i = 0; i < started; i++) CloseHandle(th[i]);
    return started == n ? 0 : 1;
}

#else

static void *trampoline(void *p)
{
    Job *j = (Job *)p;
    j->fn(j->arg);
    return NULL;
}

int sm_run_parallel(SmWorkFn fn, void **args, int n)
{
    if (n <= 0) return 0;
    if (n > SM_MAX_THREADS) n = SM_MAX_THREADS;
    if (n == 1) { fn(args[0]); return 0; }

    Job jobs[SM_MAX_THREADS];
    pthread_t th[SM_MAX_THREADS];
    int started = 0;

    for (int i = 0; i < n; i++) {
        jobs[i].fn = fn;
        jobs[i].arg = args[i];
        if (pthread_create(&th[i], NULL, trampoline, &jobs[i]) != 0) break;
        started++;
    }

    for (int i = started; i < n; i++) fn(args[i]);
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
    return started == n ? 0 : 1;
}

#endif
