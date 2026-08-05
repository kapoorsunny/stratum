/* The smallest thread abstraction that does the job.
 *
 * Storage answers many reads at once far better than it answers them one at
 * a time, and a measurement taken with one read outstanding says nothing
 * about a machine serving many users. Measuring that needs threads, and
 * needing threads should not mean needing a library.
 *
 * Start some workers, wait for them, done. There is no pool, no queue and
 * no synchronisation, because every worker here is given a slice of the
 * work up front and touches nothing the others touch.
 */
#ifndef STRATUM_THREAD_H
#define STRATUM_THREAD_H

#include <stddef.h>

#define SM_MAX_THREADS 256

typedef void (*SmWorkFn)(void *arg);

/* Run fn(args[i]) on its own thread for each of n, and return once every one
 * has finished. With n of 1 the work runs on the calling thread and no
 * thread is created at all. */
int sm_run_parallel(SmWorkFn fn, void **args, int n);

#endif
