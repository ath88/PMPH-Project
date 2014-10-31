#include <cuda.h>

#include "ProjHelperFun.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

size_t initial_memory_usage;
void reportMemoryUsageInit() {
	size_t free;
	size_t total;
	cuMemGetInfo(&free, &total);
	initial_memory_usage = total - free;
}
void reportMemoryUsage() {
	size_t free;
	size_t total;
	cuMemGetInfo(&free, &total);
	printf("Memory usage: %dMiB (base %dMiB, ours %dMiB) / %dMiB\n",
			(total - free)/1024/1024,
			(initial_memory_usage)/1024/1024,
			(total - free - initial_memory_usage)/1024/1024,
			total/1024/1024);
}
