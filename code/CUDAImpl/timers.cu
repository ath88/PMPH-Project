#include "timers.h"

static int indent;

int timeval_subtract_(
		struct timeval *result, struct timeval *t2, struct timeval *t1) {
	unsigned int resolution = 1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec)
			- (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return diff < 0;
}

int timeval_add_(
		struct timeval *result, struct timeval *t2, struct timeval *t1) {
	unsigned int resolution = 1000000;
	long int diff = (t2->tv_usec + resolution * t2->tv_sec)
			+ (t1->tv_usec + resolution * t1->tv_sec);
	result->tv_sec = diff / resolution;
	result->tv_usec = diff % resolution;
	return diff < 0;
}

unsigned long int timeval_get_mu_s(struct timeval *t) {
	return t->tv_sec * 1e6 + t->tv_usec;
}

void timers_init() {
	indent = 0;
}

void timer_indent(int i) {
	indent += i;
}
int timer_indent_get() {
	return indent;
}
