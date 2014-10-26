#ifndef TIMERS_H
#define TIMERS_H

#include <time.h>
#include <sys/time.h>


int timeval_subtract_(
		struct timeval *result, struct timeval *t2, struct timeval *t1);

int timeval_add_(
		struct timeval *result, struct timeval *t2, struct timeval *t1);

unsigned long int timeval_get_mu_s(struct timeval *t);

//int timeval_subtract_get_mu_s(struct timeval *t1, struct timeval *t2) {
//	struct timeval dt;
//	timeval_subtract(&dt, t1, t2);
//	return timeval_get_mu_s_(&dt);
//}

void timers_init();

void timer_indent(int i);
int timer_indent_get();

#define TIMER_DEFINE(name) \
	struct timeval timer_##name##_total; \
	struct timeval timer_##name##_accumulate; \
	struct timeval timer_##name##_start; \
	struct timeval timer_##name##_end; \
	;

#define TIMER_INIT(name) \
	timer_##name##_total.tv_usec = 0; \
	timer_##name##_total.tv_sec = 0; \
	;

#define TIMER_START(name) \
	gettimeofday(&timer_##name##_start, NULL); \
	;

#define TIMER_STOP(name) \
	gettimeofday(&timer_##name##_end, NULL); \
	timeval_subtract_( \
			&timer_##name##_accumulate, \
			&timer_##name##_end, \
			&timer_##name##_start); \
	timeval_add_( \
			&timer_##name##_total, \
			&timer_##name##_total, \
			&timer_##name##_accumulate); \
	;

#define TIMER_REPORT(name) { \
	for(int i = 0; i < timer_indent_get(); i += 1) { \
		printf("  "); \
	} \
	printf("%-20s%*i microseconds\n", \
			#name " ", \
			14 - timer_indent_get() * 2, \
			timeval_get_mu_s(&timer_##name##_total)); \
};

//#define TIMER_GROUP(name_string) \
//	for(int i = 0; i < timer_indent_get(); i += 1) { \
//		printf("  "); \
//	} \
//	printf("%s:\n", name_string); \
//	timer_indent(1); \
//	;
#define TIMER_GROUP() \
	timer_indent(1); \
	;

#define TIMER_GROUP_END() \
	timer_indent(-1); \
	;

#endif // TIMERS_H
