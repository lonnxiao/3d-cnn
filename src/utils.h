/******************************************************************************
* filename: 
* file function: 
* last modified: 
* NOT need to be modified.
*******************************************************************************/
#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "list.h"

#define SECRET_NUM -1234

void free_ptrs(void **ptrs, int n);
char *basecfg(char *cfgfile);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
char *find_replace(char *str, char *orig, char *rep);
void error(const char *s);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void top_k(float *a, int n, int k, int *index);
char *fgetl(FILE *fp);
char *copy_string(char *s);
float rand_uniform(float min, float max);
int rand_int(int min, int max);
float sum_array(float *a, int n);
float sec(clock_t clocks);
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);

#endif

