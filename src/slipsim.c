#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define RSTRCT restrict
#define MAX_C_ALLOC 16384
#define MAX_C_COUNT 8192
#define ALLOC_INCREASE_FACTOR 2
#define REALLOC_THRESHOLD 2.0
#define INITIAL_C_ALLOC 1024
#define KEEP_C_LIFETIMES 1

const size_t size_t_size = sizeof(size_t);
const size_t double_size = sizeof(double);

gsl_rng* rand_core;

typedef struct rand_buffer {
	size_t buffer_size;
	ssize_t position;
	double* buffer;
} rand_buffer;

const size_t rand_buffer_size = sizeof(rand_buffer);

void allocate_rand_buffer(rand_buffer*const RSTRCT rbuf, const size_t size) {
	assert(rbuf);
	assert(!(rbuf->buffer));
	rbuf->buffer = (double*) malloc(size*double_size);
	rbuf->buffer_size = size;
	rbuf->position = -1;
	return;
}

void free_rand_buffer(rand_buffer*const RSTRCT rbuf) {
	assert(rbuf);
	assert(rbuf->buffer);
	free(rbuf->buffer);
	rbuf->buffer = NULL;
	rbuf->buffer_size = 0;
	rbuf->position = -1;
	return;
}

void fill_rand_buffer_flat(rand_buffer*const RSTRCT rbuf) {
	assert(rbuf);
	assert(rbuf->buffer);
	assert(rbuf->buffer_size > 0);
	const size_t size = rbuf->buffer_size;
	for (size_t i=0; i<size; ++i) {
		rbuf->buffer[i] = gsl_rng_uniform(rand_core);
	}
	rbuf->position = size - 1;
	return;
}

void fill_rand_buffer_gaussian(rand_buffer*const RSTRCT rbuf) {
	assert(rbuf);
	assert(rbuf->buffer);
	assert(rbuf->buffer_size > 0);
	const size_t size = rbuf->buffer_size;
	for (size_t i=0; i<size; ++i) {
		rbuf->buffer[i] = gsl_ran_gaussian(rand_core, 1.0);
	}
	rbuf->position = size - 1;
	return;
}

double rand_flat(rand_buffer*const RSTRCT rbuf) {
	assert(rbuf);
	assert(rbuf->buffer);
	assert(rbuf->buffer_size > 0);
	switch (rbuf->position) {
		case -1:
			fill_rand_buffer_flat(rbuf);
			break;
	}
	return rbuf->buffer[(rbuf->position)--];
}

double rand_gaussian(rand_buffer*const RSTRCT rbuf) {
	assert(rbuf);
	assert(rbuf->buffer);
	assert(rbuf->buffer_size > 0);
	switch (rbuf->position) {
		case -1:
			fill_rand_buffer_gaussian(rbuf);
			break;
	}
	return rbuf->buffer[(rbuf->position)--];
}

typedef double coord;
const size_t coord_size = sizeof(coord);
const size_t three_coord_size = 3*coord_size;
const size_t coord_ptr_size = sizeof(coord*);

typedef ssize_t t_index;
const size_t t_index_size = sizeof(t_index);
const size_t t_index_ptr_size = sizeof(t_index*);

typedef struct polygroup1 {
	coord** r_coords;
	coord** n_coords;
	size_t* c_count;
	size_t* c_start;
	size_t* c_allocated;
	size_t p_count;
	#if KEEP_C_LIFETIMES == 1
	t_index tsteps;
	t_index** life_tsteps;
	#endif
} polygroup1;

const size_t polygroup1_size = sizeof(polygroup1);

void alloc_polygroup1(polygroup1*const RSTRCT pg, const size_t poly_count_in, \
  const size_t c_allocated_in) {
	assert(pg);
	assert(poly_count_in > 0);
	assert(c_allocated_in > 0);
	assert(!(pg->r_coords));
	assert(!(pg->n_coords));
	assert(!(pg->c_count));
	assert(!(pg->c_start));
	assert(!(pg->c_allocated));
	#if KEEP_C_LIFETIMES == 1
	assert(!(pg->life_tsteps));
	#endif
	pg->r_coords = (coord**) malloc(poly_count_in*coord_ptr_size);
	for (size_t i=0; i<poly_count_in; ++i) {
		pg->r_coords[i] = (coord*) malloc(c_allocated_in*coord_size);
	}
	pg->n_coords = (coord**) malloc(poly_count_in*coord_ptr_size);
	for (size_t i=0; i<poly_count_in; ++i) {
		pg->n_coords[i] = (coord*) malloc(c_allocated_in*coord_size);
	}
	pg->c_count = (size_t*) malloc(poly_count_in*size_t_size);
	pg->c_start = (size_t*) malloc(poly_count_in*size_t_size);
	pg->c_allocated = (size_t*) malloc(poly_count_in*size_t_size);
	for (size_t i=0; i<poly_count_in; ++i) {
		pg->c_allocated[i] = c_allocated_in;
	}
	pg->p_count = poly_count_in;
	pg->tsteps = 0;
	#if KEEP_C_LIFETIMES == 1
	pg->life_tsteps = (t_index**) malloc(poly_count_in*t_index_ptr_size);
	for (size_t i=0; i<c_allocated_in; ++i) {
		pg->life_tsteps[i] = (t_index*) malloc(c_allocated_in*t_index_size);
	}
	#endif
	return;
}

void free_polygroup1(polygroup1*const RSTRCT pg) {
	assert(pg);
	assert(pg->p_count > 0);
	assert(pg->r_coords);
	assert(pg->n_coords);
	assert(pg->c_count);
	assert(pg->c_start);
	assert(pg->c_allocated);
	const size_t pc = pg->p_count;
	for (size_t i=0; i<pc; ++i) {
		free(pg->r_coords[i]);
	}
	free(pg->r_coords);
	pg->r_coords = NULL;
	for (size_t i=0; i<pc; ++i) {
		free(pg->n_coords[i]);
	}
	free(pg->n_coords);
	pg->n_coords = NULL;
	free(pg->c_count);
	pg->c_count = NULL;
	free(pg->c_start);
	pg->c_start = NULL;
	free(pg->c_allocated);
	pg->c_allocated = NULL;
	pg->p_count = 0;
	pg->tsteps = -1;
	#if KEEP_C_LIFETIMES == 1
	for (size_t i=0; i<pc; ++i) {
		free(pg->life_tsteps[i]);
	}
	free(pg->life_tsteps);
	pg->life_tsteps = NULL;
	#endif
	return;
}

void increase_c_alloc(polygroup1*const RSTRCT pg, const size_t p_index) {
	assert(pg);
	assert(pg->c_allocated);
	const size_t new_alloc = ALLOC_INCREASE_FACTOR*pg->c_allocated[p_index];
	assert(new_alloc <= MAX_C_ALLOC);
	pg->r_coords[p_index] = \
	  (coord*) realloc(pg->r_coords[p_index], new_alloc*three_coord_size);
	pg->n_coords[p_index] = \
  	  (coord*) realloc(pg->n_coords[p_index], new_alloc*coord_size);
	#if KEEP_C_LIFETIMES == 1
	pg->life_tsteps[p_index] = \
	  (t_index*) realloc(pg->life_tsteps[p_index], new_alloc*t_index_size);
	#endif
	pg->c_allocated[p_index] = new_alloc;
	return;
}

void recenter_constraints(polygroup1*const RSTRCT pg, const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	const size_t new_start = \
	  (pg->c_allocated[p_index]/2) - (pg->c_count[p_index]/2);
	memmove(pg->r_coords[p_index] + (3*new_start), \
	  pg->r_coords[p_index] + (3*pg->c_start[p_index]), \
	  pg->c_count[p_index]*three_coord_size);
	memmove(pg->n_coords[p_index] + new_start, \
	  pg->n_coords[p_index] + pg->c_start[p_index], \
	  pg->c_count[p_index]*coord_size);
	#if KEEP_C_LIFETIMES == 1
	memmove(pg->life_tsteps[p_index] + new_start, \
	  pg->life_tsteps[p_index] + pg->c_start[p_index], \
	  pg->c_count[p_index]*t_index_size);
	#endif
	pg->c_start[p_index] = new_start;
	return;
}

int create_constraint(polygroup1*const RSTRCT pg, const size_t p_index, \
  const ssize_t c_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(c_index <= pg->c_count[p_index]);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	if (((double) (pg->c_count[p_index] + 1)) > \
	  ((double) pg->c_allocated[p_index])/REALLOC_THRESHOLD) {
		increase_c_alloc(pg, p_index);
	}
	if ((pg->c_start[p_index] == 0) || \
	  (pg->c_start[p_index] + pg->c_count[p_index] == \
	  pg->c_allocated[p_index])) {
		recenter_constraints(pg, p_index);
	}
	switch (c_index) {
		case 0:
			break;
		case -1:
			break;
		default:
			break;
	}
	return 0;
}

int remove_constraint(polygroup1*const RSTRCT pg, const size_t p_index, \
  const ssize_t c_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(c_index < pg->c_count[p_index]);
	if (pg->c_count[p_index] < 2) {
		return 1;
	}
	switch (c_index) {
		case 0:
			++(pg->c_start[p_index]);
			break;
		case -1:
			break;
		default: {
			//This should be changed to move either up or down, depending on
			//which section is shorter.
			coord*const r_coord = pg->r_coords[p_index] + 3*c_index;
			memmove(r_coord, r_coord + 1, \
			  (pg->c_count[p_index] - c_index - 1)*three_coord_size);
			coord*const n_coord = pg->n_coords[p_index] + c_index;
			memmove(n_coord, n_coord + 1, \
			  (pg->c_count[p_index] - c_index - 1)*coord_size);
			#if KEEP_C_LIFETIMES == 1
			t_index*const life_coord = pg->life_tsteps[p_index] + c_index;
			memmove(life_coord, life_coord + 1, \
			  (pg->c_count[p_index] - c_index - 1)*t_index_size);
			#endif
			break;
		}
	}
	--(pg->c_count[p_index]);
	return 0;
}

typedef double dparam;

typedef struct dynamic_params {
	dparam gamma;
	dparam zeta;
	coord cd_length;
	dparam delta_t;
} dynamic_params;

const size_t dynamic_params_size = sizeof(dynamic_params);


int main() {

	return 0;
}
