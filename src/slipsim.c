#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define RSTRCT restrict
#define MAX_C_ALLOC 16384
#define MAX_C_COUNT 8192
#define ALLOC_INCREASE_FACTOR 2
#define REALLOC_THRESHOLD 0.5
#define INITIAL_C_ALLOC 1024
#define ENABLE_REALLOC 1
#define LEFT_ALIGN 0
#define KEEP_C_LIFETIMES 1

#define ARC_MAX 100.0
#define ALPHA 10.0
#define ARC_CREATE_INTERVAL 10.0
#define MONOMER_LENGTH 1.0
//Below is Sqrt(2 kb T / zeta)
#define ARC_DIFFUSION_STDDEV 0.01
//Below is b/Sqrt(3), it is not adjustable.
#define SPACE_STDDEV MONOMER_LENGTH*0.5773502691896258


const size_t size_t_size = sizeof(size_t);
const size_t double_size = sizeof(double);

gsl_rng* rand_core;

typedef struct rand_buffer {
	size_t buffer_size;
	ssize_t position;
	double* buffer;
} rand_buffer;

const size_t rand_buffer_size = sizeof(rand_buffer);

rand_buffer* global_flat_rbuf;
rand_buffer* global_gaussian_rbuf;

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

double rand_flat(void) {
	assert(global_flat_rbuf);
	assert(global_flat_rbuf->buffer);
	assert(global_flat_rbuf->buffer_size > 0);
	if (global_flat_rbuf->position == -1) {
		fill_rand_buffer_flat(global_flat_rbuf);
	}
	return global_flat_rbuf->buffer[(global_flat_rbuf->position)--];
}

double rand_gaussian(void) {
	assert(global_gaussian_rbuf);
	assert(global_gaussian_rbuf->buffer);
	assert(global_gaussian_rbuf->buffer_size > 0);
	if (global_gaussian_rbuf->position == -1) {
		fill_rand_buffer_gaussian(global_gaussian_rbuf);
	}
	return global_gaussian_rbuf->buffer[(global_gaussian_rbuf->position)--];
}

typedef double coord;
const size_t coord_size = sizeof(coord);
const size_t three_coord_size = 3*coord_size;
const size_t coord_ptr_size = sizeof(coord*);
#define coord_min DBL_TRUE_MIN

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
	t_index tsteps;
	#if KEEP_C_LIFETIMES == 1
	t_index** life_tsteps;
	#endif
} polygroup1;

const size_t polygroup1_size = sizeof(polygroup1);

void alloc_polygroup1(polygroup1*const RSTRCT pg, const size_t poly_count_in,
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

void recenter_constraints_if_needed(polygroup1*const RSTRCT pg, \
                                    const size_t p_index) {
	if ((pg->c_start[p_index] == 0) || \
	  (pg->c_start[p_index] + pg->c_count[p_index] == \
	  pg->c_allocated[p_index])) {
		recenter_constraints(pg, p_index);
	}
	return;
}

#if ENABLE_REALLOC == 1
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

static inline int realloc_if_needed(polygroup1*const RSTRCT pg,
                                    const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	int performed_realloc = 0;
	if (((double) (pg->c_count[p_index] + 1)) > \
	  ((double) pg->c_allocated[p_index])*REALLOC_THRESHOLD) {
		increase_c_alloc(pg, p_index);
		performed_realloc = 1;
	}
	return performed_realloc;
}
#endif

void copy_in_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                        const size_t address,
                        const coord*const RSTRCT coords_in) {
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(address < pg->c_allocated[p_index]);
	assert(coords_in[3] < ARC_MAX);
	memcpy(pg->r_coords + 3*address, coords_in, three_coord_size);
	pg->n_coords[p_index][address] = coords_in[3];
	return;
}

int create_low_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                          const coord*const RSTRCT coords_in) {
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(coords_in[3] < ARC_MAX);
	recenter_constraints_if_needed(pg, p_index);
	#if ENABLE_REALLOC == 1
	realloc_if_needed(pg, p_index);
	#else
	assert(pg->c_count[p_index] < pg->c_allocated[p_index]);
	#endif
	--(pg->c_start[p_index]);
	const size_t address = pg->c_start[p_index];
	copy_in_constraint(pg, p_index, address, coords_in);
	++(pg->c_count[p_index]);
	return 0;
}

int create_high_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                           const coord*const RSTRCT coords_in) {
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(coords_in[3] < ARC_MAX);
	recenter_constraints_if_needed(pg, p_index);
	#if ENABLE_REALLOC == 1
	realloc_if_needed(pg, p_index);
	#else
	assert(pg->c_count[p_index] < pg->c_allocated[p_index]);
	#endif
	const size_t address = pg->c_start[p_index] + pg->c_count[p_index];
	copy_in_constraint(pg, p_index, address, coords_in);
	++(pg->c_count[p_index]);
	return 0;
}

int create_mid_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                          const size_t c_index,
                          const coord*const RSTRCT coords_in) {
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(c_index > 0);
	assert(c_index < (pg->c_count[p_index] - 1));
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(coords_in[3] < ARC_MAX);
	recenter_constraints_if_needed(pg, p_index);
	#if ENABLE_REALLOC == 1
	realloc_if_needed(pg, p_index);
	#else
	assert(pg->c_count[p_index] < pg->c_allocated[p_index]);
	#endif
	const size_t address = pg->c_start[p_index] + c_index;
	const size_t half_count = (pg->c_count[p_index])/2;
	if (c_index > half_count) {  //This could probably be made branchless.
		coord*const r_ptr = pg->r_coords[p_index] + 3*address;
		coord*const n_ptr = pg->n_coords[p_index] + address;
		const size_t move_count = pg->c_count[p_index] - c_index;
		memmove(r_ptr + 3, r_ptr, move_count*three_coord_size);
		memmove(n_ptr + 1, n_ptr, move_count*coord_size);
	}
	else {
		--(pg->c_start[p_index]);
		coord*const r_ptr = pg->r_coords[p_index] + 3*(pg->c_start[p_index]);
		coord*const n_ptr = pg->n_coords[p_index] + pg->c_start[p_index];
		const size_t move_count = c_index - pg->c_start[p_index];
		memmove(r_ptr - 3, r_ptr, move_count*three_coord_size);
		memmove(n_ptr - 1, n_ptr, move_count*coord_size);
	}
	copy_in_constraint(pg, p_index, address, coords_in);
	++(pg->c_count[p_index]);
	return 0;
}

int remove_low_constraint(polygroup1*const RSTRCT pg, const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	if (pg->c_count[p_index] < 2) {
		return 1;
	}
	++(pg->c_start[p_index]);
	--(pg->c_count[p_index]);
	return 0;
}

int remove_high_constraint(polygroup1*const RSTRCT pg, const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	if (pg->c_count[p_index] < 2) {
		return 1;
	}
	--(pg->c_count[p_index]);
	return 0;
}

int remove_mid_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                          const size_t c_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	if (pg->c_count[p_index] < 2) {
		return 1;
	}
	assert(c_index < (pg->c_count[p_index] - 1));
	size_t orig_address;
	int delta;
	size_t move_length;
	const size_t half_count_less_one = (pg->c_count[p_index])/2 - 1;
	if (c_index > half_count_less_one) {
		orig_address = pg->c_start[p_index] + c_index + 1;
		delta = -1;
		move_length = (pg->c_count[p_index] - c_index) - 1;
	}
	else {
		orig_address = pg->c_start[p_index];
		delta = 1;
		move_length = c_index;
	}
	coord*const r_coord_orig = pg->r_coords[p_index] + 3*orig_address;
	memmove(r_coord_orig + 3*delta, r_coord_orig, move_length*three_coord_size);
	coord*const n_coord_orig = pg->n_coords[p_index] + orig_address;
	memmove(n_coord_orig + delta, n_coord_orig, move_length*coord_size);
	#if KEEP_C_LIFETIMES == 1
	t_index*const life_coord_orig = pg->life_tsteps[p_index] + orig_address;
	memmove(life_coord_orig + delta, life_coord_orig, move_length*t_index_size);
	#endif
	--(pg->c_count[p_index]);
	return 0;
}

int mc_move_create_low_constraint(polygroup1*const RSTRCT pg,
                                  const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	const coord strand_length = \
	  rand_flat()*(pg->n_coords[p_index][pg->c_start[p_index]]);
	coord coords_new[4];
	coords_new[3] = strand_length;
	if (strand_length < ALPHA) {
		if (rand_flat() > ((double) strand_length)/ALPHA) {
			return 1;
		}
	}
	const coord*const r_coords = pg->r_coords[p_index];
	const size_t index = pg->c_start[p_index];
	const coord space_stddev_strand = SPACE_STDDEV*sqrt(strand_length);
	coords_new[0] = space_stddev_strand*rand_gaussian() + r_coords[index];
	coords_new[1] = space_stddev_strand*rand_gaussian() + r_coords[index + 1];
	coords_new[2] = space_stddev_strand*rand_gaussian() + r_coords[index + 2];
	create_low_constraint(pg, p_index, coords_new);
	return 0;
}

int mc_move_create_high_constraint(polygroup1*const RSTRCT pg,
                                   const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	const coord strand_length = rand_flat()*(ARC_MAX - \
	  (pg->n_coords[p_index][pg->c_start[p_index] + pg->c_count[p_index] - 1]));
	coord coords_new[4];
	coords_new[3] = ARC_MAX - strand_length;
	if (strand_length < ALPHA) {
		if (rand_flat() > ((double) strand_length)/ALPHA) {
			return 1;
		}
	}
	const coord*const r_coords = pg->r_coords[p_index];
	const size_t index = pg->c_start[p_index];
	const coord space_stddev_strand = SPACE_STDDEV*sqrt(strand_length);
	coords_new[0] = space_stddev_strand*rand_gaussian() + r_coords[index];
	coords_new[1] = space_stddev_strand*rand_gaussian() + r_coords[index + 1];
	coords_new[2] = space_stddev_strand*rand_gaussian() + r_coords[index + 2];
	create_high_constraint(pg, p_index, coords_new);
	return 0;
}

int move_arc_across_low_constraint(polygroup1*const RSTRCT pg,
                                   const size_t p_index,
                                   const coord delta_arc) {
	//This should be the function used if there is only one constraint.
	assert(pg);
	assert(p_index < pg->p_count);
	const size_t index = pg->c_start[p_index];
	pg->n_coords[p_index][index] += delta_arc;
	assert(pg->n_coords[p_index][index] > 0.0);
	assert(pg->n_coords[p_index][index] < ARC_MAX);
	if (pg->c_count[p_index] > 1) {
		assert(pg->n_coords[p_index][index] < pg->n_coords[p_index][index + 1]);
	}
	return 0;
}

int move_arc_across_high_constraint(polygroup1*const RSTRCT pg,
                                    const size_t p_index,
                                    const coord delta_arc) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const size_t index = pg->c_start[p_index] + pg->c_count[p_index] - 1;
	pg->n_coords[p_index][index] += delta_arc;
	assert(pg->n_coords[p_index][index] > 0.0);
	assert(pg->n_coords[p_index][index] < ARC_MAX);
	assert(pg->n_coords[p_index][index] > pg->n_coords[p_index][index - 1]);
	return 0;
}

int move_arc_across_mid_constraint(polygroup1*const RSTRCT pg,
                                   const size_t p_index,
                                   const size_t c_index,
                                   const coord delta_arc) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 2);
	assert(c_index > 0);
	assert(c_index < (pg->c_count[p_index] - 1));
	const size_t index = pg->c_start[p_index] + c_index;
	pg->n_coords[p_index][index] += delta_arc;
	assert(pg->n_coords[p_index][index] > 0.0);
	assert(pg->n_coords[p_index][index] < ARC_MAX);
	assert(pg->n_coords[p_index][index] < pg->n_coords[p_index][index + 1]);
	assert(pg->n_coords[p_index][index] > pg->n_coords[p_index][index - 1]);
	return 0;
}

int mc_move_arc_across_low_constraint(polygroup1*const RSTRCT pg,
                                      const size_t p_index) {
	//This should be the function used if there is only one constraint.
	assert(pg);
	assert(p_index < pg->p_count);
	const size_t index = pg->c_start[p_index];
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord high_arc_limit = \
	  (pg->c_count[p_index] == 1) ? ARC_MAX : pg->n_coords[p_index][index + 1];
	if (!(new_arc_coord > 0.0) || !(new_arc_coord < high_arc_limit)) {
		return 2;
	}
	//MC roll.
	return 0;
}

int mc_move_arc_across_high_constraint(polygroup1*const RSTRCT pg,
                                       const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const size_t index = pg->c_start[p_index] + pg->c_count[p_index] - 1;
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord low_arc_limit = pg->n_coords[p_index][index - 1];
	if (!(new_arc_coord > low_arc_limit) || !(new_arc_coord < ARC_MAX)) {
		return 2;
	}
	//MC roll.
	return 0;
}

int mc_move_arc_across_mid_constraint(polygroup1*const RSTRCT pg,
                                      const size_t p_index,
                                      const size_t c_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 2);
	assert(c_index > 0);
	assert(c_index < (pg->c_count[p_index] - 1));
	const size_t index = pg->c_start[p_index] + c_index;
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord low_arc_limit = pg->n_coords[p_index][index - 1];
	const coord high_arc_limit = pg->n_coords[p_index][index + 1];
	if (!(new_arc_coord > low_arc_limit) || !(new_arc_coord < high_arc_limit)) {
		return 2;
	}
	//MC roll.
	return 0;
}


int main() {

	return 0;
}
