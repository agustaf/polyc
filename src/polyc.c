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
#define RAND_CHECK_BUFFER 1
#define MAX_C_ALLOC 16384
#define MAX_C_COUNT 8192
#define ALLOC_INCREASE_FACTOR 2
#define REALLOC_THRESHOLD 0.5
#define INITIAL_C_ALLOC 1024
#define ENABLE_REALLOC 0
#define LEFT_ALIGN 0
#define KEEP_C_LIFETIMES 1

#define ARC_MAX 100.0
#define ALPHA 10.0
#define ARC_CREATE_END_INTERVAL 10.0
#define MONOMER_LENGTH 1.0
//Below is the minimum strand arc length (moves creating smaller strands
//will be rejected).
#define ARC_MIN ALPHA/10000000.0
#define ARC_CE_DELTA (ARC_CREATE_END_INTERVAL - ARC_MIN)

//Below is Sqrt(2 kb T / zeta)
#define ARC_DIFFUSION_STDDEV 0.01
//Below is b/Sqrt(3), it is not adjustable.
#define SPACE_STDDEV MONOMER_LENGTH*0.5773502691896258
//Below is 3/(2b*b), it is not adjustable.
#define THREE_BY_TWO_BSQ 1.5/(MONOMER_LENGTH*MONOMER_LENGTH)


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
	for (size_t i=(rbuf->position + 1); i<size; ++i) {
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
	for (size_t i=(rbuf->position + 1); i<size; ++i) {
		rbuf->buffer[i] = gsl_ran_gaussian(rand_core, 1.0);
	}
	rbuf->position = size - 1;
	return;
}

double rand_flat(void) {
	assert(global_flat_rbuf);
	assert(global_flat_rbuf->buffer);
	assert(global_flat_rbuf->buffer_size > 0);
	#if RAND_CHECK_BUFFER == 1
	if (global_flat_rbuf->position == -1) {
		fill_rand_buffer_flat(global_flat_rbuf);
	}
	#endif
	return global_flat_rbuf->buffer[(global_flat_rbuf->position)--];
}

double rand_gaussian(void) {
	assert(global_gaussian_rbuf);
	assert(global_gaussian_rbuf->buffer);
	assert(global_gaussian_rbuf->buffer_size > 0);
	#if RAND_CHECK_BUFFER == 1
	if (global_gaussian_rbuf->position == -1) {
		fill_rand_buffer_gaussian(global_gaussian_rbuf);
	}
	#endif
	return global_gaussian_rbuf->buffer[(global_gaussian_rbuf->position)--];
}

void fill_from_zero_size_t(size_t*const RSTRCT array, const size_t length) {
	assert(array);
	assert(length > 0);
	for (size_t i=0; i<length; ++i) {
		array[i] = i;
	}
	return;
}

size_t random_select_and_pop_size_t(size_t*const RSTRCT array, \
  const size_t length) {
	assert(array);
	assert(length > 0);
	const size_t index = (size_t) (length*rand_flat());
	const size_t result = array[index];
	size_t*const array_ptr = array + index;
	memmove(array_ptr, array_ptr + 1, ((length - index) - 1)*size_t_size);
	return result;
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
	size_t id;
	#if KEEP_C_LIFETIMES == 1
	t_index** life_tsteps;
	#endif
} polygroup1;

const size_t polygroup1_size = sizeof(polygroup1);

void alloc_polygroup1_members(polygroup1*const RSTRCT pg,
                              const size_t poly_count_in,
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
		pg->r_coords[i] = (coord*) malloc(c_allocated_in*three_coord_size);
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
	for (size_t i=0; i<poly_count_in; ++i) {
		pg->life_tsteps[i] = (t_index*) malloc(c_allocated_in*t_index_size);
	}
	#endif
	return;
}

polygroup1* create_polygroup1(const size_t poly_count_in, \
  const size_t c_allocated_in, const size_t id_in) {
	assert(poly_count_in > 0);
	assert(c_allocated_in > 0);
	polygroup1* pg = (polygroup1*) malloc(polygroup1_size);
	pg->r_coords = NULL;
	pg->n_coords = NULL;
	pg->c_count = NULL;
	pg->c_start = NULL;
	pg->c_allocated = NULL;
	#if KEEP_C_LIFETIMES == 1
	pg->life_tsteps = NULL;
	pg->id = id_in;
	#endif
	alloc_polygroup1_members(pg, poly_count_in, c_allocated_in);
	return pg;
}

void free_polygroup1_members(polygroup1*const RSTRCT pg) {
	assert(pg);
	assert(pg->p_count > 0);
	assert(pg->r_coords);
	assert(pg->n_coords);
	assert(pg->c_count);
	assert(pg->c_start);
	assert(pg->c_allocated);
	const size_t p_count = pg->p_count;
	for (size_t i=0; i<p_count; ++i) {
		free(pg->r_coords[i]);
	}
	free(pg->r_coords);
	pg->r_coords = NULL;
	for (size_t i=0; i<p_count; ++i) {
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
	for (size_t i=0; i<p_count; ++i) {
		free(pg->life_tsteps[i]);
	}
	free(pg->life_tsteps);
	pg->life_tsteps = NULL;
	#endif
	return;
}

size_t total_constraints(const polygroup1*const RSTRCT pg) {
	assert(pg);
	const size_t p_count = pg->p_count;
	size_t c_sum = pg->c_count[0];
	for (size_t i=1; i<p_count; ++i) {
		c_sum += pg->c_count[i];
	}
	return c_sum;
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
	assert(pg);
	assert(p_index < pg->p_count);
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
	/*This function copies r_coords and n_coords values of a new constraint to
	a specified p_index and c_index. It does not enforce the arc-length
	ordering of the constraints, and it does not adjust the constraint counts.
	It will overwrite another constraint at the specified location.*/
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(address < pg->c_allocated[p_index]);
	assert(coords_in[3] < ARC_MAX - ARC_MIN);
	assert(coords_in[3] > ARC_MIN);
	memcpy(pg->r_coords[p_index] + 3*address, coords_in, three_coord_size);
	pg->n_coords[p_index][address] = coords_in[3];
	return;
}

int create_low_constraint(polygroup1*const RSTRCT pg, const size_t p_index,
                          const coord*const RSTRCT coords_in) {
	assert(pg);
	assert(coords_in);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	assert(coords_in[3] > ARC_MIN);
	const size_t old_low_address = pg->c_start[p_index];
	assert(coords_in[3] + ARC_MIN < pg->n_coords[p_index][old_low_address]);
	recenter_constraints_if_needed(pg, p_index);
	#if ENABLE_REALLOC == 1
	realloc_if_needed(pg, p_index);
	#else
	assert(pg->c_count[p_index] < pg->c_allocated[p_index]);
	#endif
	const size_t address = --(pg->c_start[p_index]);
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
	assert(coords_in[3] < ARC_MAX - ARC_MIN);
	const size_t old_high_address = \
	  pg->c_start[p_index] + pg->c_count[p_index] - 1;
	assert(coords_in[3] - ARC_MIN > pg->n_coords[p_index][old_high_address]);
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
	size_t address = pg->c_start[p_index] + c_index;
	assert(coords_in[3] + ARC_MIN < pg->n_coords[p_index][address]);
	assert(coords_in[3] - ARC_MIN > pg->n_coords[p_index][address - 1]);
	recenter_constraints_if_needed(pg, p_index);
	#if ENABLE_REALLOC == 1
	realloc_if_needed(pg, p_index);
	#else
	assert(pg->c_count[p_index] < pg->c_allocated[p_index]);
	#endif
	const size_t high_length = pg->c_count[p_index] - c_index;
	const size_t move_low = (size_t) (c_index < high_length);
	const size_t move_length = high_length + move_low*(c_index - high_length);
	assert(move_length > 0);
	const size_t initial = address - move_low*c_index;
	const size_t final = initial + 1 - move_low*2;
	coord*const n_ptr = pg->n_coords[p_index];
	coord*const r_ptr = pg->r_coords[p_index];
	memmove(n_ptr + final, n_ptr + initial, move_length*coord_size);
	memmove(r_ptr + 3*final, r_ptr + 3*initial, move_length*three_coord_size);
	pg->c_start[p_index] -= move_low;
	address -= move_low;
	copy_in_constraint(pg, p_index, address, coords_in);
	++(pg->c_count[p_index]);
	#if KEEP_C_LIFETIMES == 1
	t_index*const l_ptr = pg->life_tsteps[p_index];
	memmove(l_ptr + final, l_ptr + initial, move_length*t_index_size);
	l_ptr[address] = 0;
	#endif
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
	assert(c_index > 0);
	assert(c_index < (pg->c_count[p_index] - 1));
	const size_t high_length = pg->c_count[p_index] - c_index - 1;
	const size_t move_low = (size_t) (c_index < high_length);
	const size_t move_length = high_length + move_low*(c_index - high_length);
	assert(move_length > 0);
	const size_t address = pg->c_start[p_index] + c_index;
	const size_t initial = address + 1 - move_low*(c_index + 1);
	const size_t final = initial - 1 + move_low*2;
	coord*const n_ptr = pg->n_coords[p_index];
	coord*const r_ptr = pg->r_coords[p_index];
	memmove(n_ptr + final, n_ptr + initial, move_length*coord_size);
	memmove(r_ptr + 3*final, r_ptr + 3*initial, move_length*three_coord_size);
	pg->c_start[p_index] += move_low;
	--(pg->c_count[p_index]);
	#if KEEP_C_LIFETIMES == 1
	t_index*const l_ptr = pg->life_tsteps[p_index];
	memmove(l_ptr + final, l_ptr + initial, move_length*t_index_size);
	#endif
	return 0;
}

int mc_move_create_low_constraint(polygroup1*const RSTRCT pg,
                                  const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	const coord arc_low = \
	  pg->n_coords[p_index][pg->c_start[p_index]] - 2.0*ARC_MIN;
	const coord create_length = \
	  (arc_low < ARC_CE_DELTA) ? arc_low : ARC_CE_DELTA;
	const coord strand_length = rand_flat()*create_length + ARC_MIN;
	if (strand_length < ALPHA) {
		if (rand_flat() > ((double) strand_length)/ALPHA) {
			return 1;
		}
	}
	const coord*const r_coords = pg->r_coords[p_index];
	const size_t index = pg->c_start[p_index];
	const coord space_stddev_strand = SPACE_STDDEV*sqrt(strand_length);
	const coord coords_new[4] = {
		space_stddev_strand*rand_gaussian() + r_coords[index],
		space_stddev_strand*rand_gaussian() + r_coords[index + 1],
		space_stddev_strand*rand_gaussian() + r_coords[index + 2],
		strand_length
	};
	create_low_constraint(pg, p_index, coords_new);
	return 0;
}

int mc_move_create_high_constraint(polygroup1*const RSTRCT pg,
                                   const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] < MAX_C_COUNT);
	const size_t end_index = pg->c_start[p_index] + pg->c_count[p_index] - 1;
	const coord arc_high = \
	  (ARC_MAX - pg->n_coords[p_index][end_index]) - 2*ARC_MIN;
	const coord create_length = \
	  (arc_high < ARC_CE_DELTA) ? arc_high : ARC_CE_DELTA;
	const coord strand_length = rand_flat()*create_length + ARC_MIN;
	if (strand_length < ALPHA) {
		if (rand_flat() > ((double) strand_length)/ALPHA) {
			return 1;
		}
	}
	const coord*const r_coords = pg->r_coords[p_index];
	const size_t index = pg->c_start[p_index];
	const coord space_stddev_strand = SPACE_STDDEV*sqrt(strand_length);
	const coord coords_new[4] = {
		space_stddev_strand*rand_gaussian() + r_coords[index],
		space_stddev_strand*rand_gaussian() + r_coords[index + 1],
		space_stddev_strand*rand_gaussian() + r_coords[index + 2],
		ARC_MAX - strand_length
	};
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
	const coord new_arc_loc = pg->n_coords[p_index][index] + delta_arc;
	assert(new_arc_loc > ARC_MIN);
	assert(new_arc_loc < ARC_MAX - ARC_MIN);
	if (pg->c_count[p_index] > 1) {
		assert(new_arc_loc + ARC_MIN < pg->n_coords[p_index][index + 1]);
	}
	pg->n_coords[p_index][index] = new_arc_loc;
	return 0;
}

int move_arc_across_high_constraint(polygroup1*const RSTRCT pg,
                                    const size_t p_index,
                                    const coord delta_arc) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const size_t index = pg->c_start[p_index] + pg->c_count[p_index] - 1;
	const coord new_arc_loc = pg->n_coords[p_index][index] + delta_arc;
	assert(new_arc_loc < ARC_MAX - ARC_MIN);
	assert(new_arc_loc - ARC_MIN > pg->n_coords[p_index][index - 1]);
	pg->n_coords[p_index][index] = new_arc_loc;
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
	const coord new_arc_loc = pg->n_coords[p_index][index] + delta_arc;
	assert(new_arc_loc + ARC_MIN < pg->n_coords[p_index][index + 1]);
	assert(new_arc_loc - ARC_MIN > pg->n_coords[p_index][index - 1]);
	pg->n_coords[p_index][index] = new_arc_loc;
	return 0;
}

coord q_squared(const polygroup1*const RSTRCT pg, const size_t p_index, \
  const size_t c_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	assert(c_index < pg->c_count[p_index] - 1);
	const coord*const r_ptr = pg->r_coords[p_index];
	const size_t i = 3*(c_index + pg->c_start[p_index]);
	const coord x_delta = r_ptr[i + 3] - r_ptr[i];
	const coord y_delta = r_ptr[i + 4] - r_ptr[i + 1];
	const coord z_delta = r_ptr[i + 5] - r_ptr[i + 2];
	return x_delta*x_delta + y_delta*y_delta + z_delta*z_delta;
}

coord q_squared_low_strand(const polygroup1*const RSTRCT pg, \
  const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const coord*const r_ptr = pg->r_coords[p_index];
	const size_t i = 3*(pg->c_start[p_index]);
	const coord x_delta = r_ptr[i + 3] - r_ptr[i];
	const coord y_delta = r_ptr[i + 4] - r_ptr[i + 1];
	const coord z_delta = r_ptr[i + 5] - r_ptr[i + 2];
	return x_delta*x_delta + y_delta*y_delta + z_delta*z_delta;
}

coord q_squared_high_strand(const polygroup1*const RSTRCT pg, \
  const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 2);
	const coord*const r_ptr = pg->r_coords[p_index];
	const size_t i = 3*(pg->c_start[p_index] + pg->c_count[p_index] - 2);
	const coord x_delta = r_ptr[i + 3] - r_ptr[i];
	const coord y_delta = r_ptr[i + 4] - r_ptr[i + 1];
	const coord z_delta = r_ptr[i + 5] - r_ptr[i + 2];
	return x_delta*x_delta + y_delta*y_delta + z_delta*z_delta;
}

int mc_move_arc_across_single_constraint(polygroup1*const RSTRCT pg,
                                         const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] == 1);
	const size_t index = pg->c_start[p_index];
	#if KEEP_C_LIFETIMES == 1
	++(pg->life_tsteps[p_index][index]);
	#endif
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	if (!(new_arc_coord > ARC_MIN) || !(new_arc_coord < ARC_MAX - ARC_MIN)) {
		return 2;
	}
	pg->n_coords[p_index][index] = new_arc_coord;
	return 0;
}

int mc_move_arc_across_low_constraint(polygroup1*const RSTRCT pg,
                                      const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const size_t index = pg->c_start[p_index];
	#if KEEP_C_LIFETIMES == 1
	++(pg->life_tsteps[p_index][index]);
	#endif
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord high_arc_limit = pg->n_coords[p_index][index + 1] - ARC_MIN;
	if (!(new_arc_coord > ARC_MIN) || !(new_arc_coord < high_arc_limit)) {
		return 2;
	}
	//MC roll.
	const coord old_high_strand = \
	  pg->n_coords[p_index][index + 1] - pg->n_coords[p_index][index];
	const coord new_high_strand = \
	  pg->n_coords[p_index][index + 1] - new_arc_coord;
	const coord q_sqr = q_squared_low_strand(pg, p_index);
	const double accept_factor = \
	  pow(old_high_strand/new_high_strand, 1.5)* \
	  exp(THREE_BY_TWO_BSQ*q_sqr*(1.0/old_high_strand - 1.0/new_high_strand));
	if (accept_factor < 1.0) {
		if (rand_flat() > accept_factor) {
			return 1;
		}
	}
	pg->n_coords[p_index][index] = new_arc_coord;
	return 0;
}

int mc_move_arc_across_high_constraint(polygroup1*const RSTRCT pg,
                                       const size_t p_index) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(pg->c_count[p_index] > 1);
	const size_t index = pg->c_start[p_index] + pg->c_count[p_index] - 1;
	#if KEEP_C_LIFETIMES == 1
	++(pg->life_tsteps[p_index][index]);
	#endif
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord low_arc_limit = pg->n_coords[p_index][index - 1] + ARC_MIN;
	if (!(new_arc_coord > low_arc_limit) || \
	    !(new_arc_coord < ARC_MAX - ARC_MIN)) {
		return 2;
	}
	//MC roll.
	const coord old_low_strand = \
	  pg->n_coords[p_index][index] - pg->n_coords[p_index][index - 1];
	const coord new_low_strand = \
	  new_arc_coord - pg->n_coords[p_index][index - 1];
	const coord q_sqr = q_squared_high_strand(pg, p_index);
	const double accept_factor = \
	  pow(old_low_strand/new_low_strand, 1.5)* \
	  exp(THREE_BY_TWO_BSQ*q_sqr*(1.0/old_low_strand - 1.0/new_low_strand));
	if (accept_factor < 1.0) {
		if (rand_flat() > accept_factor) {
			return 1;
		}
	}
	pg->n_coords[p_index][index] = new_arc_coord;
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
	#if KEEP_C_LIFETIMES == 1
	++(pg->life_tsteps[p_index][index]);
	#endif
	const coord new_arc_coord = ARC_DIFFUSION_STDDEV*rand_gaussian() + \
	  pg->n_coords[p_index][index];
	const coord low_arc_limit = pg->n_coords[p_index][index - 1] + ARC_MIN;
	const coord high_arc_limit = pg->n_coords[p_index][index + 1] - ARC_MIN;
	if (!(new_arc_coord > low_arc_limit) || !(new_arc_coord < high_arc_limit)) {
		return 2;
	}
	//MC roll.
	const coord old_high_strand = \
	  pg->n_coords[p_index][index + 1] - pg->n_coords[p_index][index];
	const coord new_high_strand = \
	  pg->n_coords[p_index][index + 1] - new_arc_coord;
	const coord old_low_strand = \
	  pg->n_coords[p_index][index] - pg->n_coords[p_index][index - 1];
	const coord new_low_strand = \
	  new_arc_coord - pg->n_coords[p_index][index - 1];
	const coord q_sqr = q_squared(pg, p_index, c_index);
	const double accept_factor = \
	  pow(old_low_strand*old_high_strand/(new_low_strand*new_high_strand), 1.5)* \
	  exp(THREE_BY_TWO_BSQ*q_sqr*(1.0/old_low_strand - 1.0/new_low_strand + \
	  1.0/old_high_strand - 1.0/new_high_strand));
	if (accept_factor < 1.0) {
		if (rand_flat() > accept_factor) {
			return 1;
		}
	}
	pg->n_coords[p_index][index] = new_arc_coord;
	return 0;
}

void mc_move_arc_across_all_random_order(polygroup1*const RSTRCT pg, \
  const size_t p_index, size_t* index_buffer, const size_t ib_length) {
	assert(pg);
	assert(p_index < pg->p_count);
	assert(index_buffer);
	const size_t c_count = pg->c_count[p_index];
	if (c_count == 1) {
		mc_move_arc_across_single_constraint(pg, p_index);
		return;
	}
	const size_t c_count_less_one = c_count - 1;
	assert(ib_length > c_count);
	fill_from_zero_size_t(index_buffer, c_count);
	for (size_t i=c_count; i>0; --i) {
		const size_t index = \
		  random_select_and_pop_size_t(index_buffer, i);
		if (index == 0) {
			mc_move_arc_across_low_constraint(pg, p_index);
		}
		else if (index == c_count_less_one) {
			mc_move_arc_across_high_constraint(pg, p_index);
		}
		else{
			mc_move_arc_across_mid_constraint(pg, p_index, index);
		}
	}
	return;
}

void mc_move_arc_across_all_sequential(polygroup1*const RSTRCT pg, \
  const size_t p_index) {
	assert(pg);
	assert(pg->p_count > 0);
	if (pg->c_count[p_index] == 1) {
		mc_move_arc_across_single_constraint(pg, p_index);
		return;
	}
	mc_move_arc_across_low_constraint(pg, p_index);
	const size_t c_count = pg->c_count[p_index];
	for (size_t i=0; i<c_count; ++i) {
		mc_move_arc_across_mid_constraint(pg, p_index, i);
	}
	mc_move_arc_across_high_constraint(pg, p_index);
	return;
}

typedef struct polygroup1_history {
	size_t* pg_id;
	size_t* p_index;
	size_t* c_count;
	t_index* t_steps;
	coord* r_coords;
	coord* n_coords;
	size_t p_allocated;
	size_t c_allocated;
	size_t p_entries;
	size_t c_entries;
} polygroup1_history;

const size_t polygroup1_history_size = sizeof(polygroup1_history);

polygroup1_history* create_polygroup1_history(const size_t p_entries_in, \
                                              const size_t c_entries_in) {
	assert(p_entries_in > 0);
	assert(c_entries_in >= p_entries_in);
	polygroup1_history*const pgh = \
	  (polygroup1_history*) malloc(polygroup1_history_size);
	assert(pgh);
	pgh->pg_id = (size_t*) malloc(p_entries_in*size_t_size);
	pgh->p_index = (size_t*) malloc(p_entries_in*size_t_size);
	pgh->c_count = (size_t*) malloc(p_entries_in*size_t_size);
	pgh->t_steps = (t_index*) malloc(p_entries_in*t_index_size);
	pgh->r_coords = (coord*) malloc(c_entries_in*three_coord_size);
	pgh->n_coords = (coord*) malloc(c_entries_in*coord_size);
	pgh->p_allocated = p_entries_in;
	pgh->c_allocated = c_entries_in;
	pgh->p_entries = 0;
	pgh->c_entries = 0;
	assert(pgh->pg_id);
	assert(pgh->p_index);
	assert(pgh->c_count);
	assert(pgh->t_steps);
	assert(pgh->r_coords);
	assert(pgh->n_coords);
	return pgh;
}

void free_polygroup1_history_members(polygroup1_history*const RSTRCT pgh) {
	assert(pgh);
	assert(pgh->pg_id);
	assert(pgh->p_index);
	assert(pgh->c_count);
	assert(pgh->t_steps);
	assert(pgh->r_coords);
	assert(pgh->n_coords);
	free(pgh->pg_id);
	pgh->pg_id = NULL;
	free(pgh->p_index);
	pgh->p_index = NULL;
	free(pgh->c_count);
	pgh->c_count = NULL;
	free(pgh->t_steps);
	pgh->t_steps = NULL;
	free(pgh->r_coords);
	pgh->r_coords = NULL;
	free(pgh->n_coords);
	pgh->n_coords = NULL;
	pgh->p_allocated = 0;
	pgh->c_allocated = 0;
	pgh->p_entries = 0;
	pgh->c_entries = 0;
	return;
}

void poll_polygroup1_history(const polygroup1*const RSTRCT pg, \
                             polygroup1_history*const RSTRCT pgh) {
	assert(pg);
	assert(pgh);
	const size_t p_count = pg->p_count;
	assert(pgh->p_entries + p_count <= pgh->p_allocated);
	const size_t c_sum = total_constraints(pg);
	assert(pgh->c_entries + c_sum < pgh->c_allocated);
	for (size_t i=0; i<p_count; ++i, ++(pgh->p_entries)) {
		const size_t pe = pgh->p_entries;
		pgh->pg_id[pe] = pg->id;
		pgh->p_index[pe] = i;
		pgh->t_steps[pe] = pg->tsteps;
		const size_t c_count = pg->c_count[i];
		pgh->c_count[pe] = c_count;
		for (size_t j=0; j<c_count; ++j, ++(pgh->c_entries)) {
			const size_t ce = pgh->c_entries;
			const size_t three_ce = 3*ce;
			const size_t index = pg->c_start[i] + j;
			const size_t three_index = 3*index;
			pgh->r_coords[three_ce] = pg->r_coords[i][three_index];
			pgh->r_coords[three_ce + 1] = pg->r_coords[i][three_index + 1];
			pgh->r_coords[three_ce + 2] = pg->r_coords[i][three_index + 2];
			pgh->n_coords[ce] = pg->n_coords[i][index];
		}
	}
	return;
}

void write_reset_polygroup1_history(polygroup1_history*const RSTRCT pgh, \
  FILE*const RSTRCT fp) {
	assert(pgh);
	const size_t p_entries = pgh->p_entries;
	size_t c_entry = 0;
	for (size_t i=0; i<p_entries; ++i) {
		const size_t c_count = pgh->c_count[i];
		for (size_t j=0; j<c_count; ++j, ++c_entry) {
			const size_t three_c_entry = 3*c_entry;
			fprintf(fp, "%ld %ld %ld %f %f %f %f\n", \
			  pgh->pg_id[i], \
			  pgh->p_index[i], \
			  pgh->t_steps[i], \
			  pgh->r_coords[three_c_entry], \
			  pgh->r_coords[three_c_entry + 1], \
			  pgh->r_coords[three_c_entry + 2], \
			  pgh->n_coords[c_entry] \
			);
		}
	}
	pgh->p_entries = 0;
	pgh->c_entries = 0;
	return;
}


void print_polygroup1_state(const polygroup1*const RSTRCT pg) {
	assert(pg);
	assert(pg->p_count > 0);
	const size_t p_count = pg->p_count;
	printf("Printing polygroup1 state, p_count = %ld\n", p_count);
	for (size_t i=0; i<p_count; ++i) {
		printf("polymer index: %ld, c_count: %ld, c_start: %ld, c_allocated: %ld\n", \
		  i, pg->c_count[i], pg->c_start[i], pg->c_allocated[i]);
		const size_t limit = pg->c_start[i] + pg->c_count[i];
		for (size_t j=pg->c_start[i]; j<limit; ++j) {
			const size_t three_j = 3*j;
			printf("    %lf  %lf  %lf    %lf\n", pg->r_coords[i][three_j], \
			  pg->r_coords[i][three_j + 1], pg->r_coords[i][three_j + 2], \
			  pg->n_coords[i][j]);
		}
	}
	printf("\n");
	return;
}

polygroup1* create_test_polygroup1(void) {
	polygroup1* pg = create_polygroup1(1, 32, 1234);
	pg->c_count[0] = 2;
	size_t index = 15;
	size_t three_index = 3*index;
	pg->c_start[0] = index;
	pg->r_coords[0][three_index] = 1.0;
	pg->r_coords[0][three_index + 1] = 2.0;
	pg->r_coords[0][three_index + 2] = 3.0;
	pg->n_coords[0][index] = ARC_MAX*0.5;
	++index;
	three_index = 3*index;
	pg->r_coords[0][three_index] = 7.0;
	pg->r_coords[0][three_index + 1] = 7.0;
	pg->r_coords[0][three_index + 2] = 7.0;
	pg->n_coords[0][index] = ARC_MAX*0.75;
	return pg;
}

void run_polygroup1_tests(void) {
	polygroup1_history* pgh = create_polygroup1_history(64, 128);

	polygroup1* pg = create_test_polygroup1();
	printf("\nInitial polygroup state.\n");
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating low constraint.\n");
	coord new_coords[4] = {4.0, 5.0, 1.0, ARC_MAX*0.25};
	create_low_constraint(pg, 0, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating low constraint.\n");
	new_coords[0] = 2.0;
	new_coords[1] = 4.1;
	new_coords[2] = 6.2;
	new_coords[3] = 0.1*ARC_MAX;
	create_low_constraint(pg, 0, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating high constraint.\n");
	new_coords[3] = 0.9*ARC_MAX;
	create_high_constraint(pg, 0, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating high constraint.\n");
	new_coords[0] = 4.0;
	new_coords[1] = 4.1;
	new_coords[2] = 4.2;
	new_coords[3] = 0.95*ARC_MAX;
	create_high_constraint(pg, 0, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating mid constraint at index 1 (in lower half).\n");
	new_coords[0] = 0.1;
	new_coords[1] = 0.2;
	new_coords[2] = 0.3;
	new_coords[3] = 0.2*ARC_MAX;
	create_mid_constraint(pg, 0, 1, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Creating mid constraint at index 4 (in upper half).\n");
	new_coords[0] = 0.1;
	new_coords[1] = 0.2;
	new_coords[2] = 0.3;
	new_coords[3] = 0.6*ARC_MAX;
	create_mid_constraint(pg, 0, 4, new_coords);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Removing high constraint.\n");
	remove_high_constraint(pg, 0);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Removing low constraint.\n");
	remove_low_constraint(pg, 0);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Removing mid constraint at index 2 (in lower half).\n");
	remove_mid_constraint(pg, 0, 2);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Removing mid constraint at index 3 (in upper half).\n");
	remove_mid_constraint(pg, 0, 3);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Moving arc 3.0 across low constraint.\n");
	move_arc_across_low_constraint(pg, 0, 3.0);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Moving arc -5.0 across high constraint.\n");
	move_arc_across_high_constraint(pg, 0, -5.0);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);

	printf("Moving arc 7.0 across constraint 1.\n");
	move_arc_across_mid_constraint(pg, 0, 1, 7.0);
	print_polygroup1_state(pg);
	poll_polygroup1_history(pg, pgh);
	++(pg->tsteps);


	free_polygroup1_members(pg);
	free(pg);
	pg = NULL;

	FILE* fp = fopen("polygroup1_test.txt", "a");
	write_reset_polygroup1_history(pgh, fp);
	fclose(fp);
	fp = NULL;
	free_polygroup1_history_members(pgh);
	free(pgh);
	pgh = NULL;
}


int main() {

	run_polygroup1_tests();

	return 0;
}