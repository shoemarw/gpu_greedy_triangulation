/*
*  par_greedy_triangle.cu
*  Parallel GPU version
*  Author: Randy Shoemaker
*  Some code borrowed from the serial version.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

// Timer marcros written by Professor Mike Lam of JMU
#define START_TIMER(NAME) gettimeofday(&tv, NULL); \
	double NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0);
#define STOP_TIMER(NAME) gettimeofday(&tv, NULL); \
	NAME ## _time = tv.tv_sec+(tv.tv_usec/1000000.0) - (NAME ## _time);
#define GET_TIMER(NAME) (NAME##_time)


  //                                     //
 //             Structs                 //
//	                                   //
/*
 * Structs from the serial version, written by Eliza, Randy, & Alex
*/

/*
 * Represents a point in the plane using cartesian coordinates.
 */
typedef struct {
	double x;
	double y;
} point_t;

/*
 * Represents a line segment between two points.
 */
typedef struct {
	point_t *p;
	point_t *q;
	double len;
} line_t;

  //                                     //
 //          Wrapper Functions          //
//	                                   //
/*
 * Wrappers from the serial version, written by Eliza, Randy, & Alex
*/

/*
 * Prints a point.
 */
void print_point(point_t* point) {
	printf(" (%f , %f) ", point->x, point->y);
}

/*
 * Prints a line.
 */
void print_line(line_t* line) {

	print_point(line->p);
	print_point(line->q);
	printf("\n");
}

/* 
 * Computes the Euclidian distance between two points.
 */
double distance(point_t* p, point_t* q) {
	double delta_x = p->x - q->x;
	double delta_y = p->y - q->y;
	return sqrt(delta_x * delta_x + delta_y * delta_y);
}

/* 
 * Compares two lines. Used for sorting lines with qsort.
 */
int compare(const void* a, const void* b) {
	double result = ((line_t*) a)->len - ((line_t*) b)->len;
	// Make sure that cmp returns an int.
	if (result < 0) {
		return -1;
	}
	else if (result > 0) {
		return 1;
	}
	else {
		return 0;
	}
}

/*
 * Copies 'size' values of 'from' to array 'to'
 */
void copy_array(line_t from[], line_t to[], int size) {
	for (int i = 0; i < size; i++) {
		to[i] = from[i];
	}
}

/*
 * Wrapper function for calloc. It checks for errors as well.
 */
void* allocate(size_t size) {
	void* address = malloc(size);
	if (!address) 	{
		fprintf(stderr, "Cannot malloc, out of memory\n");
		exit(EXIT_FAILURE);
	}
	
	memset(address, 0, size);
	return address;
}

/*
 * Prints a list of lines where each line is stored as an
 * array of four doubles. Author: Randy
 */
void print_lines(double *l, int num_lines) {
	for (int i = 0; i< num_lines; i++) {
		printf("(%lf, %lf) (%lf, %lf)\n", l[4*i],   l[4*i+1],
										  l[4*i+2], l[4*i+3]);
	}
}

  //                         //
 //       DEVICE CODE       //
//                         //
/*
 * Device code written by Randy
*/

__device__ void d_print_lines(double *l, int num_lines) {
	for (int i = 0; i< num_lines; i++) {
		printf("(%lf, %lf) (%lf, %lf)\n", l[4*i],   l[4*i+1],
										  l[4*i+2], l[4*i+3]);
	}
}

// Device function for determining the equality of two points
// p and q. The points are arrays of two ints.
__device__ bool d_is_equal(double *p, double *q) {
	return (p[0] == q[0]) && (p[1] == q[1]);
}

// Device function for determining if the lines I,J share any
// endpoints. The lines are arrays of four ints.
__device__ bool d_share_endpoint(double *I, double *J) {
	// See if any of the end points of I and J are equal.
	// The first two ints in a line are its x-coordinate
	// and the second two ints in a line are its y-coord-
	// -inate. So a line (in this function) is described
	// by four ints.
	return    d_is_equal(&I[0], &J[0]) || d_is_equal(&I[0], &J[2])
		   || d_is_equal(&I[2], &J[0]) || d_is_equal(&I[2], &J[2]);
}

// Helper sign function because nvcc doesnt let me use sign
__device__ int d_sign(double x) { 
	int t = x<0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

// The function for computing orient. Returns 0 if p,q,r are
// colinear, 1 if the traversal from p to q to r is clock-
// -wise, -1 if the traversal from p to q to r is CCW.
__device__ int d_orient(double *p, double *q, double *r) {
	double o = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1]);
	// Use sign(float x) to avoid conditional branching
	return d_sign(o);
}

// Device code for finding the max of two doubles 
__device__ double d_max(double a, double b) {
	return (a > b) ? a : b;
}

// Device code for finding the min of two doubles
__device__ double d_min(double a, double b) {
	return (a < b) ? a : b;
}

// The function for computing lies_on. Returns true iff
// the point q lies on the line segment formed by p and r.
__device__ bool d_lies_on(double *p, double *q, double *r) {
	return (q[0] <= d_max(p[0], r[0]) &&
			q[0] >= d_min(p[0], r[0]) &&
			q[1] <= d_max(p[1], r[1]) &&
			q[1] >= d_min(p[1], r[1])
		   );
}

// Checks if the lines I,J intersect
__device__ bool d_intersects(double *I, double *J) {
	// Get the orientation of I_p, I,q, J_p
	int o1 = d_orient(&I[0], &I[2], &J[0]);
	// Get the orientation of I_p, I_q, J_q
	int o2 = d_orient(&I[0], &I[2], &J[2]);
	// Get the orientation of J_p, J_q, I_p
	int o3 = d_orient(&J[0], &J[2], &I[0]);
	// Get the orientation of J_p, J_q, I_q
	int o4 = d_orient(&J[0], &J[2], &I[2]);
	return    (o1 != o2 && o3 != o4)   
		   || (o1 == 0 && d_lies_on(&I[0], &J[0], &I[2]))
		   || (o2 == 0 && d_lies_on(&I[0], &J[2], &I[2]))
		   || (o3 == 0 && d_lies_on(&J[0], &I[0], &J[2]))
		   || (o4 == 0 && d_lies_on(&J[0], &I[2], &J[2]));
}

__device__ bool d_conflicts(double *I, double *J) {
	return !d_share_endpoint(I, J) && d_intersects(I, J);
}

__global__ void triangulate(double *L, int *num_l, int *smallest_idx, int *indicator) {
	// Incicator is an array whose k^th element is T iff L[k] is non-empty
	// Shoot for maximum memory coalescing:
	// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels
	// -grid-stride-loops/
	// We must subtract 1 from the blockId.x so that block 1 handles what
	// block zero would have if it were processing lines and so block k
	// handles what block k-1 would have if block zero were processing
	// lines. We must also subtract 1 from the gridDim.x to account for
	// the fact that block zero is not processing lines.
	int small = *smallest_idx;
	for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < *num_l; 
		     j += blockDim.x * gridDim.x) {
		// Run intersection test only on non-empty lines
		if (indicator[j] && d_conflicts(&L[4*small], &L[4*j])) {
			// There is an intersection so set line j to be
			// the 'empty line'
			indicator[j] = 0;
		}
	}

}

  //                         //
 //          MAIN           //
//                         //

int main(int argc, char *argv[]) {
	
	// Make sure we get the expected input.
	if (argc != 4) {
		printf("Usage %s <filename>, test<>pts<>, num_tpb, num_blocks\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	int num_tpb = strtol(argv[2], NULL, 10);
	int num_blocks = strtol(argv[3], NULL, 10);

	  //                         //
	 //  Read points from file  //
	//                         // 
	
	// Open the input file for reading. 
	char *fn = argv[1];
	FILE* fin = fopen(fn, "r");
	if (!fin) {
		fprintf(stderr, "ERROR: Could not open %s\n", fn);
		exit(EXIT_FAILURE);
	}

	long num_points;
	fscanf(fin, "%ld\n", &num_points);
	point_t* points = (point_t*) allocate(num_points * sizeof(point_t));
	
	// Read in and store the point s.
	double x, y;   // The Cartesian coordinates of a point.
	long i = 0;    // Index for storing points.

	while (fscanf(fin, "%lf %lf\n", &x, &y) == 2) {
		// Put the values in a point struct and store.
		point_t *p = (point_t*) allocate(sizeof(point_t));
		p->x = x;
		p->y = y;
		
		// Make sure input file didn't make num_points too small.
		if (i >= num_points) {
			fprintf(stderr, "%s", "ERROR: the number of lines exceeds expectation\n");
			exit(EXIT_FAILURE);
		}
		
		points[i] = *p;
		i++;
		free(p);
	}
	fclose(fin);
	
	  //                      //
	 //  Generate all lines  //
	//                      //
	
	// utility struct for timing calls
	struct timeval tv;
	START_TIMER(generate)
	// Make all possible line segments between the points
	// and compute the length of each line.
	int num_lines = ((num_points)*(num_points-1))/2;
	line_t* lines = (line_t*) allocate(num_lines * sizeof(line_t));
	
	long index = 0;
	for (int i = 0; i < num_points; i++) {
		for (int j = i+1; j < num_points; j++) {
			double length = distance(&points[i], &points[j]);
			line_t* l = (line_t*) allocate(sizeof(line_t));
			// set the values of the line and store it.
			l->p =         &points[i];
			l->q =         &points[j];
			l->len =       length;
			lines[index] = *l;
			index++;
			free(l);
		}
	}
	STOP_TIMER(generate)
	
	  //                                      //
	 //  Sort the lines from small to large  //
	//                                      //
	
	START_TIMER(sort)
	qsort(lines, num_lines, sizeof(line_t), compare);
	STOP_TIMER(sort)
		
	  //                                   //
	 //  Greedily build the tringulation  //
	//	                                 //
	
	START_TIMER(triangulate)
	// The triangulation will be stored as an array of 4 doubles. The doubles
	// give the coordinates of the end points. The length is left out so the
	// ammount of data is decreased by 20%.
	int size = sizeof(double);
	double *segments = (double*) allocate(num_lines*size*4);
	int *indicator; // Array whose k^th element is 1 iff the k^th line does not
	                // conflict with any line in the triangulation. In the end
					// the k^th element is 1 if the k^th line belongs to the Tri
	int smallest_idx = 0;
	indicator = (int*) allocate(num_lines*sizeof(int));
	//Transform the lines into arrays of doubles
	for (int i = 0; i < num_lines; i++) {
		point_t p = *(lines[i].p);
		point_t q = *(lines[i].q);
		segments[4*i]   = p.x;
		segments[4*i+1] = p.y;
		segments[4*i+2] = q.x;
		segments[4*i+3] = q.y;
		// Set every element of indicator to be true
		indicator[i] = 1;
	}

	// Device copies
	double *d_lines;
	int *d_num_lines, *d_smallest_idx;
	int *d_indicator;

	// Allocate space on the device
	cudaMalloc((void **)&d_lines, num_lines*4*size);
	cudaMalloc((void **)&d_num_lines, sizeof(int));
	cudaMalloc((void **)&d_smallest_idx, sizeof(int));
	cudaMalloc((void **)&d_indicator, num_lines*sizeof(int));

	// Copy the lines into and number of lines into device memory
	cudaMemcpy(d_lines, segments, num_lines*4*size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_num_lines, &num_lines, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indicator, indicator, num_lines*sizeof(int), cudaMemcpyHostToDevice);

	//SYCHRONIZE THE ITERATIONS OF THE WHILE LOOP VIA KERNEL CALLS
	while (smallest_idx < num_lines) {
		// Copy the index of the new smallest line into device memory.
		cudaMemcpy(d_smallest_idx, &smallest_idx, sizeof(int), cudaMemcpyHostToDevice);

		// Call the device to do a round of eliminations.
		triangulate<<<num_blocks, num_tpb>>>(d_lines, d_num_lines, 
											 d_smallest_idx, d_indicator);

		// Read in the indicator array from the device so we know which lines are
		// non-empty.
		cudaMemcpy(indicator, d_indicator, num_lines*sizeof(int), cudaMemcpyDeviceToHost);


		// Increment the index of the smallest line past all empty lines
		smallest_idx++;
		while (smallest_idx < num_lines && !indicator[smallest_idx]) {
			// Skip over all the 'empty' lines
			smallest_idx++;
		}

	}

	STOP_TIMER(triangulate)
	
	  //                                     //
	 //  Triangulation Done, Display Stats  //
	//	                                   //
	
	// These stats are only for the portions of the code specific to the three
	// phases of building the greedy triangulation. Generate all lines, sort the
	// lines in non-decreasing order, and greedily adding line segments to the
	// triangulation.
	printf("Gent: %.4f  Sort: %.4f  Tria: %.4f\n",
			GET_TIMER(generate), GET_TIMER(sort), GET_TIMER(triangulate));

	// Copy the non-empty triangluation lines over to triang array
	int tlines = 0;
	double* triang   = (double*) allocate(num_lines*size*4);
	for (int i = 0; i < num_lines; i++) {
		if (indicator[i]) {
			triang[4*tlines]     = segments[4*i];
			triang[4*tlines + 1] = segments[4*i + 1];
			triang[4*tlines + 2] = segments[4*i + 2];
			triang[4*tlines + 3] = segments[4*i + 3];
			tlines++; // keep track of number of good lines
		}
	}
	
	
	// Store the triangulation in a file. 
	FILE* write_file = fopen("triangle_result.txt", "w");
	if (!write_file) {
		fprintf(stderr, "ERROR: Could not open %s\n", "triangle_result.txt");
		exit(EXIT_FAILURE);
	}
	
	// The first line of the file specifies the number of lines in the file.
	fprintf(write_file, "%ld\n", tlines);

	for (int i = 0; i < tlines; i++) {
		// Write the non-empty lines to the file
		fprintf(write_file, "(%lf, %lf) (%lf, %lf)\n", triang[4*i], 
		                                           	   triang[4*i+1], 
		                                               triang[4*i+2], 
		                                               triang[4*i+3]);
	}
	
	fclose(write_file);
	
	// Clean up and exit
	free(triang);
	free(points);
	free(lines);
	return (EXIT_SUCCESS);
}

