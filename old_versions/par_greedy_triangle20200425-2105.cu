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
typedef struct
{
	double x;
	double y;
} point_t;

/*
 * Represents a line segment between two points.
 */
typedef struct
{
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
void print_point(point_t* point)
{
	printf(" (%f , %f) ", point->x, point->y);
}

/*
 * Prints a line.
 */
void print_line(line_t* line)
{

	print_point(line->p);
	print_point(line->q);
	printf("\n");
}

/* 
 * Computes the Euclidian distance between two points.
 */
double distance(point_t* p, point_t* q)
{
	double delta_x = p->x - q->x;
	double delta_y = p->y - q->y;
	return sqrt(delta_x * delta_x + delta_y * delta_y);
}

/* 
 * Compares two lines. Used for sorting lines with qsort.
 */
int compare(const void* a, const void* b)
{
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
void copy_array(line_t from[], line_t to[], int size)
{
	for (int i = 0; i < size; i++)
	{
		to[i] = from[i];
	}
}

/*
 * Wrapper function for calloc. It checks for errors as well.
 */
void* allocate(size_t size)
{
	void* address = malloc(size);
	if (!address)
	{
		fprintf(stderr, "Cannot malloc, out of memory\n");
		exit(EXIT_FAILURE);
	}
	
	memset(address, 0, size);
	return address;
}


  //                         //
 //       DEVICE CODE       //
//                         //
/*
 * Device code written by Randy
*/

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

// The function for computing lies_on. Returns true iff
// the point q lies on the line segment formed by p and r.
__device__ bool d_lies_on(double *p, double *q, double *r) {
	return (q[0] <= max(p[0], r[0]) &&
			q[0] >= min(p[0], r[0]) &&
			q[1] <= max(p[1], r[1]) &&
			q[1] >= min(p[1], r[1])
		   );
}

// Checks if the lines I,J intersect
__device__ int d_intersects(double *I, double *J) {
	// Get the orientation of I_p, I,q, J_p
	int o1 = d_orient(&I[0], &I[2], &J[0]);
	// Get the orientation of I_p, I_q, J_q
	int o2 = d_orient(&I[0], &I[2], &J[2]);
	// Get the orientation of J_p, J_q, I_p
	int o3 = d_orient(&J[0], &J[2], &I[0]);
	// Get the orientation of J_p, J_q, I_q
	int o4 = d_orient(&J[0], &J[2], &I[1]);

	// See if no subset of 3 points is colinear and
	// an intersection has occured.
	if (o1 != o2 && o3 != o4) {
		return 1;
	}
	// Three of the points are colinear.
	else {
		return    (o1 == 0 && d_lies_on(&I[0], &J[0], &I[2]))
			   || (o2 == 0 && d_lies_on(&I[0], &J[2], &I[2]))
			   || (o3 == 0 && d_lies_on(&J[0], &I[0], &J[2]))
			   || (o4 == 0 && d_lies_on(&J[0], &I[2], &J[2]));
	}
}

// Only used for testing....
__global__ void d_conflicts(double *I, double *J, int *result) {
	*result = d_share_endpoint(I, J) || !d_intersects(I, J);
}

// Need to make this function visible to cuda_greedy_triangle.c at compile time.
int d_conflicts_wrapper(double *I, double *J) {
	int size = sizeof(double);
	double *d_I, *d_J;
	int *d_result; // device copies

	// Allocate space on the device
	cudaMalloc((void **)&d_I, size*4);
	cudaMalloc((void **)&d_J, size*4);
	cudaMalloc((void **)&d_result, sizeof(int));

	// Copy the lines into device memory
	cudaMemcpy(d_I, I, size*4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_J, J, size*4, cudaMemcpyHostToDevice);

	// Run the conflict test
	d_conflicts<<<1,1>>>(d_I, d_J, d_result);
	cudaDeviceSynchronize();


	// Get the results back to the host
	int result = 0;
	cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_I); cudaFree(d_J); cudaFree(d_result);
	return result;
}

  //                         //
 //          MAIN           //
//                         //

int main(int argc, char *argv[]) {
	
	// Make sure we get the expected input.
	if (argc != 2) {
		printf("Usage %s <filename>, argv[0] \n", argv[0]);
		exit(EXIT_FAILURE);
	}

	  //                         //
	 //  Read points from file  //
	//                         // 
	
	// Open the input file for reading. 
	char *fn = argv[1];
	FILE* fin = fopen(fn, "r");
	if (!fin)
	{
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
		if (i >= num_points) 
		{
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
	// The triangulation will be stored as an array of lines.
	// Allocate space for the triangulation.
	line_t* triang = (line_t*) allocate(num_lines*sizeof(line_t));
	
	// unknown keeps track of how many lines remain that may or may not be in the 
	// triangulation. 
	long unknown = num_lines;
	long tlines  = 0;        // Tracks the number of lines in the triangulation.
	// Keep going until there are no more lines of unknown status.
	while (unknown > 0) {
		// Add the smallest element of lines array to the triangulation.
		triang[tlines] = lines[0]; // lines[0] is always the smallest line.
		tlines++;  // increment the count of lines in triang.
		unknown--; // decrement unknown since a line has been added to triang.
		line_t* temp = (line_t*) allocate(num_lines * sizeof(line_t));

		int end = unknown;
		int temp_size = 0;
		
		for (int j = 1; j < end + 1; j++)
		{
			// Run intersection test. 
			double I[4], J[4];
			point_t Ip = *(lines[0].p);
			point_t Iq = *(lines[0].q);
			point_t Jp = *(lines[j].p);
			point_t Jq = *(lines[j].q);

			// Build the smaller line
			I[0] = Ip.x;
			I[1] = Ip.y;
			I[2] = Iq.x;
			I[3] = Iq.y;
			// Build the jth line
			J[0] = Jp.x;
			J[1] = Jp.y;
			J[2] = Jq.x;
			J[3] = Jq.y;

			int result = d_conflicts_wrapper(I, J);

			if (result) 
			{
				temp[temp_size] = lines[j];
				temp_size++;
			}
			else
			{
				unknown--;
			}
		}
				
		copy_array(temp, lines, temp_size);
		free(temp);
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
	
	// Store the triangulation in a file. 
	FILE* write_file = fopen("triangle_result.txt", "w");
	if (!write_file)
	{
		fprintf(stderr, "ERROR: Could not open %s\n", "triangle_result.txt");
		exit(EXIT_FAILURE);
	}
	
	// The first line of the file specifies the number of lines in the file.
	fprintf(write_file, "%ld\n", tlines);

	printf("\n\n\nTHE TRIANGULATION:\n");

	for (int i = 0; i < tlines; i++)
	{
		print_line(&triang[i]);
		point_t p = *(triang[i].p);
		point_t q = *(triang[i].q);
		fprintf(write_file, "(%lf, %lf) (%lf, %lf)\n", p.x, p.y, q.x, q.y);
	}
	
	fclose(write_file);
	
	// Clean up and exit
	free(triang);
	free(points);
	free(lines);
	return (EXIT_SUCCESS);
}

