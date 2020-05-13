/*
*  plotter.c
*  takes a file containing the lines of a triangulation
*  and produces a bmp file containing a plot of the 
*  triangulation.
*  Author: Randy Shoemaker
*/

#include "greedy_triangle.h"

int main(int argc, char *argv[])
{
	// Open the input file for reading. 
	char *fn = "triangle_result.txt";  // We assume the triangulation file's name.
	FILE* fin = fopen(fn, "r");
    if (!fin)
    {
        fprintf(stderr, "ERROR: Could not open %s\n", fn);
        exit(EXIT_FAILURE);
    }

    // The first line of a file must contain a number indicating
	// the number of triangulation lines in the file. Read this 
	// value and use it to allocate storage for the points.
	long num_lines;
	fscanf(fin, "%ld\n", &num_lines);

	turtle_init(1920, 1080); // standard resolution (width, height)
    
    turtle_draw_line(-960, 0, 960, 0); // draws x-axis
    turtle_draw_line(0, -540, 0, 540); // draws y-axis
    
    turtle_set_pen_color(255, 0, 0); // sets the color of the pen to red
                                     // triangles will appear in red axis in black
    
    int scale = 10; // Scales all of the points to fit the window

	// Read in the lines and store them.
	double px, py, qx, qy;  // Cartesian coordinates of endpoints.

	while (fscanf(fin, "(%lf, %lf) (%lf, %lf)\n", &px, &py, &qx, &qy) == 4) {
		// plot the line
		int p_x = (int) px;
        int p_y = (int) py;
        
        int q_x = (int) qx;
        int q_y = (int) qy;
		turtle_draw_line(p_x * scale, p_y * scale , q_x * scale , q_y * scale);		
	}
	fclose(fin);


	turtle_save_bmp("triangulation_graphics.bmp"); // save image
    
    turtle_cleanup(); // clean turtle
	return 0;
}