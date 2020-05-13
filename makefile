default: wrappers.o greedy_triangle.o turtle.o generate_points.o plotter.o
	gcc --std=c99 -o tri greedy_triangle.o wrappers.o turtle.o -lm
	gcc --std=c99 -o gen generate_points.o wrappers.o turtle.o -lm
	gcc --std=c99 -o plot plotter.o wrappers.o turtle.o -lm
	nvcc par_greedy_triangle.cu -o par
	
generate_points.o: generate_points.c
	gcc -g --std=c99 -Wall -c generate_points.c
	
turtle.o: turtle.c
	gcc -g --std=c99 -Wall -c turtle.c

wrappers.o: wrappers.c
	gcc -g --std=c99 -Wall -c wrappers.c
	
greedy_triangle.o: greedy_triangle.c
	gcc -g  --std=c99 -Wall -c greedy_triangle.c

plotter.o: plotter.c
	gcc -g --std=c99 -Wall -c plotter.c
	
clean:
	rm *.o; rm tri; rm gen; rm plot; rm par;
