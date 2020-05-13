default: wrappers.o greedy_triangle_find_An.o generate_points.o
	gcc --std=c99 -o gen generate_points.o wrappers.o -lm
	gcc --std=c99 -o tri greedy_triangle_find_An.o wrappers.o -lm

generate_points.o: generate_points.c
	gcc -g --std=c99 -Wall -c generate_points.c

wrappers.o: wrappers.c
	gcc -g --std=c99 -Wall -c wrappers.c
	
greedy_triangle_find_An.o: greedy_triangle_find_An.c
	gcc -g --std=c99 -Wall -c greedy_triangle_find_An.c
	
clean:
	rm *.o; rm tri; rm gen;
