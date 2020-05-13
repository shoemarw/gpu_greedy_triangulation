# gpu_greedy_triangulation

run make to generate all executables.

run ./gen <number of points> <seed> to generate an input file for our triangulaiton algorithms. The file will contain the specified number of points and use a pseudorandom number generator with the specified seed to generate the coordinates.

run ./tri <filename of file made by gen> to triangulate a point set with the serial algorithm.

run ./par <filename of file made by gen> <number of blocks> <number of threads per block> to triangulate a point set with our GPU implementation. Uses the specified number of blocks and threads per block.
