# gpu_greedy_triangulation

GNU General Public License v3.0

run make to generate all executables.

run ./gen -(number of points)- -(seed)- to generate an input file for our triangulaiton algorithms. The file will contain the specified number of points and use a pseudorandom number generator with the specified seed to generate the coordinates.

run ./tri -(filename of file made by gen)- to triangulate a point set with the serial algorithm. This will produce a file containing the triangulation and another bmp file containing an image of the triangulation.

run ./par -(filename of file made by gen)- -(number of blocks)- -(number of threads per block)- to triangulate a point set with our GPU implementation. Uses the specified number of blocks and threads per block. This will produce a file containing the triangulation. To produce an image of this triangulation run ./plot immediately after running ./par.

use bash to run experiments.sh to reproduce our results. This will create a file log.csv

Please contact the author with any issues.
