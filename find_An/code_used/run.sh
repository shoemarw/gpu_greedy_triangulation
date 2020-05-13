#!bin/bash
echo "Starting"
for s in 37 41 61 93 97; do
	for x in 4 16 64 256 512 1024; do
		"./gen" $x $s
		echo "== $x points, seed is $s"
		"./tri" "test"$x"pts"$s
	done
done
