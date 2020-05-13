echo "Starting experiments..."
echo "Triangulation, Points, Blocks, Threads" >> log.csv
for x in 4096 5792 8192; do
    for b in 256 512 1024; do
        for t in 128 256 512 1024; do
            "./par" "test"$x"pts93" $b $t >> log.csv
            echo ", $x, $b, $t" >> log.csv
        done
    done
done
echo "Finished"
