docker run --user $(id -u):$(id -g) -i -t --rm -P -v `pwd`:/home/dealii/app:rw heltai/dealii:v9.1.1-gcc-mpi /bin/sh -c "cd app; $@"
