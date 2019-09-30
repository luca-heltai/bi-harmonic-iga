FROM dealii/dealii:v9.1.1-gcc-mpi-fulldepsspack-debugrelease

MAINTAINER luca.heltai@gmail.com

# get deal.II repo
USER       root
RUN        spack install googletest ninja \
           && spack clean -a \
	   && cd /usr/local && spack view add . googletest ninja

RUN	   chmod 777 /home/$USER

USER 	   $USER
