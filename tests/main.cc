#include <deal.II/base/mpi.h>

#include <gtest/gtest.h>

#include <fstream>

using namespace dealii;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  std::ofstream logfile;

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      logfile.open("output");
      deallog.attach(logfile);
      deallog << std::setprecision(4);
      deallog.depth_file(10);
    }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
