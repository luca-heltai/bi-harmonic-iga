/* ---------------------------------------------------------------------
 * Copyright (C) 2011 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/differentiation/sd.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <list>

#include "grid_generator.h"
#include "iga_handler.h"
#include "test_cases.h"

using namespace dealii;
using dealii::numbers::PI;


template <int dim>
class BiLaplacian
{
public:
  BiLaplacian(const std::vector<std::vector<double>> &      knots,
              const std::vector<std::vector<unsigned int>> &mults,
              const unsigned int                            degree,
              ParsedConvergenceTable &                      convergence_table);

  void
  run(unsigned int cycle);

  void
  add_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solver");
    prm.add_parameter("Max iterations", max_iter);
    prm.add_parameter("Tolerance", tolerance);
    prm.add_parameter("Reduction", reduction);
    prm.leave_subsection();
  }

private:
  void
  make_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results(const unsigned int iteration);
  void
  compute_error(const unsigned int cycle);

  IgaHandler<dim, dim> iga_handler;

  unsigned int degree;

  unsigned int max_iter  = 10000;
  double       tolerance = 1e-12;
  double       reduction = 1e-12;

  Triangulation<dim> &triangulation;
  FE_Bernstein<dim> & fe;
  DoFHandler<dim> &   dof_handler;

  MappingFEField<dim> &mappingfe;

  AffineConstraints<double> bspline_constraints;

  SparseMatrix<double> bspline_system_matrix;

  Vector<double> bspline_solution;
  Vector<double> bspline_system_rhs;

  SparsityPattern sparsity_bspline;

  TrilinosWrappers::PreconditionAMG precondition;

  ParsedConvergenceTable &convergence_table;
};



template <int dim>
BiLaplacian<dim>::BiLaplacian(
  const std::vector<std::vector<double>> &      knots,
  const std::vector<std::vector<unsigned int>> &mults,
  const unsigned int                            degree,
  ParsedConvergenceTable &                      convergence_table)
  : iga_handler(knots, mults, degree)
  , degree(degree)
  , triangulation(iga_handler.tria)
  , fe(iga_handler.fe)
  , dof_handler(iga_handler.dh)
  , mappingfe(*iga_handler.map_fe)
  , convergence_table(convergence_table)
{}

template <int dim>
void
BiLaplacian<dim>::make_grid()
{
  std::cout << std::endl
            << "Degree: " << degree << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
BiLaplacian<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl
            << "Number of degrees of freedom IGA: " << iga_handler.n_bspline
            << std::endl
            << std::endl;

  DynamicSparsityPattern bspline_sp(iga_handler.n_bspline);
  iga_handler.make_sparsity_pattern(bspline_sp);

  sparsity_bspline.copy_from(bspline_sp);
  bspline_system_matrix.reinit(sparsity_bspline);

  bspline_solution.reinit(iga_handler.n_bspline);
  bspline_system_rhs.reinit(iga_handler.n_bspline);

  // Boundary values
  QGauss<dim - 1> boundary_quad(fe.degree + 2);

  std::map<types::global_dof_index, double> boundary_values;

  // FunctionParser<dim> boundary_funct(boundary_values_expression);
  Solution<dim> boundary_funct;

  std::map<types::boundary_id, const Function<dim> *> functions;
  functions[0] = &boundary_funct;

  iga_handler.project_boundary_values(functions,
                                      boundary_quad,
                                      bspline_constraints);

  std::vector<unsigned int> dofs(dim);
  for (unsigned int i = 0; i < dim; ++i)
    {
      dofs[i] =
        n_dofs({iga_handler.knot_vectors[i]}, {iga_handler.mults[i]}, degree);
    }

  for (unsigned int i = 0; i < iga_handler.n_bspline; ++i)
    {
      // Second and second to last dof along x
      if (i % dofs[0] == 1 || i % dofs[0] == (dofs[0] - 2))
        bspline_constraints.add_line(i);

      // Second and second to last dof along y
      if ((i / dofs[0]) % dofs[1] == 1 ||
          (i / dofs[0]) % dofs[1] == (dofs[1] - 2))
        bspline_constraints.add_line(i);
    }
  bspline_constraints.close();
}



template <int dim>
void
BiLaplacian<dim>::assemble_system()
{
  std::cout << "   Assembling system..." << std::endl;

  bspline_system_matrix = 0;
  bspline_system_rhs    = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  // FunctionParser<dim> right_hand_side(forcing_term_expression);
  RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values( // mappingfe,
    fe,
    quadrature_formula,
    update_values | update_hessians | update_quadrature_points |
      update_JxW_values);

  FEValuesExtractors::Scalar scalar(0);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const auto delta_phi_i =
              trace(fe_values[scalar].hessian(i, q_point));
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const auto delta_phi_j =
                  trace(fe_values[scalar].hessian(j, q_point));
                cell_matrix(i, j) +=
                  (delta_phi_j * delta_phi_i * fe_values.JxW(q_point));
              }

            cell_rhs(i) +=
              (fe_values.shape_value(i, q_point) *
               right_hand_side.value(fe_values.quadrature_point(q_point)) *
               fe_values.JxW(q_point));
          }

      iga_handler.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             cell,
                                             bspline_system_matrix,
                                             bspline_system_rhs,
                                             bspline_constraints);
    }
}


template <int dim>
void
BiLaplacian<dim>::solve()
{
  std::cout << "   Solving system..." << std::endl;

  ReductionControl         reduction_control(max_iter, tolerance, reduction);
  SolverCG<Vector<double>> solver(reduction_control);

  precondition.initialize(bspline_system_matrix);

  solver.solve(bspline_system_matrix,
               bspline_solution,
               bspline_system_rhs,
               precondition);
  bspline_constraints.distribute(bspline_solution);

  std::cout << "      Error: " << reduction_control.initial_value() << " -> "
            << reduction_control.last_value() << " in "
            << reduction_control.last_step() << " CG iterations." << std::endl;
}



template <int dim>
void
BiLaplacian<dim>::output_results(const unsigned int iteration)
{
  std::cout << "   Writing graphical output..." << std::endl;

  DataOut<dim> data_out;

  data_out.set_flags(
    DataOutBase::VtkFlags(std::numeric_limits<double>::min(),
                          iteration,
                          true,
                          DataOutBase::VtkFlags::best_compression,
                          true));


  data_out.attach_dof_handler(dof_handler);

  Vector<double> bspline_sol_dh(dof_handler.n_dofs());

  Vector<double> exact_dh(dof_handler.n_dofs());

  iga_handler.transform_vector_into_fe_space(bspline_sol_dh, bspline_solution);
  AffineConstraints<double> empty_constraints;
  empty_constraints.close();
  VectorTools::project(dof_handler,
                       empty_constraints,
                       QGauss<dim>(2 * degree + 1),
                       Solution<dim>(),
                       exact_dh);

  data_out.add_data_vector(bspline_sol_dh, "u");
  data_out.add_data_vector(exact_dh, "u_exact");

  data_out.build_patches(degree);

  std::ofstream output_vtu(
    (std::string("output_") + Utilities::int_to_string(iteration, 3) + ".vtu")
      .c_str());
  data_out.write_vtu(output_vtu);
}


template <int dim>
void
BiLaplacian<dim>::compute_error(const unsigned int)
{
  std::cout << "   Computing error..." << std::endl;

  Vector<double> bspline_sol_dh(dof_handler.n_dofs());
  iga_handler.transform_vector_into_fe_space(bspline_sol_dh, bspline_solution);

  // FunctionParser<dim> exact_solution(exact_solution_expression);
  Solution<dim> exact_solution;

  auto compute_H2_error = [&]() {
    const QGauss<dim> quadrature_formula(2 * fe.degree + 1);
    Vector<double>    error_per_cell(triangulation.n_active_cells());

    FEValues<dim> fe_values( // mappingfe,
      fe,
      quadrature_formula,
      update_values | update_hessians | update_quadrature_points |
        update_JxW_values);

    FEValuesExtractors::Scalar scalar(0);
    const unsigned int         n_q_points = quadrature_formula.size();

    std::vector<SymmetricTensor<2, dim>> exact_hessians(n_q_points);
    std::vector<Tensor<2, dim>>          hessians(n_q_points);
    unsigned int                         id = 0;
    for (auto cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values[scalar].get_function_hessians(bspline_sol_dh, hessians);
        exact_solution.hessian_list(fe_values.get_quadrature_points(),
                                    exact_hessians);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            error_per_cell[id] +=
              ((exact_hessians[q_point] - hessians[q_point]).norm() *
               fe_values.JxW(q_point));
          }
        ++id;
      }
    return std::sqrt(error_per_cell.l2_norm());
  };

  convergence_table.add_extra_column("u_H2_seminorm", compute_H2_error);

  convergence_table.error_from_exact(dof_handler,
                                     bspline_sol_dh,
                                     exact_solution);
}


template <int dim>
void
BiLaplacian<dim>::run(unsigned int cycle)
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results(cycle);
  compute_error(cycle);
}


int
main(int argc, char *argv[])
{
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  try
    {
      deallog.depth_console(0);

      // Create a plate
      std::vector<unsigned int> subdivisions(1);
      unsigned int              degree          = 2;
      std::string               refinement_type = "h";
      unsigned int              n_cycle         = 7;

      ParameterHandler prm;

      ParsedConvergenceTable convergence_table(
        {"u"}, {{VectorTools::L2_norm, VectorTools::H1_norm}});

      prm.enter_subsection("Global parameters");
      prm.add_parameter("Number of cycles", n_cycle);
      prm.add_parameter("Degree", degree, "", Patterns::Integer(2));
      prm.leave_subsection();

      prm.enter_subsection("Error");
      convergence_table.add_parameters(prm);
      prm.leave_subsection();

      for (unsigned int cycle = 1; cycle < n_cycle; ++cycle)
        {
          subdivisions[0] = Utilities::pow(2, cycle);

          std::vector<std::vector<double>> knots(
            1, std::vector<double>(subdivisions[0] + 1));

          for (unsigned int i = 0; i < subdivisions[0] + 1; ++i)
            knots[0][i] = 0.0 + i * 1.0 / subdivisions[0];

          std::vector<std::vector<unsigned int>> mults(
            1, std::vector<unsigned int>(subdivisions[0] + 1, 1));

          // maximum continuity
          {
            for (unsigned int i = 0; i < subdivisions[0]; ++i)
              mults[0][i] = 1;
          }

          // open knot vectors
          mults[0][0]               = degree + 1;
          mults[0][subdivisions[0]] = degree + 1;

          mults.push_back(mults[0]);
          knots.push_back(knots[0]);

          BiLaplacian<2> bilaplacian(knots, mults, degree, convergence_table);
          bilaplacian.add_parameters(prm);

          ParameterAcceptor::initialize("parameters.prm",
                                        "used_parameters.prm",
                                        ParameterHandler::ShortText,
                                        prm);

          bilaplacian.run(cycle);
        }
      convergence_table.output_table(std::cout);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
