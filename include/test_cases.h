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
#ifndef test_cases_h
#define test_cases_h

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

using namespace dealii;
using dealii::numbers::PI;

namespace simple
{
  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution()
      : Function<dim>()
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const
    {
      (void)component;
      return sin(PI * p[0]) * sin(PI * p[1]);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component = 0) const
    {
      (void)component;
      Tensor<1, dim> r;
      r[0] = PI * cos(PI * p[0]) * sin(PI * p[1]);
      r[1] = PI * cos(PI * p[1]) * sin(PI * p[0]);
      return r;
    }

    virtual void
    hessian_list(const std::vector<Point<dim>> &       points,
                 std::vector<SymmetricTensor<2, dim>> &hessians,
                 const unsigned int                    component = 0) const
    {
      (void)component;
      for (unsigned i = 0; i < points.size(); ++i)
        {
          const double x = points[i][0];
          const double y = points[i][1];

          hessians[i][0][0] = -PI * PI * sin(PI * x) * sin(PI * y);
          hessians[i][0][1] = PI * PI * cos(PI * x) * cos(PI * y);
          hessians[i][1][1] = -PI * PI * sin(PI * x) * sin(PI * y);
        }
    }
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const

    {
      (void)component;
      return 4 * std::pow(PI, 4.0) * sin(PI * p[0]) * sin(PI * p[1]);
    }
  };
} // namespace simple

namespace sin4
{
  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution()
      : Function<dim>()
    {}
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component = 0) const;
    virtual void
    hessian_list(const std::vector<Point<dim>> &       points,
                 std::vector<SymmetricTensor<2, dim>> &hessians,
                 const unsigned int                    component = 0) const;
  };

  template <int dim>
  double
  Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return std::sin(PI * p(0)) * std::sin(PI * p(0)) * std::sin(PI * p(1)) *
           std::sin(PI * p(1));
  }

  template <int dim>
  Tensor<1, dim>
  Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
  {
    Tensor<1, dim> return_value;

    return_value[0] = 2 * PI * std::cos(PI * p(0)) * std::sin(PI * p(0)) *
                      std::sin(PI * p(1)) * std::sin(PI * p(1));
    return_value[1] = 2 * PI * std::cos(PI * p(1)) * std::sin(PI * p(0)) *
                      std::sin(PI * p(0)) * std::sin(PI * p(1));
    return return_value;
  }

  template <int dim>
  void
  Solution<dim>::hessian_list(const std::vector<Point<dim>> &       points,
                              std::vector<SymmetricTensor<2, dim>> &hessians,
                              const unsigned int) const
  {
    Tensor<1, dim> p;
    for (unsigned i = 0; i < points.size(); ++i)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            p[d] = points[i][d];
          } // d-loop
        for (unsigned int d = 0; d < dim; ++d)
          {
            hessians[i][d][d] =
              2 * PI * PI * std::cos(PI * p[d]) * std::cos(PI * p[d]) *
                std::sin(PI * p[(d + 1) % dim]) *
                std::sin(PI * p[(d + 1) % dim]) -
              2 * PI * PI * std::sin(PI * p[d]) * std::sin(PI * p[d]) *
                std::sin(PI * p[(d + 1) % dim]) *
                std::sin(PI * p[(d + 1) % dim]);
            hessians[i][d][(d + 1) % dim] = 4 * PI * PI * std::cos(PI * p[d]) *
                                            std::cos(PI * p[(d + 1) % dim]) *
                                            std::sin(PI * p[d]) *
                                            std::sin(PI * p[(d + 1) % dim]);
            hessians[i][(d + 1) % dim][d] = hessians[i][d][(d + 1) % dim];
          }
      }
  }

  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;
  };

  template <int dim>
  double
  RightHandSide<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
  {
    Assert(component == 0, ExcNotImplemented());
    (void)component;
    return (8 * std::pow(PI, 4) *
            (8 * std::sin(PI * p(0)) * std::sin(PI * p(0)) *
               std::sin(PI * p(1)) * std::sin(PI * p(1)) -
             3 * std::sin(PI * p(0)) * std::sin(PI * p(0)) -
             3 * std::sin(PI * p(1)) * std::sin(PI * p(1)) + 1));
  }
} // namespace sin4


using namespace simple;
// using namespace sin4;

#endif