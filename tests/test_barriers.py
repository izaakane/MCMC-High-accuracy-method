import torch

torch.set_default_dtype(torch.float64)

import unittest

from conspacesampler.barriers import (
    BoxBarrier,
    EllipsoidBarrier,
    SimplexBarrier,
    ComposeBarrier,
    PolytopeBarrier,
)
from conspacesampler.utils import define_ellipsoid


class TestBoxBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, box_barrier: BoxBarrier):
        x = torch.rand(2, box_barrier.dimension)
        x = x * (2 * box_barrier.bounds) - box_barrier.bounds
        self.assertTrue(
            torch.all(box_barrier.feasibility(x)), msg=str(box_barrier.feasibility(x))
        )

        y = torch.tile(torch.Tensor([-1, 1]), (box_barrier.dimension, 1)).T
        y = y * (box_barrier.bounds + 1)  # outside
        self.assertFalse(
            torch.any(box_barrier.feasibility(y)),
            msg=str(box_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            bounds = torch.rand(dim) * 4 + 0.1  # bounds can take values in (0.1, 4.1)
            box_barrier = BoxBarrier(bounds=bounds)
            self._test_feasibility_single(box_barrier)

    def _test_gradient_inverse_gradient_single(self, box_barrier: BoxBarrier):
        for i in range(13):
            x = torch.rand(2, box_barrier.dimension)
            x = x * (2 * box_barrier.bounds) - box_barrier.bounds
            y = torch.randn(2, box_barrier.dimension)

            grad = box_barrier.gradient(x)
            inv_grad = box_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    box_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(box_barrier.inverse_gradient(grad) - x))),
            )
            self.assertTrue(
                torch.allclose(
                    box_barrier.gradient(inv_grad), y, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(box_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            bounds = torch.rand(dim) * 4 + 0.1
            box_barrier = BoxBarrier(bounds=bounds)
            self._test_gradient_inverse_gradient_single(box_barrier)


class TestEllipsoidBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, ellipsoid_barrier: EllipsoidBarrier):
        x = torch.randn(2, ellipsoid_barrier.dimension)
        x = (
            x
            / torch.sqrt(
                ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
            )
            * torch.rand(1)
        )  # to be in the interior
        self.assertTrue(
            torch.all(ellipsoid_barrier.feasibility(x)),
            msg=str(ellipsoid_barrier.feasibility(x)),
        )

        y = torch.randn(2, ellipsoid_barrier.dimension)
        y = (
            y
            / torch.sqrt(
                ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
            )
            * (torch.rand(1) + 1)
        )  # to be in the exterior
        self.assertFalse(
            torch.any(ellipsoid_barrier.feasibility(y)),
            msg=str(ellipsoid_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_feasibility_single(ellipsoid_barrier)

    def _test_ellipsoid_map_inverse_single(self, ellipsoid_barrier: EllipsoidBarrier):
        for i in range(13):
            x = torch.randn(2, ellipsoid_barrier.dimension)
            Ax = ellipsoid_barrier._ellipsoid_map(x)
            invAx = ellipsoid_barrier._inverse_ellipsoid_map(x)

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier._inverse_ellipsoid_map(Ax),
                    x,
                    rtol=1e-04,
                    atol=1e-06,
                ),
                msg=str(
                    torch.max(
                        torch.abs(ellipsoid_barrier._inverse_ellipsoid_map(Ax) - x)
                    )
                ),
            )

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier._ellipsoid_map(invAx), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(ellipsoid_barrier._ellipsoid_map(invAx) - x))
                ),
            )

    def test_ellipsoid_map_inverse(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_ellipsoid_map_inverse_single(ellipsoid_barrier)

    def _test_gradient_inverse_gradient_single(
        self, ellipsoid_barrier: EllipsoidBarrier
    ):
        for i in range(13):
            x = torch.randn(2, ellipsoid_barrier.dimension)
            y = torch.randn(2, ellipsoid_barrier.dimension)
            x = (
                x
                / torch.sqrt(
                    ellipsoid_barrier._ellipsoid_inner_product(x).unsqueeze(dim=-1)
                )
                * torch.rand(1)
            )  # to be in the interior

            grad = ellipsoid_barrier.gradient(x)
            inv_grad = ellipsoid_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(ellipsoid_barrier.inverse_gradient(grad) - x))
                ),
            )
            self.assertTrue(
                torch.allclose(
                    ellipsoid_barrier.gradient(inv_grad),
                    y,
                    atol=1e-06,
                    rtol=1e-04,
                ),
                msg=str(torch.max(torch.abs(ellipsoid_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            ellipsoid = define_ellipsoid(dim, random_seed=97)
            ellipsoid_barrier = EllipsoidBarrier(ellipsoid=ellipsoid)
            self._test_gradient_inverse_gradient_single(ellipsoid_barrier)


class TestSimplexBarrier(unittest.TestCase):
    # Let's test the implementation
    def _test_feasibility_single(self, simplex_barrier: SimplexBarrier):
        x = torch.rand(2, simplex_barrier.dimension)
        x = x / x.sum(dim=-1, keepdims=True) * torch.rand(2, 1)
        self.assertTrue(
            torch.all(simplex_barrier.feasibility(x)),
            msg=str(simplex_barrier.feasibility(x)),
        )

        y = torch.rand(2, simplex_barrier.dimension)
        y = y / y.sum(dim=-1, keepdims=True) * (torch.rand(2, 1) + 1.1)
        self.assertFalse(
            torch.any(simplex_barrier.feasibility(y)),
            msg=str(simplex_barrier.feasibility(y)),
        )

    def test_feasibility(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            simplex_barrier = SimplexBarrier(dimension=dim)
            self._test_feasibility_single(simplex_barrier)

    def _test_gradient_inverse_gradient_single(self, simplex_barrier: SimplexBarrier):
        for i in range(13):
            x = torch.rand(2, simplex_barrier.dimension)
            y = torch.randn(2, simplex_barrier.dimension)
            x = (
                x / x.sum(dim=-1, keepdims=True) * torch.Tensor([[0.5], [0.99]])
            )  # to be in the interior

            # two cases:
            # points close to the boundary
            # points far away from the boundary (well in the interior)

            grad = simplex_barrier.gradient(x)
            inv_grad = simplex_barrier.inverse_gradient(y)

            self.assertTrue(
                torch.allclose(
                    simplex_barrier.inverse_gradient(grad), x, rtol=1e-04, atol=1e-06
                ),
                msg=str(
                    torch.max(torch.abs(simplex_barrier.inverse_gradient(grad) - x))
                ),
            )
            self.assertTrue(
                torch.allclose(
                    simplex_barrier.gradient(inv_grad), y, rtol=1e-04, atol=1e-06
                ),
                msg=str(torch.max(torch.abs(simplex_barrier.gradient(inv_grad) - y))),
            )

    def test_gradient_inverse_gradient(self):
        dimensions = [3, 5, 7, 11]
        for dim in dimensions:
            simplex_barrier = SimplexBarrier(dimension=dim)
            self._test_gradient_inverse_gradient_single(simplex_barrier)


class TestPolytopeBarrier(unittest.TestCase):
    def _test_diag_hess_single(self, dim: int):
        A = torch.empty(dim).bernoulli_() * 2 - 1
        A *= torch.rand(dim) * 3 + 1
        b = torch.rand(dim) * 2 - 1

        full_polytope = PolytopeBarrier({"A": torch.diag_embed(A), "b": b})
        diag_polytope = PolytopeBarrier({"A": A, "b": b})

        x = torch.rand(23, dim) * 2  # feasible points
        pos_A = A > 0
        bounds = b / A
        x[:, pos_A] += bounds[pos_A] - 2
        x[:, ~pos_A] += bounds[~pos_A]

        y = torch.ones(23, dim)  # infeasible points
        y[:, pos_A] = bounds[pos_A] + 1
        y[:, ~pos_A] = bounds[~pos_A] - 1

        for poly in [full_polytope, diag_polytope]:
            name = "Full" if not poly.diag_hess else "Diag"
            self.assertTrue(
                torch.all(poly.feasibility(x)),
                f"{name} polytope feasibility fail",
            )

            self.assertFalse(
                torch.any(poly.feasibility(y)),
                f"{name} polytope infeasibility fail",
            )

        self.assertTrue(
            torch.allclose(full_polytope.value(x), diag_polytope.value(x)),
            "Value mismatch between diag and full versions",
        )

        self.assertTrue(
            torch.allclose(
                full_polytope.hessian(x), torch.diag_embed(diag_polytope.hessian(x))
            ),
            "Hessian mismatch between diag and full versions",
        )

    def test_diag_hess(self):
        for dim in [5, 7, 9, 11]:
            self._test_diag_hess_single(dim=dim)


class TestComposeBarrier(unittest.TestCase):
    def setUp(self):
        self.dim = 43
        self.barriers = [
            BoxBarrier(bounds=torch.ones(self.dim)),
            EllipsoidBarrier(
                ellipsoid={"eigvals": torch.ones(self.dim), "rot": torch.eye(self.dim)}
            ),
        ]

    def test_feasibility(self):
        # note that the ball is inside the box
        x = torch.randn(19, self.dim)
        x = (
            x
            / torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
            * torch.rand(1)
        )
        my_composed_barrier = ComposeBarrier(self.barriers)
        self.assertTrue(
            my_composed_barrier.feasibility(x).all(), "invalid check for feasibility"
        )

        # consider points outside the ball but inside the box
        x = torch.empty(self.dim).bernoulli_() * 2 - 1
        x = x * torch.rand(19, 1) * (self.dim**0.5 - 1) + 1
        self.assertFalse(
            my_composed_barrier.feasibility(x).all(), "invalid check for feasibility"
        )

    def test_hessian(self):
        x = torch.randn(19, self.dim)
        x = (
            x
            / torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
            * torch.rand(1)
        )
        my_composed_barrier = ComposeBarrier(self.barriers)
        my_hessian = my_composed_barrier.hessian(x=x)
        manual_hessian = self.barriers[0].hessian(x=x).diag_embed()
        manual_hessian += self.barriers[1].hessian(x=x)
        self.assertTrue(
            torch.allclose(
                my_hessian,
                manual_hessian,
            ),
            "invalid composed Hessian",
        )


if __name__ == "__main__":
    unittest.main()
