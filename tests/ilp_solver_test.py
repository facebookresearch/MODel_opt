import unittest

from olla import ilp_solver


class ILPSolverTest(unittest.TestCase):
    def setUp(self):
        pass

    def testSimpleProblem(self):
        solver = ilp_solver.ILPSolver()
        x1 = solver.create_integer_var("x1", -10, 10)
        x2 = solver.create_real_var("x2")
        solver.add_constraint(x1 + 3 * x2 >= 4)
        solver.add_constraint(10 * x1 + 3 * x2 <= 40)

        # Minimize the objective
        solver.set_objective_function(x1**2 + 2 * x2, maximize=False)
        s = solver.solve()
        self.assertAlmostEqual(s[x1], 0.0)
        self.assertAlmostEqual(s[x2], 1.3333333333)

        # Maximize the objective
        solver.set_objective_function(x1**2 + 2 * x2, maximize=True)
        s = solver.solve()
        self.assertAlmostEqual(s[x1], -10.0)
        self.assertAlmostEqual(s[x2], 46.66666666666667)

    def testBounds(self):
        solver = ilp_solver.ILPSolver()
        x1 = solver.create_integer_var("x1", 7, 7)
        x2 = solver.create_integer_var("x2")
        solver.add_constraint(x2 <= 13)
        solver.set_objective_function(x1 + x2)
        s = solver.solve()
        self.assertAlmostEqual(s[x1], 7.0)
        self.assertAlmostEqual(s[x2], 13.0)

    def testWrite(self):
        solver = ilp_solver.ILPSolver()
        x1 = solver.create_integer_var("x1", -10, 10)
        x2 = solver.create_real_var("x2")
        solver.add_constraint(x1 + 3 * x2 >= 4)
        solver.add_constraint(10 * x1 + 3 * x2 <= 40)
        solver.set_objective_function(x1**2 + 2 * x2, maximize=False)
        solver.write("simple_pb")

    def testGurobi(self):
        solver = ilp_solver.ILPSolver(solver="GUROBI")
        x1 = solver.create_integer_var("x1", 7, 7)
        x2 = solver.create_integer_var("x2")
        solver.add_constraint(x2 <= 13)
        solver.set_objective_function(x1 + x2)
        s = solver.solve()
        self.assertAlmostEqual(s[x1], 7.0)
        self.assertAlmostEqual(s[x2], 13.0)
