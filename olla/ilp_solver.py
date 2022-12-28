import sys
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from olla.gurobi_utils import get_gurobi_env

import gurobipy as gr

logger = logging.getLogger(__name__)

# Wrap is the xpress solver (https://pypi.org/project/xpress/, doc available at
# https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/GUID-616C323F-05D8-3460-B0D7-80F77DA7D046.html)
# to ensure that we can easily try other solvers in the future if needs be.
class ILPSolver:
    # A negative timeout is a hard timeout, in which case the solver returns within the specified amount of time whether or not it found a solution
    # A positive timeout is soft: the solver can exceed the timeout in order to find one feasible solution.
    # Valid solvers are "xpress" and "gurobi"
    def __init__(
        self,
        timeout_s=None,
        rel_stop=None,
        solver="gurobi",
        method=None,
        int_feas_tol=None,
        extra_params=None,
    ):
        self.vars = []
        self.timeout = timeout_s
        self.rel_stop = rel_stop
        self.method = method
        self.int_feas_tol = int_feas_tol
        self.extra_params = extra_params
        self.num_constraints = 0
        if solver == "GUROBI" or solver == "gurobi":
            self.model = gr.Model("gurobi", env=get_gurobi_env())
        else:
            raise Exception("Currently only Gurobi solver is supported.")
        self._message_callback = None

        self.binary = gr.GRB.BINARY
        self.integer = gr.GRB.INTEGER
        self.continuous = gr.GRB.CONTINUOUS
        self.semicontinuous = gr.GRB.SEMICONT
        self.semiinteger = gr.GRB.SEMIINT
        self.minimize = gr.GRB.MINIMIZE
        self.maximize = gr.GRB.MAXIMIZE

        self.mip_optimal = gr.GRB.OPTIMAL
        self.mip_solution = gr.GRB.SUBOPTIMAL
        self.mip_infeas = gr.GRB.INFEASIBLE
        self.mip_unbounded = gr.GRB.UNBOUNDED
        self.mip_no_sol_found = gr.GRB.TIME_LIMIT
        self.mip_lp_not_optimal = gr.GRB.LOADED
        self.lp_optimal = gr.GRB.OPTIMAL
        self.lp_infeas = gr.GRB.INFEASIBLE
        self.lp_unbounded = gr.GRB.UNBOUNDED

        @dataclass
        class GurobiAttributes:
            bestbound = None
            mipstatus = gr.GRB.LOADED
        self.attributes = GurobiAttributes()



    def create_integer_var(self, name, lower_bound=None, upper_bound=None):
        assert name is not None
        if type(name) != str:
            name = str(name)
        lb = -sys.maxsize if lower_bound is None else lower_bound
        ub = sys.maxsize if upper_bound is None else upper_bound
        v = self.model.addVar(
            name=name, lb=lb, ub=ub, vtype=self.integer
        )
        self.vars.append(v)
        return v

    def create_real_var(self, name, lower_bound=None, upper_bound=None):
        assert name is not None
        if type(name) != str:
            name = str(name)
        lb = -float("inf") if lower_bound is None else lower_bound
        ub = float("inf") if upper_bound is None else upper_bound
        v = self.model.addVar(
            name=name, lb=lb, ub=ub, vtype=self.continuous
        )
        self.vars.append(v)
        return v

    def create_binary_var(self, name):
        assert name is not None
        if type(name) != str:
            name = str(name)
        v = self.model.addVar(name=name, vtype=self.binary)
        self.vars.append(v)
        return v

    def set_objective_function(self, equation, maximize=True):
        self.of = equation
        if maximize:
            self.model.setObjective(equation, sense=self.maximize)
        else:
            self.model.setObjective(equation, sense=self.minimize)

    def add_constraint(self, cns, name=""):
        self.num_constraints += 1
        # Make sure names aren't too long for gurobi
        if len(name) >= 250:
            # old_name = name
            num_cns = str(self.num_constraints)
            name = name[: 250 - len(num_cns)] + num_cns
            # print(
            #    f"truncated contraint name from {old_name} to {name}"
            # )
        self.model.addLConstr(cns, name=name)

    def solve(self):
        # Solve the problem. Return the result as a dictionary of values
        # indexed by the corresponding variables or an empty dictionary if the
        # problem is infeasible.
        if self.timeout:
            timeout = float("inf") if self.timeout == 0 else abs(self.timeout)
            self.model.setParam("maxtime", timeout)
        if self.rel_stop:
            self.model.setParam("miprelstop", self.rel_stop)
        if self.method:
            self.model.setParam("Method", self.method)
        if self.int_feas_tol:
            self.model.setParam("IntFeasTol", self.int_feas_tol)
        if self.extra_params:
            for param, value in self.extra_params.items():
                self.model.setParam(param, value)

        self.model.optimize(self._message_callback)
        logger.debug(f"Gurobi problem status {self.model.Status}")
        logger.debug("Extracting results...")
        if self.model.Status == gr.GRB.OPTIMAL:
            self.gurobi_recorded_solution = self._get_var_name_map_value()
        if self.model.IsMIP:
            self.attributes.bestbound = self.model.ObjBound
            self.attributes.mipstatus = self.getProbStatus()

        if self.getProbStatus() == self.mip_infeas:
            self.model.computeIIS()
            violations = ""
            for c in self.model.getConstrs():
                if c.IISConstr:
                    violations += "\n" + str(c)

            raise RuntimeError(
                "Problem is not feasible as the following constraint(s) cannot be satisfied: "
                + violations
            )
        elif (
            # mip_no_sol_found actually means timeout !
            self.getProbStatus() != self.mip_no_sol_found
            and self.getProbStatus() != self.mip_optimal
            and self.getProbStatus() != self.mip_solution
        ):
            raise RuntimeError("Unknown error detected while solving ilp")

        if self.model.SolCount == 0:
            raise RuntimeError("No solution found")

        result = {}
        for v in self.vars:
            result[v] = v.X
        return result

    def solve_relaxation(self):
        self.model.update()
        relaxed_model = self.model.relax()
        relaxed_model.optimize()
        result = {}
        for v in relaxed_model.getVars():
            # Need to index the result by variable name since the relaxed model
            # will be released as soon as we exit the method
            result[v.varName] = v.X
        return result

    def getProbStatus(self) -> Any:
        return self.model.Status

    def _get_var_name_map_value(self) -> Dict[str, Any]:
        name_map_value = {}
        for v in self.model.getVars():
            name_map_value[v.varName] = v.getAttr(gr.GRB.Attr.X)
        return name_map_value

    def write(self, filename: Any, filetype: str = "lp") -> None:

        if filetype:
            filename += "." + filetype
        else:
            if not (
                ".mps" in filename
                or ".rew" in filename
                or ".lp" in filename
                or ".rlp" in filename
            ):
                filename += ".lp"
                logger.debug(f"filename updated {filename}")
        # Added for debugging
        # self.model.computeIIS()
        # self.model.write("model.ilp")

        self.model.write(filename)

    def __str__(self):
        result = "Num Variables = " + str(len(self.vars))
        return result
