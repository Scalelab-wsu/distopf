from collections.abc import Callable, Collection
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
import distopf as opf
from distopf import (
    LinDistModelL,
    LinDistModelCapMI,
)
from distopf.base import LinDistBase


def gradient_load_min(model: LinDistBase, *args, **kwargs) -> np.ndarray:
    """
    Gradient of the objective function to minimize the load at the substation.
    c has a 1 for each active power flow out of the substation.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ

    Returns
    -------
    c :  1-D array
        The coefficients of the linear objective function to be minimized.
    """
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.idx("pij", model.swing_bus, ph)] = 1
    return c


def gradient_curtail(model: LinDistBase, *args, **kwargs) -> np.ndarray:
    """
    Gradient of the objective function to minimize curtailment of DERs.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ

    Returns
    -------
    c :  1-D array
        The coefficients of the linear objective function to be minimized.

    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    c = np.zeros(model.n_x)
    c[all_pg_idx] = -1
    return c


# ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~
def cp_obj_loss(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    index_list = []
    r_list = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        i = model.x_maps[a].bi
        j = model.x_maps[a].bj
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        index_list = np.append(index_list, model.x_maps[a].pij.to_numpy().flatten())
        index_list = np.append(index_list, model.x_maps[a].qij.to_numpy().flatten())
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    if isinstance(xk, cp.Variable):
        return cp.vdot(r, xk[ix] ** 2)
    else:
        return np.vdot(r, xk[ix] ** 2)


def cp_obj_loss_old(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    f_list = []
    for j in range(1, model.nb):
        for a in "abc":
            if model.phase_exists(a, j):
                i = model.idx("bi", j, a)[0]
                f_list.append(
                    model.r[a + a][i, j] * (xk[model.idx("pij", j, a)[0]] ** 2)
                )
                f_list.append(
                    model.r[a + a][i, j] * (xk[model.idx("qij", j, a)[0]] ** 2)
                )
    return cp.sum(f_list)


def cp_obj_target_p_3ph(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    f = cp.Constant(0)
    target = kwargs["target"]
    if not isinstance(target, Collection):
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    if len(target) != 3:
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")

    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + error_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(
    model: LinDistBase, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    actual = 0
    target = kwargs["target"]
    if not isinstance(target, (int, float)):
        raise TypeError(
            f"target must be a float or integer value. Instead got {target}."
        )
    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + error_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    target = kwargs["target"]
    if not isinstance(target, Collection):
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    if len(target) != 3:
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    error_percent = kwargs.get("error_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target[i] - q * (1 + error_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(
    model: LinDistBase, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    actual = 0
    target = kwargs["target"]
    if not isinstance(target, (int, float)):
        raise TypeError(
            f"target must be a float or integer value. Instead got {target}."
        )
    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + error_percent[0] / 100)) ** 2
    return f


# def cp_obj_curtail(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
#     """
#     Objective function to minimize curtailment of DERs.
#     Min sum((P_der_max - P_der)^2)
#     Parameters
#     ----------
#     model : LinDistModel, or LinDistModelP, or LinDistModelQ
#     xk : cp.Variable
#
#     Returns
#     -------
#     f: cp.Expression
#         Expression to be minimized
#     """
#     f = cp.Constant(0)
#     for i in range(model.ctr_var_start_idx, model.n_x):
#         f += (model.bounds[i][1] - xk[i]) ** 2
#     return f


def cp_obj_curtail(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable

    Returns
    -------
    f: cp.Expression
        Expression to be minimized
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]) ** 2)


def cp_obj_curtail_lp(model: LinDistBase, xk: cp.Variable, **kwargs) -> cp.Expression:
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable

    Returns
    -------
    f: cp.Expression
        Expression to be minimized
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]))


def cp_obj_none(*args, **kwargs) -> cp.Constant:
    """
    For use with cvxpy_solve() to run a power flow with no optimization.

    Returns
    -------
    constant 0
    """
    return cp.Constant(0)


def cvxpy_solve(
    model: LinDistBase,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a convex optimization problem using cvxpy.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    obj_func : handle to the objective function
    kwargs :

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", cp.CLARABEL)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [m.a_eq @ x - m.b_eq.flatten() == 0]
    # lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    # ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    lb = [x >= m.x_min]
    ub = [x <= m.x_max]
    g_inequality = []
    if m.a_ub is not None and m.b_ub is not None:
        if m.a_ub.shape[0] != 0 and m.a_ub.shape[1] != 0:
            g_inequality = [m.a_ub @ x - m.b_ub <= 0]
    expression = obj_func(m, x, **kwargs)
    prob = cp.Problem(cp.Minimize(expression), g + g_inequality + ub + lb)
    prob.solve(verbose=False, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def cvxpy_mi_solve(
    model: LinDistModelCapMI,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a convex optimization problem using cvxpy.
    Parameters
    ----------
    model : LinDistModelCapMI
    obj_func : handle to the objective function
    kwargs :

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    n_u = len(m.cap_buses["a"]) + len(m.cap_buses["b"]) + len(m.cap_buses["c"])
    u_c = cp.Variable(shape=(n_u,), name="u_c", value=np.ones(n_u), boolean=True)
    u_idxs = np.r_[m.uc_map["a"], m.uc_map["b"], m.uc_map["c"]]
    gu = [x[u_idxs] == u_c]
    g_ineq = [csr_array(m.a_ub) @ x - m.b_ub.flatten() <= 0]
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    expression = obj_func(m, x, **kwargs)
    prob = cp.Problem(cp.Minimize(expression), g + ub + lb + gu + g_ineq)
    prob.solve(verbose=True, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def pf(model) -> OptimizeResult:
    c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(c, A_eq=csr_array(model.a_eq), b_eq=model.b_eq.flatten())
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res


def lp_solve(
    model: LinDistBase,
    c: np.ndarray | Callable = None,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a linear program using scipy.optimize.linprog and having the objective function:
        Min c^T x
    Parameters
    ----------
    model : LinDistBase
    c :  1-D array
        The coefficients of the linear objective function to be minimized.
    Returns
    -------
    result : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields
        below. Note that the return types of the fields may depend on whether
        the optimization was successful, therefore it is recommended to check
        `OptimizeResult.status` before relying on the other fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.
    """
    if isinstance(c, Callable):
        c = c(model)
    if c is None:
        c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(
        c,
        A_eq=csr_array(model.a_eq),
        b_eq=model.b_eq.flatten(),
        A_ub=model.a_ub,
        b_ub=model.b_ub,
        bounds=model.bounds,
    )
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res


def pyomo_solve(
    model: LinDistBase,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    import pyomo.environ as pe

    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", "ipopt")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()

    cm = pe.ConcreteModel()
    cm.n_xk = pe.RangeSet(0, model.n_x - 1)
    cm.xk = pe.Var(cm.n_xk)
    cm.constraints = pe.ConstraintList()
    for i in range(model.n_x):
        cm.constraints.add(cm.xk[i] <= model.x_max[i])
        cm.constraints.add(cm.xk[i] >= model.x_min[i])

    def equality_rule(_cm, i):
        if model.a_eq[[i], :].nnz > 0:
            return model.b_eq[i] == sum(
                _cm.xk[j] * model.a_eq[i, j]
                for j in range(model.n_x)
                if model.a_eq[i, j]
            )
        return pe.Constraint.Skip

    def inequality_rule(_cm, i):
        if model.a_ub[[i], :].nnz > 0:
            return model.b_ub[i] >= sum(
                _cm.xk[j] * model.a_ub[i, j]
                for j in range(model.n_x)
                if model.a_ub[i, j]
            )
        return pe.Constraint.Skip

    cm.equality = pe.Constraint(cm.n_xk, rule=equality_rule)
    if model.a_ub.shape[0] != 0:
        cm.ineq_set = pe.RangeSet(0, model.a_ub.shape[0] - 1)
        cm.inequality = pe.Constraint(cm.ineq_set, rule=inequality_rule)
    cm.objective = pe.Objective(expr=obj_func)
    pe.SolverFactory(solver).solve(cm)

    x_dict = cm.xk.extract_values()
    x_res = np.zeros(len(x_dict))
    for key, value in x_dict.items():
        x_res[key] = value

    result = OptimizeResult(
        fun=float(pe.value(cm.objective)),
        # success=(prob.status == "optimal"),
        # message=prob.status,
        x=x_res,
        # nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result
