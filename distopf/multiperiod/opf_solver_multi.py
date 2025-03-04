from collections.abc import Callable, Collection
from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
from distopf.multiperiod.lindist_base_modular_multi import LinDistModelMulti
from distopf.multiperiod.lindist_multi_fast import LinDistModelMultiFast


# cost = pd.read_csv("cost_data.csv")
def gradient_load_min(model):
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.branches_out_of_j("pij", 0, ph)] = 1
    return c


def gradient_curtail(model):
    c = np.zeros(model.n_x)
    for i in range(
        model.p_der_start_phase_idx["a"],
        model.p_der_start_phase_idx["c"] + len(model.gen_buses["c"]),
    ):
        c[i] = -1
    return c


def gradient_battery_efficiency(
    model: LinDistModelMulti, xk: cp.Variable, **kwargs
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    c = np.zeros(model.n_x)
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            charging_efficiency = model.bat.loc[
                model.charge_map[t][a].index, f"nc_{a}"
            ].to_numpy()
            discharging_efficiency = model.bat.loc[
                model.discharge_map[t][a].index, f"nd_{a}"
            ].to_numpy()
            c[model.charge_map[t][a].to_numpy()] = 1 - charging_efficiency
            c[model.discharge_map[t][a].to_numpy()] = (1 / discharging_efficiency) - 1
    return c


# ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~


def cp_obj_loss(model: LinDistModelMulti, xk: cp.Variable, **kwargs) -> cp.Expression:
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    index_list = []
    r_list = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            i = model.x_maps[t][a].bi
            j = model.x_maps[t][a].bj
            r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
            r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
            index_list = np.append(
                index_list, model.x_maps[t][a].pij.to_numpy().flatten()
            )
            index_list = np.append(
                index_list, model.x_maps[t][a].qij.to_numpy().flatten()
            )
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    if isinstance(xk, cp.Variable):
        return cp.vdot(r, xk[ix] ** 2)
    else:
        return np.vdot(r, xk[ix] ** 2)


def cp_battery_efficiency(
    model: LinDistModelMulti, xk: cp.Variable, **kwargs
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    vec1_list = []
    index_list = []
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            charging_efficiency = model.bat.loc[
                model.charge_map[t][a].index, f"nc_{a}"
            ].to_numpy()
            discharging_efficiency = model.bat.loc[
                model.discharge_map[t][a].index, f"nd_{a}"
            ].to_numpy()
            vec1_list.extend((1 - charging_efficiency))
            vec1_list.extend(((1 / discharging_efficiency) - 1))
            index_list.extend(model.charge_map[t][a].to_numpy())
            index_list.extend(model.discharge_map[t][a].to_numpy())
    vec1 = np.array(vec1_list)
    ix = np.array(index_list)
    if isinstance(xk, cp.Variable):
        return 1e-3 * cp.vdot(vec1, xk[ix])
    else:
        return 1e-3 * np.vdot(vec1, xk[ix])


def cp_obj_loss_batt(
    model: LinDistModelMulti, xk: cp.Variable, **kwargs
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
    return cp_obj_loss(model, xk) + cp_battery_efficiency(model, xk)


def charge_batteries(model, xk, **kwargs) -> cp.Expression:
    f_list = []
    for t in range(model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            f_list.append(-cp.sum(xk[model.soc_map[t][a].to_numpy()]))
    return cp.sum(f_list)

#
# def peak_shave(model, xk):
#     f: cp.Expression = 0
#     subs = []
#     for t in range(LinDistModelQ.n):
#         ph = 0
#         for a in "abc":
#             ph += xk[model.idx("pij", model.swing_bus + 1, a, t)[0]]
#         subs.append(ph)
#         for j in range(1, model.nb):
#             for a in "abc":
#                 if model.phase_exists(a, t, j):
#                     if LinDistModelQ.battery:
#                         dis = model.idx("pd", j, a, t)
#                         ch = model.idx("pc", j, a, t)
#                         if ch:
#                             f += 1e-3 * (1 - model.bat["nc_" + a].get(j, 1)) * (xk[ch])
#                         if dis:
#                             f += (
#                                 1e-3
#                                 * ((1 / model.bat["nd_" + a].get(j, 1)) - 1)
#                                 * (xk[dis])
#                             )
#                         # if dis:
#                         #     f += 1e-5*((1/model.bat["nd_" + a].get(j,0))-model.bat["nc_" + a].get(j,0)) * (xk[dis])
#     f += cp.max(cp.hstack(subs))
#     return f
#
#
# peak_h = [17, 18, 19, 20, 21]
# peak_price = 19
# off_peak_price = 7
#
#
# def cost_min(model, xk):
#     f: cp.Expression = 0
#     for t in range(LinDistModelQ.n):
#         if t in peak_h:
#             peak = 0
#             for a in "abc":
#                 peak += xk[model.idx("pij", model.swing_bus + 1, a, t)[0]]
#             f += peak * peak_price * 10
#         else:
#             off_peak = 0
#             for a in "abc":
#                 off_peak += xk[model.idx("pij", model.swing_bus + 1, a, t)[0]]
#             f += off_peak * off_peak_price * 10
#         for j in range(1, model.nb):
#             for a in "abc":
#                 if model.phase_exists(a, t, j):
#                     if LinDistModelQ.battery:
#                         dis = model.idx("pd", j, a, t)
#                         ch = model.idx("pc", j, a, t)
#                         if ch:
#                             f += 1e-3 * (1 - model.bat["nc_" + a].get(j, 1)) * (xk[ch])
#                         if dis:
#                             f += (
#                                 1e-3
#                                 * ((1 / model.bat["nd_" + a].get(j, 1)) - 1)
#                                 * (xk[dis])
#                             )
#                         # if dis:
#                         #     f += 1e-3*((1/model.bat["nd_" + a].get(j,0))-model.bat["nc_" + a].get(j,0)) * (xk[dis])
#     return f


def cp_obj_target_p_3ph(model, xk, **kwargs):
    f = cp.Constant(0)
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(model, xk, **kwargs):
    target_q = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target_q[i] - q * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_curtail(
    model: LinDistModelMulti, xk: cp.Variable, **kwargs
) -> cp.Expression:
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

    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    all_pg_idx = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            all_pg_idx = np.r_[all_pg_idx, model.pg_map[t][a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]) ** 2)


def cp_obj_curtail_lp(
    model: LinDistModelMulti, xk: cp.Variable, **kwargs
) -> cp.Expression:
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

    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    all_pg_idx = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            all_pg_idx = np.r_[all_pg_idx, model.pg_map[t][a].to_numpy()]
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
    model: LinDistModelMulti,
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
    solver = kwargs.get("solver", cp.OSQP)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]

    if m.a_ineq.shape[0] != 0:
        h = [csr_array(m.a_ineq) @ x - m.b_ineq.flatten() <= 0]
    else:
        h = []
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    # error_percent = kwargs.get("error_percent", np.zeros(3))
    # target = kwargs.get("target", None)
    expression = obj_func(m, x, **kwargs)
    prob = cp.Problem(cp.Minimize(expression), g + h + ub + lb)
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
    model,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    pass


def lp_solve(
    model: (LinDistModelMulti, LinDistModelMultiFast), c: np.ndarray = None
) -> OptimizeResult:
    """
    Solve a linear program using scipy.optimize.linprog and having the objective function:
        Min c^T x
    Parameters
    ----------
    model : LinDistModel
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
    if c is None:
        c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(
        c,
        A_eq=csr_array(model.a_eq),
        b_eq=model.b_eq.flatten(),
        A_ub=csr_array(model.a_ineq),
        b_ub=model.b_ineq.flatten(),
        bounds=model.bounds,
    )
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res
