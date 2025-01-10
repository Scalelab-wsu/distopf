from functools import cache
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array
import distopf as opf
from index_manager import IndexManager, Case



def get(s: pd.Series, i, default=None):
    """
    Get value at index i from a Series. Return default if it does not exist.
    Parameters
    ----------
    s : pd.Series
    i : index or key for eries
    default : value to return if it fails

    Returns
    -------
    value: value at index i or default if it doesn't exist.
    """
    try:
        return s.loc[i]
    except (KeyError, ValueError, IndexError):
        return default

class LinDistModelExperimental:
    """
    LinDistFlow Model base class.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data (r and x values, limits)
    bus_data : pd.DataFrame
        DataFrame containing bus data (loads, voltages, limits)
    gen_data : pd.DataFrame
        DataFrame containing generator/DER data
    cap_data : pd.DataFrame
        DataFrame containing capacitor data
    reg_data : pd.DataFrame
        DataFrame containing regulator data

    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        self.case = Case(
            branch_data=branch_data,
            bus_data=bus_data,
            gen_data=gen_data,
            cap_data=cap_data,
            reg_data=reg_data,
        )
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = self.case.branch
        self.bus = self.case.bus
        self.gen = self.case.gen
        self.cap = self.case.cap
        self.reg = self.case.reg

        index_manager = IndexManager(self.case)
        self.nb = self.case.nb
        self.n_x = index_manager.n_x
        self.idx = index_manager
        # ~~~~~~~~~~~~~~~~~~~~ initialize r and x ~~~~~~~~~~~~~~~~~~~~
        self.r, self.x = self._init_rx(self.branch)
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq ~~~~~~~~~~~~~~~~~~~~

        self._a_eq, self._b_eq = None, None
        self._a_ub, self._b_ub = None, None
        self._bounds = None
        self._bounds_tuple = None


    @staticmethod
    def _init_rx(branch):
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
        r = {
            "aa": csr_array((np.r_[branch.raa, branch.raa], (row, col))),
            "ab": csr_array((np.r_[branch.rab, branch.rab], (row, col))),
            "ac": csr_array((np.r_[branch.rac, branch.rac], (row, col))),
            "bb": csr_array((np.r_[branch.rbb, branch.rbb], (row, col))),
            "bc": csr_array((np.r_[branch.rbc, branch.rbc], (row, col))),
            "cc": csr_array((np.r_[branch.rcc, branch.rcc], (row, col))),
        }
        x = {
            "aa": csr_array((np.r_[branch.xaa, branch.xaa], (row, col))),
            "ab": csr_array((np.r_[branch.xab, branch.xab], (row, col))),
            "ac": csr_array((np.r_[branch.xac, branch.xac], (row, col))),
            "bb": csr_array((np.r_[branch.xbb, branch.xbb], (row, col))),
            "bc": csr_array((np.r_[branch.xbc, branch.xbc], (row, col))),
            "cc": csr_array((np.r_[branch.xcc, branch.xcc], (row, col))),
        }
        return r, x

    def init_bounds(self):
        default = 100e3  # Default for unbounded variables.
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -default
        x_lim_upper = np.ones(self.n_x) * default
        x_lim_lower, x_lim_upper = self.add_voltage_limits(x_lim_lower, x_lim_upper)
        x_lim_lower, x_lim_upper = self.add_generator_limits(x_lim_lower, x_lim_upper)
        x_lim_lower, x_lim_upper = self.user_added_limits(x_lim_lower, x_lim_upper)
        bounds = np.c_[x_lim_lower, x_lim_upper]
        # bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    def user_added_limits(self, x_lim_lower, x_lim_upper):
        """
        User added limits function. Override this function to add custom variable limits.
        Parameters
        ----------
        x_lim_lower :
        x_lim_upper :

        Returns
        -------
        x_lim_lower : lower limits for x-vector
        x_lim_upper : upper limits for x-vector

        Examples
        --------
        ```python
        p_lim = 10
        q_lim = 10
        for a in "abc":
            if not self.phase_exists(a):
                continue
            x_lim_lower[self.x_maps[a].pij] = -p_lim
            x_lim_upper[self.x_maps[a].pij] = p_lim
            x_lim_lower[self.x_maps[a].qij] = -q_lim
            x_lim_upper[self.x_maps[a].qij] = q_lim
        ```
        """
        return x_lim_lower, x_lim_upper

    def add_voltage_limits(self, x_lim_lower, x_lim_upper):
        for a in "abc":
            if not self.phase_exists(a):
                continue
            # ~~ v limits ~~:
            x_lim_upper[self.idx.v[a]] = self.bus.loc[self.idx.v[a].index, "v_max"] ** 2
            x_lim_lower[self.idx.v[a]] = self.bus.loc[self.idx.v[a].index, "v_min"] ** 2
        return x_lim_lower, x_lim_upper

    def add_generator_limits(self, x_lim_lower, x_lim_upper):
        for a in "abc":
            if not self.phase_exists(a):
                continue
            q_max_manual = self.gen[f"q{a}_max"]
            q_min_manual = self.gen[f"q{a}_min"]
            s_rated = self.gen[f"s{a}_max"]
            p_out = self.gen[f"p{a}"]
            q_min = -1 * (((s_rated**2) - (p_out**2)) ** (1 / 2))
            q_max = ((s_rated**2) - (p_out**2)) ** (1 / 2)
            for j in self.case.gen_buses[a]:
                mode = self.gen.loc[j, f"{a}_mode"]
                pg = self.idx.pg[a][j]
                qg = self.idx.pg[a][j]
                # active power bounds
                x_lim_lower[pg] = 0
                x_lim_upper[pg] = p_out[j]
                # reactive power bounds
                if mode == opf.CONSTANT_P:
                    x_lim_lower[qg] = max(q_min[j], q_min_manual[j])
                    x_lim_upper[qg] = min(q_max[j], q_max_manual[j])
                if mode != opf.CONSTANT_P:
                    # reactive power bounds
                    x_lim_lower[qg] = max(-s_rated[j], q_min_manual[j])
                    x_lim_upper[qg] = min(s_rated[j], q_max_manual[j])
        return x_lim_lower, x_lim_upper

    @cache
    def phase_exists(self, phase, index: int = None):
        if index is None:
            return self.idx.branch_maps[phase].shape[0] > 0
        return len(np.array([self.idx.bj[phase][index]]).flatten()) > 0

    def create_model(self):
        # ########## Aeq and Beq Formation ###########
        n_rows = self.n_x
        n_cols = self.n_x
        # Aeq has the same number of rows as equations with a column for each x
        a_eq = zeros((n_rows, n_cols))
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                if not self.phase_exists(a, j):
                    continue
                a_eq, b_eq = self.add_power_flow_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_voltage_drop_model(a_eq, b_eq, j, a, b, c)
                a_eq, b_eq = self.add_swing_voltage_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_regulator_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_load_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_generator_model(a_eq, b_eq, j, a)
                a_eq, b_eq = self.add_capacitor_model(a_eq, b_eq, j, a)
        return csr_array(a_eq), b_eq

    def add_power_flow_model(self, a_eq, b_eq, j, phase):
        pij = self.idx.pij[phase][j]
        qij = self.idx.qij[phase][j]
        pjk = self.idx.branches_out_of_j("pij", phase, j)
        qjk = self.idx.branches_out_of_j("qij", phase, j)
        pl = self.idx.pl[phase][j]
        ql = self.idx.ql[phase][j]
        pg = self.idx.pg[phase][j]
        qg = self.idx.qg[phase][j]
        qc = self.idx.qc[phase][j]
        # Set P equation variable coefficients in a_eq
        a_eq[pij, pij] = 1
        a_eq[pij, pjk] = -1
        a_eq[pij, pl] = -1
        a_eq[pij, pg] = 1
        # Set Q equation variable coefficients in a_eq
        a_eq[qij, qij] = 1
        a_eq[qij, qjk] = -1
        a_eq[qij, ql] = -1
        a_eq[qij, qg] = 1
        a_eq[qij, qc] = 1
        return a_eq, b_eq

    def add_voltage_drop_model(self, a_eq, b_eq, j, a, b, c):
        if self.reg is not None:
            if j in self.reg.tb:
                return a_eq, b_eq
        r, x = self.r, self.x
        aa = "".join(sorted(a + a))
        # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
        ab = "".join(sorted(a + b))
        ac = "".join(sorted(a + c))
        i = self.idx.bi[a][j][0]  # get the upstream node, i, on branch from i to j
        pij = self.idx.pij[a][j]
        qij = self.idx.qij[a][j]
        pijb = self.idx.pij[b][j]
        qijb = self.idx.qij[b][j]
        pijc = self.idx.pij[c][j]
        qijc = self.idx.qij[c][j]
        vi = self.idx.v[a][i]
        vj = self.idx.v[a][j]

        # Set V equation variable coefficients in a_eq and constants in b_eq
        a_eq[vj, vj] = 1
        a_eq[vj, vi] = -1
        a_eq[vj, pij] = 2 * r[aa][i, j]
        a_eq[vj, qij] = 2 * x[aa][i, j]
        if self.phase_exists(b, j):
            a_eq[vj, pijb] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
            a_eq[vj, qijb] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
        if self.phase_exists(c, j):
            a_eq[vj, pijc] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
            a_eq[vj, qijc] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]
        return a_eq, b_eq


    def add_regulator_model(self, a_eq, b_eq, j, a):
        i = self.idx.bi[a][j][0]  # get the upstream node, i, on branch from i to j
        vi = self.idx.v[a][i]
        vj = self.idx.v[a][j]

        if self.reg is not None:
            if j in self.reg.tb:
                reg_ratio = get(self.reg[f"ratio_{a}"], j, 1)
                a_eq[vj, vj] = 1
                a_eq[vj, vi] = -1 * reg_ratio**2
                return a_eq, b_eq
        return a_eq, b_eq

    def add_swing_voltage_model(self, a_eq, b_eq, j, a):
        i = self.idx.bi[a][j][0]  # get the upstream node, i, on branch from i to j
        vi = self.idx.v[a][i]
        # Set V equation variable coefficients in a_eq and constants in b_eq
        if self.bus.bus_type[i] == opf.SWING_BUS:  # Swing bus
            a_eq[vi, vi] = 1
            b_eq[vi] = self.bus.at[i, f"v_{a}"] ** 2
        return a_eq, b_eq

    def add_generator_model(self, a_eq, b_eq, j, phase):
        a = phase
        p_gen_nom, q_gen_nom = 0, 0
        if self.gen is not None:
            p_gen_nom = get(self.gen[f"p{a}"], j, 0)
            q_gen_nom = get(self.gen[f"q{a}"], j, 0)
        # equation indexes
        pg = self.idx.pg[a][j]
        qg = self.idx.qg[a][j]
        # Set Generator equation variable coefficients in a_eq
        if get(self.gen[f"{a}_mode"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_P]:
            a_eq[pg, pg] = 1
            b_eq[pg] = p_gen_nom
        if get(self.gen[f"{a}_mode"], j, 0) in [opf.CONSTANT_PQ, opf.CONSTANT_Q]:
            a_eq[qg, qg] = 1
            b_eq[qg] = q_gen_nom
        return a_eq, b_eq

    def add_load_model(self, a_eq, b_eq, j, phase):
        pl = self.idx.pl[phase][j]
        ql = self.idx.ql[phase][j]
        vj = self.idx.v[phase][j]
        p_load_nom, q_load_nom = 0, 0
        if self.bus.bus_type[j] == opf.PQ_BUS:
            p_load_nom = self.bus[f"pl_{phase}"][j]
            q_load_nom = self.bus[f"ql_{phase}"][j]
        if self.bus.bus_type[j] != opf.PQ_FREE:
            # Set Load equation variable coefficients in a_eq
            a_eq[pl, pl] = 1
            a_eq[pl, vj] = -(self.bus.cvr_p[j] / 2) * p_load_nom
            b_eq[pl] = (1 - (self.bus.cvr_p[j] / 2)) * p_load_nom
            a_eq[ql, ql] = 1
            a_eq[ql, vj] = -(self.bus.cvr_q[j] / 2) * q_load_nom
            b_eq[ql] = (1 - (self.bus.cvr_q[j] / 2)) * q_load_nom
        return a_eq, b_eq

    def add_capacitor_model(self, a_eq, b_eq, j, phase):
        q_cap_nom = 0
        if self.cap is not None:
            q_cap_nom = get(self.cap[f"q{phase}"], j, 0)
        # equation indexes
        vj = self.idx.v[phase][j]
        qc = self.idx.qc[phase][j]
        a_eq[qc, qc] = 1
        a_eq[qc, vj] = -q_cap_nom
        return a_eq, b_eq

    def create_inequality_constraints(self):
        a_ub, b_ub = self.create_octagon_constraints()
        return csr_array(a_ub), b_ub

    def create_hexagon_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 6
        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
            + len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
            + len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4
        ineq6 = 5

        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                if self.gen.loc[j, f"{a}_mode"] != "CONTROL_PQ":
                    continue
                pg = self.idx.pg[a][j]
                qg = self.idx.qg[a][j]
                s_rated = self.gen.at[j, f"s{a}_max"]
                # equation indexes
                a_ineq[ineq1, pg] = -sqrt(3)
                a_ineq[ineq1, qg] = -1
                b_ineq[ineq1] = sqrt(3) * s_rated
                a_ineq[ineq2, pg] = sqrt(3)
                a_ineq[ineq2, qg] = 1
                b_ineq[ineq2] = sqrt(3) * s_rated
                a_ineq[ineq3, qg] = -1
                b_ineq[ineq3] = sqrt(3) / 2 * s_rated
                a_ineq[ineq4, qg] = 1
                b_ineq[ineq4] = sqrt(3) / 2 * s_rated
                a_ineq[ineq5, pg] = sqrt(3)
                a_ineq[ineq5, qg] = -1
                b_ineq[ineq5] = sqrt(3) * s_rated
                a_ineq[ineq6, pg] = -sqrt(3)
                a_ineq[ineq6, qg] = 1
                b_ineq[ineq6] = -sqrt(3) * s_rated
                ineq1 += 6
                ineq2 += 6
                ineq3 += 6
                ineq4 += 6
                ineq5 += 6
                ineq6 += 6

        return csr_array(a_ineq), b_ineq

    def create_octagon_constraints(self):
        """
        Create inequality constraints for the optimization problem.
        """

        # ########## Aineq and Bineq Formation ###########
        n_inequalities = 5

        n_rows_ineq = n_inequalities * (
            len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
            + len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
            + len(np.where(self.gen.control_variable == "CONTROL_PQ")[0])
        )
        n_rows_ineq = max(n_rows_ineq, 1)
        a_ineq = zeros((n_rows_ineq, self.n_x))
        b_ineq = zeros(n_rows_ineq)
        ineq1 = 0
        ineq2 = 1
        ineq3 = 2
        ineq4 = 3
        ineq5 = 4

        for j in self.gen.index:
            for a in "abc":
                if not self.phase_exists(a, j):
                    continue
                if self.gen.loc[j, f"{a}_mode"] != "CONTROL_PQ":
                    continue
                pg = self.idx.pg[a][j]
                qg = self.idx.qg[a][j]
                s_rated = self.gen.at[j, f"s{a}_max"]
                # equation indexes
                a_ineq[ineq1, pg] = sqrt(2)
                a_ineq[ineq1, qg] = -2 + sqrt(2)
                b_ineq[ineq1] = sqrt(2) * s_rated
                a_ineq[ineq2, pg] = sqrt(2)
                a_ineq[ineq2, qg] = 2 - sqrt(2)
                b_ineq[ineq2] = sqrt(2) * s_rated
                a_ineq[ineq3, pg] = -1 + sqrt(2)
                a_ineq[ineq3, qg] = 1
                b_ineq[ineq3] = s_rated
                a_ineq[ineq4, pg] = -1 + sqrt(2)
                a_ineq[ineq4, qg] = -1
                b_ineq[ineq4] = s_rated
                a_ineq[ineq5, pg] = -1
                b_ineq[ineq5] = 0
                ineq1 += n_inequalities
                ineq2 += n_inequalities
                ineq3 += n_inequalities
                ineq4 += n_inequalities
                ineq5 += n_inequalities

        return csr_array(a_ineq), b_ineq

    def get_device_variables(self, x, variable_map):
        index = np.unique(
            np.r_[
                variable_map["a"].index,
                variable_map["b"].index,
                variable_map["c"].index,
            ]
        )
        bus_id = index + 1
        df = pd.DataFrame(columns=["id", "name", "a", "b", "c"], index=bus_id)
        df.id = bus_id
        df.loc[bus_id, "name"] = self.bus.loc[index, "name"].to_numpy()
        for a in "abc":
            df.loc[variable_map[a].index + 1, a] = x[variable_map[a]]
        return df

    def get_voltages(self, x):
        v_df = self.idx.v.parse_x(x, self.bus)
        v_df.loc[:, ["a", "b", "c"]] = v_df.loc[:, ["a", "b", "c"]] ** 0.5
        return v_df

    def get_p_loads(self, x):
        return self.idx.pl.parse_x(x, self.bus)

    def get_q_loads(self, x):
        return self.idx.ql.parse_x(x, self.bus)

    def get_q_gens(self, x):
        return self.idx.qg.parse_x(x, self.bus)

    def get_p_gens(self, x):
        return self.idx.pg.parse_x(x, self.bus)

    def get_q_caps(self, x):
        return self.idx.qc.parse_x(x, self.bus)

    def get_apparent_power_flows(self, x):

        s_df = pd.DataFrame(
            columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"], index=range(2, self.nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self.idx.bi[ph].to_numpy()
            fb_names = self.bus.name[fb_idxs].to_numpy()
            tb_idxs = self.idx.bj[ph].to_numpy()
            tb_names = self.bus.name[tb_idxs].to_numpy()
            s_df.loc[tb_idxs + 1, "fb"] = fb_idxs + 1
            s_df.loc[tb_idxs + 1, "tb"] = tb_idxs + 1
            s_df.loc[tb_idxs + 1, "from_name"] = fb_names
            s_df.loc[tb_idxs + 1, "to_name"] = tb_names
            s_df.loc[tb_idxs + 1, ph] = (
                x[self.idx.pij[ph]] + 1j * x[self.idx.qij[ph]]
            )
        return s_df

    def update(
        self,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        if bus_data is not None:
            self.bus = bus_data
        if gen_data is not None:
            self.gen = gen_data
        if cap_data is not None:
            self.cap = cap_data
        if reg_data is not None:
            self.reg = reg_data


        for j in range(1, self.nb):
            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                if not self.phase_exists(a, j):
                    continue
                if bus_data is not None:
                    self._a_eq, self._b_eq = self.add_swing_voltage_model(self.a_eq, self.b_eq, j, a)
                    self._a_eq, self._b_eq = self.add_load_model(self.a_eq, self.b_eq, j, a)
                if gen_data is not None:
                    self._a_eq, self._b_eq = self.add_generator_model(self.a_eq, self.b_eq, j, a)
                if cap_data is not None:
                    self._a_eq, self._b_eq = self.add_capacitor_model(self.a_eq, self.b_eq, j, a)
                if reg_data is not None:
                    self._a_eq, self._b_eq = self.add_regulator_model(self.a_eq, self.b_eq, j, a)

    @property
    def branch_data(self):
        return self.branch

    @property
    def bus_data(self):
        return self.bus

    @property
    def gen_data(self):
        return self.gen

    @property
    def cap_data(self):
        return self.cap

    @property
    def reg_data(self):
        return self.reg

    @property
    def a_eq(self):
        if self._a_eq is None:
            self._a_eq, self._b_eq = self.create_model()
        return self._a_eq

    @property
    def b_eq(self):
        if self._b_eq is None:
            self._a_eq, self._b_eq = self.create_model()
        return self._b_eq

    @property
    def a_ub(self):
        if self._a_ub is None:
            self._a_ub, self._b_ub = self.create_inequality_constraints()
        return self._a_ub

    @property
    def b_ub(self):
        if self._b_ub is None:
            self._a_ub, self._b_ub = self.create_inequality_constraints()
        return self._b_ub

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        if self._bounds_tuple is None:
            self._bounds_tuple = list(map(tuple, self._bounds))
        return self._bounds_tuple

    @property
    def x_min(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        return self._bounds[:, 0]

    @property
    def x_max(self):
        if self._bounds is None:
            self._bounds = self.init_bounds()
        return self._bounds[:, 1]

if __name__ == '__main__':
    from distopf import CASES_DIR

    case = opf.DistOPFCase(
        data_path=CASES_DIR/"csv"/"ieee123_30der",
        gen_mult=1,
        load_mult=1,
        v_swing=1.0,
        v_max=1.05,
        v_min=0.95,
    )

    model = opf.LinDistModel(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    model2 = LinDistModelExperimental(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,)

    result = opf.lp_solve(model, np.zeros(model.n_x))
    result2 = opf.lp_solve(model2, np.zeros(model.n_x))
    print(result.runtime)
    print(result2.runtime)