import pandas as pd
import distopf as opf
from distopf.base import LinDistBase

class LinDistModel(LinDistBase):
    """
    LinDistFlow Model base class for linear power flow modeling.

    This class represents a linearized distribution model used for calculating
    power flows, voltages, and other system properties in a distribution network
    using the linearized branch-flow formulation from [1]. The model is composed of several power system components
    such as buses, branches, generators, capacitors, and regulators.

    Parameters
    ----------
    branch_data : pd.DataFrame
        DataFrame containing branch data including resistance and reactance values and limits.
    bus_data : pd.DataFrame
        DataFrame containing bus data such as loads, voltages, and limits.
    gen_data : pd.DataFrame
        DataFrame containing generator data.
    cap_data : pd.DataFrame
        DataFrame containing capacitor data.
    reg_data : pd.DataFrame
        DataFrame containing regulator data.

    References
    ----------
    [1] R. R. Jha, A. Dubey, C.-C. Liu, and K. P. Schneider,
    “Bi-Level Volt-VAR Optimization to Coordinate Smart Inverters
    With Voltage Control Devices,”
    IEEE Trans. Power Syst., vol. 34, no. 3, pp. 1801–1813,
    May 2019, doi: 10.1109/TPWRS.2018.2890613.
    """

    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        super().__init__(branch_data, bus_data, gen_data, cap_data, reg_data)
        self.pl_map, self.n_x = self._add_device_variables(self.n_x, self.load_buses)
        self.ql_map, self.n_x = self._add_device_variables(self.n_x, self.load_buses)
        self.build()

    def additional_variable_idx(self, var, node_j, phase):
        """
        User added index function. Override this function to add custom variables. Return None if `var` is not found.
        Parameters
        ----------
        var : name of variable
        node_j : node index (0 based; bus.id - 1)
        phase : "a", "b", or "c"

        Returns
        -------
        ix : index or list of indices of variable within x-vector or None if `var` is not found.
        """
        if var in ["pl", "p_load"]:  # reactive power load at node
            return self.pl_map[phase].get(node_j, [])
        if var in ["ql", "q_load"]:  # reactive power injection by capacitor
            return self.ql_map[phase].get(node_j, [])
        return None

    def add_load_model(self, a_eq, b_eq, j, phase):
        pij = self.idx("pij", j, phase)
        qij = self.idx("qij", j, phase)
        pl = self.idx("pl", j, phase)
        ql = self.idx("ql", j, phase)
        vj = self.idx("v", j, phase)
        p_load_nom, q_load_nom = 0, 0
        a_eq[pij, pl] = -1  # add load variable to power flow equation
        a_eq[qij, ql] = -1  # add load variable to power flow equation
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

    def get_p_loads(self, x):
        return self.get_device_variables(x, self.pl_map)

    def get_q_loads(self, x):
        return self.get_device_variables(x, self.ql_map)


if __name__ == '__main__':

    # Prepare the case data
    case = opf.DistOPFCase(data_path="ieee123_30der")
    # Initialize the LinDistModel
    model = LinDistModel(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    model2 = opf.LinDistModel(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    # Solve the model using the specified objective function
    result = opf.lp_solve(model, opf.gradient_load_min(model))
    result2 = opf.lp_solve(model2, opf.gradient_load_min(model2))
    # Extract and plot results
    v = model.get_voltages(result.x)
    v2 = model.get_voltages(result.x)
    s = model.get_apparent_power_flows(result.x)
    s2 = model2.get_apparent_power_flows(result.x)
    opf.compare_voltages(v, v2).show()
    opf.compare_flows(s, s2).show()