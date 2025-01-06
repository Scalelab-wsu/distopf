from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_array, vstack
import distopf as opf
from dataclasses import dataclass, field
from utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
    get,
)

@dataclass
class VarMap:
    a: pd.Series = field(default_factory=pd.Series)
    b: pd.Series = field(default_factory=pd.Series)
    c: pd.Series = field(default_factory=pd.Series)
    def __getitem__(self, key):
        return self.__dict__[key]

    def parse_x(self, x, bus):
        index = np.unique(
            np.r_[
                self.a.index,
                self.b.index,
                self.c.index,
            ]
        )
        bus_id = index + 1
        df = pd.DataFrame(columns=["id", "name", "a", "b", "c"], index=bus_id)
        df.id = bus_id
        df.loc[bus_id, "name"] = bus.loc[index, "name"].to_numpy()
        for a in "abc":
            df.loc[self[a].index + 1, a] = x[self[a]]
        return df

@dataclass
class BranchVarMap:
    a: pd.DataFrame = field(default_factory=pd.DataFrame(columns=["bi", "bj", "pij", "qij"]))
    b: pd.DataFrame = field(default_factory=pd.DataFrame(columns=["bi", "bj", "pij", "qij"]))
    c: pd.DataFrame = field(default_factory=pd.DataFrame(columns=["bi", "bj", "pij", "qij"]))

    def __getitem__(self, key):
        return self.__dict__[key]

    def parse_real(self, x, bus):
        nb = len(bus.id)
        s_df = pd.DataFrame(
            columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"], index=range(2, nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self[ph].bi  # self.x_maps[ph].bi.values
            fb_names = bus.name[fb_idxs].to_numpy()
            tb_idxs = self[ph].bj  #self.x_maps[ph].bj.values
            tb_names = bus.name[tb_idxs].to_numpy()
            s_df.loc[self[ph].bj + 1, "fb"] = fb_idxs + 1
            s_df.loc[self[ph].bj + 1, "tb"] = tb_idxs + 1
            s_df.loc[self[ph].bj + 1, "from_name"] = fb_names
            s_df.loc[self[ph].bj + 1, "to_name"] = tb_names
            s_df.loc[self[ph].bj + 1, ph] = (
                x[self[ph].pij]
            )
        return s_df

    def parse_imag(self, x, bus):
        nb = len(bus.id)
        s_df = pd.DataFrame(
            columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"], index=range(2, nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            fb_idxs = self[ph].bi  # self.x_maps[ph].bi.values
            fb_names = bus.name[fb_idxs].to_numpy()
            tb_idxs = self[ph].bj  #self.x_maps[ph].bj.values
            tb_names = bus.name[tb_idxs].to_numpy()
            s_df.loc[self[ph].bj + 1, "fb"] = fb_idxs + 1
            s_df.loc[self[ph].bj + 1, "tb"] = tb_idxs + 1
            s_df.loc[self[ph].bj + 1, "from_name"] = fb_names
            s_df.loc[self[ph].bj + 1, "to_name"] = tb_names
            s_df.loc[self[ph].bj + 1, ph] = (
                x[self[ph].qij]
            )
        return s_df


class Case:
    def __init__(
        self,
        branch_data: pd.DataFrame = None,
        bus_data: pd.DataFrame = None,
        gen_data: pd.DataFrame = None,
        cap_data: pd.DataFrame = None,
        reg_data: pd.DataFrame = None,
    ):
        # ~~~~~~~~~~~~~~~~~~~~ Load Data Frames ~~~~~~~~~~~~~~~~~~~~
        self.branch = handle_branch_input(branch_data)
        self.bus = handle_bus_input(bus_data)
        self.gen = handle_gen_input(gen_data)
        self.cap = handle_cap_input(cap_data)
        self.reg = handle_reg_input(reg_data)
        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.swing_bus = self.bus.loc[self.bus.bus_type == "SWING"].index[0]
        self.all_buses = {
            "a": self.bus.loc[self.bus.phases.str.contains("a")].index.to_numpy(),
            "b": self.bus.loc[self.bus.phases.str.contains("b")].index.to_numpy(),
            "c": self.bus.loc[self.bus.phases.str.contains("c")].index.to_numpy(),
        }
        self.load_buses = {
            "a": self.all_buses["a"][np.where(self.all_buses["a"] != self.swing_bus)],
            "b": self.all_buses["b"][np.where(self.all_buses["b"] != self.swing_bus)],
            "c": self.all_buses["c"][np.where(self.all_buses["c"] != self.swing_bus)],
        }
        self.gen_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.gen.shape[0] > 0:
            self.gen_buses = {
                "a": self.gen.loc[self.gen.phases.str.contains("a")].index.to_numpy(),
                "b": self.gen.loc[self.gen.phases.str.contains("b")].index.to_numpy(),
                "c": self.gen.loc[self.gen.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_gens = (
                len(self.gen_buses["a"])
                + len(self.gen_buses["b"])
                + len(self.gen_buses["c"])
            )
        self.cap_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.cap.shape[0] > 0:
            self.cap_buses = {
                "a": self.cap.loc[self.cap.phases.str.contains("a")].index.to_numpy(),
                "b": self.cap.loc[self.cap.phases.str.contains("b")].index.to_numpy(),
                "c": self.cap.loc[self.cap.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_caps = (
                len(self.cap_buses["a"])
                + len(self.cap_buses["b"])
                + len(self.cap_buses["c"])
            )
        self.reg_buses = dict(a=np.array([]), b=np.array([]), c=np.array([]))
        if self.reg.shape[0] > 0:
            self.reg_buses = {
                "a": self.reg.loc[self.reg.phases.str.contains("a")].index.to_numpy(),
                "b": self.reg.loc[self.reg.phases.str.contains("b")].index.to_numpy(),
                "c": self.reg.loc[self.reg.phases.str.contains("c")].index.to_numpy(),
            }
            self.n_regs = (
                len(self.reg_buses["a"])
                + len(self.reg_buses["b"])
                + len(self.reg_buses["c"])
            )


class IndexManager:
    """
    LinDistFlow Model base class.

    Parameters
    ----------
    case : Case
        Case containing branch, bus, gen, cap, and reg data
    """

    def __init__(
        self,
        case: Case,
    ):
        self.branch_maps, self.n_x = self._add_branch_variables(case)
        self.bi = VarMap(
            pd.Series(self.branch_maps.a.bi),
            pd.Series(self.branch_maps.b.bi),
            pd.Series(self.branch_maps.c.bi),
        )
        self.bj = VarMap(
            pd.Series(self.branch_maps.a.bj),
            pd.Series(self.branch_maps.b.bj),
            pd.Series(self.branch_maps.c.bj),
        )
        self.pij = VarMap(
            pd.Series(self.branch_maps.a.pij),
            pd.Series(self.branch_maps.b.pij),
            pd.Series(self.branch_maps.c.pij),
        )
        self.qij = VarMap(
            pd.Series(self.branch_maps.a.qij),
            pd.Series(self.branch_maps.b.qij),
            pd.Series(self.branch_maps.c.qij),
        )
        self.v, self.n_x = self._add_device_variables(self.n_x, case.all_buses)
        self.pl, self.n_x = self._add_device_variables(self.n_x, case.load_buses)
        self.ql, self.n_x = self._add_device_variables(self.n_x, case.load_buses)
        self.pg, self.n_x = self._add_device_variables(self.n_x, case.gen_buses)
        self.qg, self.n_x = self._add_device_variables(self.n_x, case.gen_buses)
        self.qc, self.n_x = self._add_device_variables(self.n_x, case.cap_buses)

    @staticmethod
    def _add_branch_variables(case, n_x=0):
        x_maps = {}
        for a in "abc":
            indices = case.branch.phases.str.contains(a)
            lines = case.branch.loc[indices, ["fb", "tb"]].values.astype(int) - 1
            n_lines = len(lines)
            df = pd.DataFrame(columns=["bi", "bj", "pij", "qij"], index=range(n_lines))
            if n_lines == 0:
                x_maps[a] = df.astype(int)
                continue
            g = nx.Graph()
            g.add_edges_from(lines)
            i_root = case.swing_bus
            # i_root = list(set(lines[:, 0]) - set(lines[:, 1]))[0]  # root node is only node with no from-bus
            edges = np.array(list(nx.dfs_edges(g, source=i_root)))
            df["bi"] = edges[:, 0]
            df["bj"] = edges[:, 1]
            df["pij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            df["qij"] = np.array([i for i in range(n_x, n_x + n_lines)])
            n_x = n_x + n_lines
            x_maps[a] = df.astype(int)
        branch_maps = BranchVarMap(x_maps["a"], x_maps["b"], x_maps["c"])
        return branch_maps, n_x

    @staticmethod
    def _add_device_variables(n_x: int, device_buses: dict):
        n_a = len(device_buses["a"])
        n_b = len(device_buses["b"])
        n_c = len(device_buses["c"])
        device_maps = VarMap(
            a=pd.Series(range(n_x, n_x + n_a), index=device_buses["a"]),
            b=pd.Series(range(n_x + n_a, n_x + n_a + n_b), index=device_buses["b"]),
            c=pd.Series(
                range(n_x + n_a + n_b, n_x + n_a + n_b + n_c), index=device_buses["c"]
            ),
        )
        n_x = n_x + n_a + n_b + n_c
        return device_maps, n_x

    @cache
    def branch_into_j(self, var: str, phase: str, bus_id: int = None):
        if bus_id is None:
            return self.branch_maps[phase].loc[:, var].to_numpy()
        idx = self.branch_maps[phase].loc[self.branch_maps[phase].bj == bus_id, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)

    @cache
    def branches_out_of_j(self, var: str, phase: str, bus_id: int = None):
        if bus_id is None:
            raise ValueError("bus_id is required to get pjk or qjk indices")
        idx = self.branch_maps[phase].loc[self.branch_maps[phase].bi == bus_id, var].to_numpy()
        return idx[~np.isnan(idx)].astype(int)



if __name__ == '__main__':
    pass


