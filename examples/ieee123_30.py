import numpy as np
import distopf as opf

case = opf.DistOPFCase(
    data_path="ieee123_30der",
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
# Solve model using provided objective function
result = opf.lp_solve(model, np.zeros(model.n_x))
# result = cvxpy_solve(model, cp_obj_loss)
print(result.fun)
v = model.get_voltages(result.x)
s = model.get_apparent_power_flows(result.x)
qg = model.get_q_gens(result.x)
fig = opf.plot_network(model, v, s, q_gen=qg).show()
opf.plot_voltages(v).show()
opf.plot_power_flows(s).show()

fig.write_image("123_plot.pdf", format="pdf", width=1000, height=1000)
import time

time.sleep(1)
fig.write_image("123_plot.pdf", format="pdf", width=1000, height=1000)
