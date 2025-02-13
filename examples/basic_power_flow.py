import distopf as opf

case = opf.DistOPFCase(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der",
    objective_functions=opf.cp_obj_loss,
    control_variable="PQ",
)
case.run_pf()
case.plot_network().show()
