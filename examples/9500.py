import distopf as opf

# dss_parser = opf.DSSParser(opf.CASES_DIR/"dss"/"9500-primary-network"/"Master.dss")
# dss_parser.to_csv(opf.CASES_DIR/"csv"/"9500")
case = opf.DistOPFCase(
    data_path="9500-primary-network",
    v_min=0.8,
    v_max=1.2,
)
case.reg_data = case.reg_data.loc[:, ["fb", "tb", "phases", "tap_a", "tap_b", "tap_c"]]
case.reg_data.loc[case.reg_data.fb == 407, "tap_a"] = 0
case.reg_data.loc[case.reg_data.fb == 407, "tap_b"] = 0
case.reg_data.loc[case.reg_data.fb == 407, "tap_c"] = 0
case.reg_data.loc[case.reg_data.fb == 147, "tap_a"] = 0
case.reg_data.loc[case.reg_data.fb == 147, "tap_b"] = 0
case.reg_data.loc[case.reg_data.fb == 147, "tap_c"] = 0
case.reg_data.loc[case.reg_data.fb == 345, "tap_a"] = 0
case.reg_data.loc[case.reg_data.fb == 345, "tap_b"] = 0
case.reg_data.loc[case.reg_data.fb == 345, "tap_c"] = 0
case.run_pf()
case.plot_network().write_html("9500_network.html")
case.plot_voltages().show()
results = case.run(opf.cp_obj_loss, control_capacitors=True)
case.plot_network().write_html("9500_network_loss.html")
case.plot_voltages().show()
pass
