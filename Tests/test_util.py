from ..HydroCNHS.util import form_sim_seq, update_sim_seq_with_group


back_track_dict = {"N1": ["N2", "N3"],
                   "N2": ["R1"],
                   "R1": ["N4"],
                   "N3": ["R2"],
                   "R2": ["N5"]}
node_list = ["N1", "n1"]

sim_seq = form_sim_seq(node_list, back_track_dict)
update_sim_seq = update_sim_seq_with_group(sim_seq, ["N2","R2"], back_track_dict)

print(sim_seq)
print(update_sim_seq)