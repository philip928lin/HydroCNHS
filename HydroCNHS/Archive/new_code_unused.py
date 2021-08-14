#%% New formSimSeq (Done util.py)
import itertools
def form_sim_seq(node_list, back_tracking_dict):
    key_set = set(back_tracking_dict)
    if isinstance(node_list, str):
        if node_list not in key_set:
            return []
        node_list = [node_list]
    sim_seq = node_list
    def find_upstream_node(node):
        upstream_node = set(back_tracking_dict[node])
        not_top_node = upstream_node.intersection(key_set)
        top_node = upstream_node - not_top_node
        return list(not_top_node), list(top_node)
    while node_list != []:
        layer_nodes = []
        for node in node_list:
            not_top_node, top_node = find_upstream_node(node)
            sim_seq = top_node + sim_seq
            layer_nodes = layer_nodes + not_top_node
        # Back filling 
        sim_seq = layer_nodes + sim_seq
        node_list = layer_nodes
    return sim_seq

def update_sim_seq_with_group(sim_seq, group, back_tracking_dict):
    branch_dict = {}
    for node in group:
        branch_dict[node] = form_sim_seq(node, back_tracking_dict)[:-1]
    update_seq = list(
        itertools.chain.from_iterable(branch_dict.values())
        )
    update_seq = update_seq + group
    if len(update_seq) != len(set(update_seq)):
        print("Given group {} is not eligible. Update simulation sequence fail.".format(group))
        return sim_seq  # Not update
    else:
        remain_node = [n for n in sim_seq if n not in update_seq]
        update_seq = update_seq + remain_node
        return update_seq

form_sim_seq(node_list = "G", back_tracking_dict=BackTrackingDict)
form_sim_seq(node_list = ["C1","C2"], back_tracking_dict=BackTrackingDict)


sim_seq = form_sim_seq(node_list = "G", back_tracking_dict=BackTrackingDict)
update_seq = update_sim_seq_with_group(sim_seq, 
                                    group=["R2","S1"],
                                    back_tracking_dict=BackTrackingDict)