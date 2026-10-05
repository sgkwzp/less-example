import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
from networkx.algorithms.lowest_common_ancestors import lowest_common_ancestor


'''
The code is modified based on Graph2Vid, ECCV2022
Please refer to [https://github.com/SamsungLabs/Graph2Vid] to access the official Graph2Vid implementation
'''

class Node:
    def __init__(self, node_id, parents):
        self.node_id = node_id
        self.parents = parents
        self.neighbors_up = set()

    def __repr__(self):
        return f"N{self.node_id}: p" + "_".join(str(p.node_id) for p in self.parents)

    def push_down_neighbors(self, neighbors_up):
        self.neighbors_up = self.neighbors_up.union(neighbors_up)
        if len(self.parents) == 0:
            self.neighbors_down = set()
        else:
            neigh = neighbors_up.union({self.node_id})
            all_neighbors_down = [p.push_down_neighbors(neigh) for p in self.parents]
            self.neighbors_down = set().union(*all_neighbors_down)

        return self.neighbors_down.union({self.node_id})

    def get_thread(self):
        return self.neighbors_up.union(self.neighbors_down)

    def get_parallel_nodes(self, all_nodes):
        return all_nodes - self.get_thread().union({self.node_id})


def compute_generalized_metadag_costs(
    sample,
    idx2node,
    drop_base=1.0,
    node_drop_base=-200
):
    """
    Parameters
    """

    action_logits = sample["action_logits"]

    action_prob = torch.softmax(action_logits, dim=0).permute(1, 0) # N x K

    sims = action_prob
    num_frames, num_steps = sims.size()

    ############## updated drop cost
    # use entropy of action probabilities to decide drop cost

    ## no drop 
    # baseline_logit = torch.tensor([0.0])
    # drop_logits = baseline_logit.repeat([1, num_frames])
    # drop_costs = -drop_logits.squeeze()

    ## dynamic drop
    # baseline_logit = torch.tensor([drop_base])
    # drop_logits = baseline_logit.repeat([1, num_frames])
    # drop_costs = -drop_logits.squeeze()
    # max_drop_cost = torch.tensor([1/(num_steps) for i in range(num_steps)])
    # max_drop_cost = - torch.sum(max_drop_cost * torch.log(max_drop_cost))
    # drop_costs = drop_costs.to(sims.device)
    # drop_costs = drop_costs * - torch.sum(sims.to(drop_costs.device) * torch.log(sims.to(drop_costs.device)), dim=1) / max_drop_cost

    ## fixed prob drop
    # 0.9: most of the frames become errors
    # 0.5: some background frrames become errors, the slip errors cannot be found

    # debugging
    # print(sample["video_id"])
    # zz = sims[sample["label"] == 3]
    # for i in range(len(zz)):
    #     print("%.3f"%(zz[i][3]), end=" ")
    # print()

    # baseline_logit = torch.tensor([drop_base])
    # drop_logits = baseline_logit.repeat([1, num_frames])  # making it of shape [1, N]
    # drop_costs = -drop_logits.squeeze()
    
    # topk
    # if drop_base == -100:
    #     baseline_logit = torch.tensor([0.0])
    #     drop_logits = baseline_logit.repeat([1, num_frames])  # making it of shape [1, N]
    #     drop_costs = -drop_logits.squeeze()
    # else:
    #     k = max([1, int(torch.numel(sims) * drop_base)])
    #     baseline_logit = torch.topk(sims.reshape([-1]), k).values[-1].detach()
    #     drop_logits = baseline_logit.repeat([1, num_frames])  # making it of shape [1, N]
    #     drop_costs = -drop_logits.squeeze()

    ## dynamic drop version 2
    if drop_base == -100:
        baseline_logit = torch.tensor([0.0])
        drop_logits = baseline_logit.repeat([1, num_frames])  # making it of shape [1, N]
        drop_costs = -drop_logits.squeeze()
    else:
        max_drop_cost = torch.tensor([1/(num_steps) for i in range(num_steps)])#.to(sims.device)
        max_drop_cost = - torch.sum(max_drop_cost * torch.log(max_drop_cost))
        drop_costs = - torch.sum(sims * torch.log(sims), dim=1) / max_drop_cost
        drop_costs = - (drop_costs + drop_base)

    # print("debug in graph_utils")
    # every frame drops

    # print(sample["video_id"])
    # spe_sim = sims[sample["label"] == 3]
    # spe_drop = drop_costs[sample["label"] == 3]
    # for i in range(len(spe_sim)):
    #     print("%.3f, %.3f/"%(spe_sim[i][3], spe_drop[i]), end=" ")
    # print()
    
    ############### updated node drop cost
    ### use highest action probability across frames to decide node drop cost

    ## no node drop
    # node_base_logits = torch.tensor([-200])
    # node_drop_logits = node_base_logits.repeat([num_steps])
    # node_drop_costs = -node_drop_logits

    # ## dynamic node drop
    node_base_logits = torch.tensor([node_drop_base])
    node_drop_logits = node_base_logits.repeat([num_steps])
    node_drop_costs = -node_drop_logits
    # values, _ = torch.topk(sims[:, :].to(node_drop_costs.device), 1, dim=0, largest=True)
    values, _ = torch.topk(sims[:, :], 1, dim=0, largest=True)
    node_drop_costs = node_drop_costs * (1 - values)
    node_drop_costs = node_drop_costs.squeeze(0)

    active_nodes = np.array([int(float(v.split(",")[0])) for v in idx2node.values()])

    meta_zx_costs = -sims.permute(1, 0).unsqueeze(0) # M = 1 x K x N
    zx_costs = meta_zx_costs[:, active_nodes, :]
    return zx_costs, drop_costs, node_drop_costs[active_nodes]


def generalized_metadag2vid(zx_costs, drop_costs, node_drop_costs, metadag, idx2node, return_meta_labels=False):
    """Generalized DAG-match algorithm that allows 
    1. drop frames or nodes. 
    2. match between differet types of steps
    
    See Algorithm xxx in the paper.

    Parameters
    ----------
    zx_costs: np.ndarray [M, K, N]
        pairwise match costs between M types (without addition), K steps and N video clips
    drop_costs: np.ndarray [N]
        drop costs for each clip
    node_drop_costs: np.ndarray [K]
        drop costs for each step
    metadag: networkx
        For each node, specifies a list of parents in the DAG.
        Assuming that the list is topologically sorted.
    exclusive: bool
        If True any clip can be matched with only one step, not many.
    return_label: bool
        if True, returns output directly useful for segmentation computation (made for convenience)
    """
    M, K, N = zx_costs.shape

    # prepare DAG parents in the usable format
    node2idx = {node_id: idx for idx, node_id in idx2node.items()}
    metadag_idx = dict()
    for idx, node in idx2node.items():
        parents_nodes = list(metadag.pred[node])
        parents_idxs = [node2idx[n] for n in parents_nodes]
        metadag_idx[idx] = parents_idxs

    # prepare the list of possible states to transition from
    prev_states_dict = dict()
    for node, parents in metadag_idx.items():
        if len(parents) == 0:
            prev_states_dict[node + 1] = [0]
        else:
            prev_states_dict[node + 1] = [s + 1 for s in parents]

    # initialize solutin matrices
    # the M + 2 last dimensions correspond to different states.
    # M types + 2 drops
    D = np.zeros([K + 1, N + 1, M + 2])


    # default matching  list: 0
    pos_states = [0]
    state2type = {}
    type_idx = 1
    normal_idx = [0, 1, 2]
    for i in range(M + 2):
        if i in normal_idx:
            state2type[i] = 0
        else: # matching for errors
            pos_states.append(i)
            state2type[i] = type_idx
            type_idx += 1
    

    # Setting the same for all DPs to change later here.
    D[1:, 0, :] = np.inf
    D[0, 1:, :] = np.inf
    D[0, 0, 1:] = np.inf

    # Allow to drop frame
    D[0, 1:, 1] = np.cumsum(drop_costs) # frame drop costs initialization in state 1
    # Allow to drop node
    D[1:, 0, 2] = np.cumsum(node_drop_costs) # node drop costs initialization in state 2

    # initialize path tracking info for each state
    P = dict()
    for xi in range(1, N + 1):
        P[(0, xi, 1)] = (0, xi - 1, 1)
    for zi in range(1, K + 1):
        prev_states = []
        for pz in prev_states_dict[zi]:
            prev_states.append((pz, 0, 2))
        P[(zi, 0, 2)] = prev_states

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # selecting the minimum cost transition (between pos and neg) for each preceeding state
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                min_idx = np.argmin(D[pz, xi - 1])
                prev_states_min.append((pz, xi - 1, min_idx))

            prev_costs = [D[s] for s in prev_states_min]
            argmin_prev_costs = np.array(prev_costs).argmin()
            min_prev_cost = prev_costs[argmin_prev_costs]
            best_prev_state = prev_states_min[argmin_prev_costs]

            # cur_states = [(zi, xi - 1, s) for s in [0, 1]]
            cur_states = [(zi, xi - 1, s) for s in range(M + 2)]
            cur_costs = [D[s] for s in cur_states]

            # all positive(matching) states
            cur_pos_states = [cur_states[s] for s in pos_states]
            cur_pos_costs = [D[s] for s in cur_pos_states]
            argmin_cur = np.array(cur_pos_costs).argmin()
            cur_pos_state = cur_pos_states[argmin_cur]
            cur_pos_cost = cur_pos_costs[argmin_cur]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # state other than 1 and 2: x is kept
            pi = 0
            for ps in pos_states:
                match_cost = zx_costs[pi][z_cost_ind, x_cost_ind]
                if cur_pos_cost < min_prev_cost:
                    D[zi, xi, ps] = cur_pos_cost + match_cost
                    P[(zi, xi, ps)] = cur_pos_state
                else:
                    D[zi, xi, ps] = min_prev_cost + match_cost
                    P[(zi, xi, ps)] = best_prev_state
                pi += 1

            # state 1: frame is dropped
            costs_neg = np.array(cur_costs) + drop_costs[x_cost_ind]
            opt_ind_neg = np.argmin(costs_neg)
            D[zi, xi, 1] = costs_neg[opt_ind_neg]
            P[(zi, xi, 1)] = cur_states[opt_ind_neg]

            # state 2: node is dropped
            prev_states_min = []
            for pz in prev_states_dict[zi]:
                min_idx = np.argmin(D[pz, xi])
                prev_states_min.append((pz, xi, min_idx))

            prev_costs = [D[s] for s in prev_states_min]

            # costs_neg = np.array(prev_costs) + node_drop_costs[z_cost_ind]
            # opt_ind_neg = np.argmin(costs_neg)
            # D[zi, xi, 2] = costs_neg[opt_ind_neg]
            # P[(zi, xi, 2)] = prev_states_min[opt_ind_neg]

            argmin_prev_costs = np.array(prev_costs).argmin()
            min_prev_cost = prev_costs[argmin_prev_costs]
            best_prev_state = prev_states_min[argmin_prev_costs]
            D[zi, xi, 2] = min_prev_cost + node_drop_costs[z_cost_ind]
            P[(zi, xi, 2)] = best_prev_state


    cur_state = D[K, N, :].argmin()

    # backtracking the solution
    # x dropped, z dropped
    labels = np.zeros([N], dtype=int)
    type_labels = np.zeros([N], dtype=int)
    meta_labels = [-1 for _ in range(N)]
    
    parents = [(K, N, cur_state)]
    while len(parents) > 0:
        zi, xi, cur_state = parents.pop(0)
        if xi > 0:
            # print(idx2node[zi - 1], xi, cur_state)
            meta_node_id = idx2node[zi - 1] if zi > 0 else -1
            meta_labels[xi - 1] = meta_node_id
            label = int(float(meta_node_id.split(",")[0])) if zi > 0 else -1
            # labels[xi - 1] = label if cur_state == 0 else -1
            if cur_state == 1:
                labels[xi - 1] = -1
            elif cur_state == 2:
                pass # do nothing, otherwise it may overwrite previous results
            else:
                labels[xi - 1] = label
            
            if cur_state == 2:
                pass  # do nothing, otherwise it may overwrite previous results
            else:
                type_labels[xi - 1] = state2type[cur_state]
            parents.append(P[(zi, xi, cur_state)])
    min_cost = D[K, N].min()

    if return_meta_labels:
        return min_cost, labels, type_labels, meta_labels
    else:
        return min_cost, labels, type_labels


def generate_metagraph(G):
    # add global sink to the graph
    presink_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    sink = 999
    G.add_node(sink)
    for node in presink_nodes:
        G.add_edge(node, sink)

    meta_G = nx.DiGraph()
    active_sink = str(sink)
    meta_G.add_node(active_sink)
    queue = [active_sink]
    while queue:
        state = queue.pop(0)
        components = [int(float(n)) for n in state.split(",")]
        active_node = components[0]
        parents = list(G.pred[active_node])
        new_components = parents + components[1:]
        for i in range(len(new_components)):
            new_active_node = new_components[i]
            par_nodes = new_components[0:i] + new_components[i + 1 :]

            # perform reduction
            par_nodes = [n for n in par_nodes if n != new_active_node]
            par_nodes = sorted(list(set(par_nodes)))

            new_state_components = [new_active_node] + par_nodes
            new_state = ",".join(str(c) for c in new_state_components)

            # checking for feasibility, i.e. like checking tokens
            feasible = True
            for par_node in par_nodes:
                if lowest_common_ancestor(G, new_active_node, par_node) == new_active_node:
                    # active node is an ancestor of some already matched nodes -> impossible -> reject candidate
                    feasible = False
            if not feasible:
                continue

            # add new state to meta_G
            if new_state not in meta_G.nodes():
                meta_G.add_node(new_state)

            if (new_state, state) not in meta_G.edges():
                meta_G.add_edge(new_state, state)
                if new_state not in queue:
                    queue.append(new_state)

    # remove sink from G
    for node in presink_nodes:
        G.remove_edge(node, sink)
    G.remove_node(sink)

    # remove sink from meta_G
    sink = str(sink)
    for node in list(meta_G.predecessors(sink)):
        meta_G.remove_edge(node, sink)
    meta_G.remove_node(sink)

    return meta_G

def remove_nodes_from_graph(G, nodes_to_remove, relabel=True):
    for node in sorted(nodes_to_remove):
        predecessors = list(G.predecessors(node))
        successors = list(G.successors(node))

        for pred in predecessors:
            G.remove_edge(pred, node)

        for succ in successors:
            G.remove_edge(node, succ)

        for pred in predecessors:
            for succ in successors:
                G.add_edge(pred, succ)
        G.remove_node(node)

    if relabel:
        mapping = {n: i for i, n in enumerate(G)}
        G = nx.relabel_nodes(G, mapping)
    return G
