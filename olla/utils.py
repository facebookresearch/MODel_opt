from inspect import currentframe

import intervaltree
from olla import dataflow_graph


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def extract_node_ordering(graph, schedule):
    node_ordering = {}
    for t, s in schedule.items():
        generated = parse_schedule_item(s[0][0])
        if t.source in node_ordering:
            assert node_ordering[t.source] == generated
        else:
            node_ordering[t.source] = generated

    # Special case for nodes that have no fanout
    for n in graph.nodes.values():
        if len(n.fanout) > 0:
            continue
        if n.is_stateful():
            continue
        available = set()
        for i in range(0, len(n.fanin)):
            s = schedule[n.fanin[i]]

            fanin_availability = set(s[1])
            for generate in s[0]:
                fanin_availability.add(parse_schedule_item(generate) + 1)
            if i == 0:
                available = fanin_availability
            else:
                available.intersection_update(fanin_availability)

        if not available:
            continue
        else:
            node_ordering[n] = sorted(available)[0]

    return node_ordering


# Make sure that tensors never overlap in time
def validate_address_allocation(memory_locations, verbose=True):
    for ts, pairs in memory_locations.items():
        used_addresses = intervaltree.IntervalTree()
        for tensor, address in pairs.items():
            if tensor.size == 0:
                continue
            if used_addresses.overlaps(address, address + tensor.size):
                print(
                    f"tensor {tensor.name} at address {address} overlaps at timestep {ts} with {str(used_addresses[address : address + tensor.size])}",
                    flush=True,
                )
                return False
            used_addresses[address : address + tensor.size] = tensor

    return True


def parse_schedule_item(item):
    item = str(item)
    item = item.split("[")[0]
    items = item.split("@")
    assert len(items) > 0
    return int(items[0])


def check_op_inputs(tensor, allocate, schedule, verbose=True):
    assert tensor.source is not None
    op = tensor.source
    for fanin in op.fanin:
        preserve = schedule[fanin][1]
        spills = [parse_schedule_item(i) for i in schedule[fanin][2]]
        for t in allocate:
            if t not in preserve and t not in spills:
                if verbose:
                    print(
                        f"Input tensor {fanin.name} not in memory when op {op.name} driving {tensor.name} is run at time {t}"
                    )
                return False
    return True


def validate_timeline(schedule, verbose=True):
    for tensor, s in schedule.items():
        if isinstance(tensor, dataflow_graph.Node):
            continue

        allocate = [parse_schedule_item(i) for i in s[0]]

        if not check_op_inputs(tensor, allocate, schedule, verbose):
            return False

        if tensor.size == 0:
            continue
        preserve = s[1]
        if len(preserve) == 0:
            continue

        spills = [parse_schedule_item(i) for i in s[2]]

        if preserve[0] - 1 not in allocate and preserve[0] - 1 not in spills:
            print(
                f"tensor {tensor.name} was not allocated/fetched before timestep {preserve[0]}"
            )
            return False

        for i in range(1, len(preserve) - 1):
            if preserve[i] == preserve[i - 1] + 1:
                continue
            if preserve[i] - 1 in spills:
                continue
            if preserve[i] - 1 in allocate:
                continue

            print(
                f"tensor {tensor.name} was not allocated/fetched/preserved before timestep {preserve[i]}"
            )
            return False

    return True


def validate_node_ordering(graph, schedule, verbose=True):
    node_order = extract_node_ordering(graph, schedule)
    order = []
    for node, ts in node_order.items():
        order.append((ts, node))
    order.sort(key=lambda obj: obj[0])
    already_executed = set()
    for _, node in order:
        for fi in node.fanin:
            for src in fi.sources:
                if src not in already_executed:
                    print(
                        f"Invalid order: {node.name} scheduled before its fanin {src.name}"
                    )
                    print(f"Complete ordering {[node.name for _, node in order]}")
                    return False
        already_executed.add(node)
    return True
