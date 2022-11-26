import copy

from olla import dataflow_graph, scheduler, utils


class MemoryPlanner:
    def __init__(self, timeout_s=None, rel_stop=None, optimize=False):
        self.timeout = timeout_s
        self.rel_stop = rel_stop
        self.optimize = optimize

    def plan(self, graph, mem_limit, user_schedule=None):
        optimized_graph = copy.deepcopy(graph)
        if self.optimize:
            if user_schedule is None:
                optimized_graph.prune(aggressive=True)
                optimized_graph.canonicalize()
                optimized_graph.constrain_allocations()
                optimized_graph.constrain_weights()
            else:
                optimized_graph.canonicalize()
                user_schedule = dataflow_graph.ScheduleConstraints(graph, user_schedule)
                user_schedule.fixup()
        else:
            optimized_graph.canonicalize()
        assert optimized_graph.check_consistency()

        solver = scheduler.Scheduler(optimized_graph)
        summary, schedule, mem_loc = solver.ComputeOptimalSchedule(
            mem_limit=mem_limit,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
        )
        assert utils.validate_address_allocation(mem_loc)
        assert utils.validate_timeline(schedule)

        # TBD parse result
        # The driver of each tensor is allocated at the time the corresponding op must run
        # The few tricks are:
        #  also schedule the nodes that were pruned
        #  insert malloc/free nodes. For malloc it's going to be the allocate nodes if any, or the edge driver
        # for free it's going to be right after the last use of a tensor.
        #  insert spill-out/spill-in nodes. For spill-in the schedule tells us when, for spill out we'll have to figure this out.
        return summary
