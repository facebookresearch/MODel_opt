import unittest

from olla import dataflow_graph, utils
from olla.native_graphs import graph_with_weights


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.t1 = dataflow_graph.Edge(None, [], name="e1", size=128)
        self.t2 = dataflow_graph.Edge(None, [], name="e2", size=128)

    def testLineNumber(self):
        ln = utils.get_linenumber()
        self.assertEqual(ln, 13)

    def testExtractNodeOrdering(self):
        g = graph_with_weights.graph

        schedule = {
            g.edges["ACTIVATION_EDGE"]: ([1], [2], []),
            g.edges["WEIGHT_EDGE"]: ([1], [2, 3], []),
            g.edges["OUTPUT_EDGE"]: ([2], [], []),
        }
        ordering = utils.extract_node_ordering(g, schedule)
        print("ORDERING: " + str(ordering))
        self.assertEqual(
            str(ordering),
            """{INPUT (None): 1, WEIGHT (stateful_node): 1, OPERATOR (None): 2, OUTPUT (None): 3}""",
        )

    def testValidMem(self):
        mem = {
            1: {self.t1: 0, self.t2: 256},
            2: {self.t1: 0, self.t2: 256},
            3: {self.t1: 0, self.t2: 128},
        }
        self.assertTrue(utils.validate_address_allocation(mem))

    def testInvalidMem(self):
        mem = {
            1: {self.t1: 0, self.t2: 64},
            2: {self.t1: 0, self.t2: 64},
            3: {self.t1: 0, self.t2: 256},
        }
        self.assertFalse(utils.validate_address_allocation(mem))
