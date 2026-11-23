"""Check the interval invariant, strict separation, and abstention on a tied boundary."""
from decimal import Decimal, localcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from examples import deepseek_router_bounds as router


class TestRouterBounds(unittest.TestCase):
    def test_encloses_possible_tails(self):
        """|tail|≤√(5·10) gives an interval containing f((3+tail)/8)+b for integer tails."""
        lo,hi = router.row_bounds(3,5,10,8,Decimal("-0.125"))
        with localcontext() as ctx:
            ctx.prec = 90
            for tail in range(-7,8):
                score = (1+((Decimal(3+tail)/8).exp())).ln().sqrt()-Decimal("0.125")
                self.assertLessEqual(lo,score)
                self.assertGreaterEqual(hi,score)

    def test_source_only_pruning_and_tie(self):
        """For x=e₁, differing first coordinates suffice; identical cutoff scores remain ambiguous."""
        matrix = np.array([[4,0,0,0],[2,0,0,0],[-2,0,0,0],[-4,0,0,0]],dtype="<f4")
        x = [1,0,0,0]
        with patch.multiple(router, EXPERTS=4, WIDTH=4, BLOCK=1, TOP=2), TemporaryDirectory() as tmp:
            path = Path(tmp)/"weights"
            for tied in (False,True):
                if tied:
                    matrix[2]=matrix[1]
                path.write_bytes((matrix.view("<u4")>>16).astype("<u2").tobytes())
                result = router.route(path,router.compile_norms(path),[Decimal(0)]*4,x)
                if tied:
                    self.assertIsNone(result["certified_set"])
                else:
                    self.assertEqual(result["certified_set"],[0,1])
                    self.assertLess(result["query_gate_bytes"],path.stat().st_size)
