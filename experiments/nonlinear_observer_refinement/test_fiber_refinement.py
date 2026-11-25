import unittest
from dataclasses import replace
from observer_closure import fixtures, form, verify
from fiber_refinement import form_fiber_plan, verify_plan, source_witness


class FiberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        actions,o = fixtures()
        cls.grounds = [form(actions[:k],o) for k in (1,2,3)]
        for k,g in enumerate(cls.grounds,1):
            if not verify(g,actions[:k],o):
                raise AssertionError("native VL-64 closure audit failed")

    def test_nonlinear_refinement(self):
        a,b,c = self.grounds
        for old,new,deficit,width in ((a,b,8,0),(b,c,13,1),(a,c,21,1)):
            plan = form_fiber_plan(old,new)
            self.assertEqual((plan.linear_deficit,plan.width),(deficit,width))
            self.assertTrue(verify_plan(plan,old,new))
            for x in range(old.size):
                self.assertEqual(plan.reconstruct(old.encode(x),source_witness(plan,old,new,x)),
                                 new.encode(x))
            self.assertFalse(verify_plan(replace(plan,width=width+1),old,new))

    def test_invalid_witness(self):
        a,b,c = self.grounds
        plan = form_fiber_plan(b,c)
        key = next(k for k,v in plan.rows if len(v)==1)
        with self.assertRaises(ValueError):
            plan.reconstruct(key,1)
        with self.assertRaises(ValueError):
            plan.reconstruct(key,2)


if __name__ == "__main__":
    unittest.main()
