"""Controls for registry coverage and exact observer-bound decisions."""
from fractions import Fraction
from itertools import product
import json
import unittest
from unittest.mock import patch

from examples import architecture_survey as survey
from examples.multi_model_readout import compile_rows, navigate


class TestArchitectureSurvey(unittest.TestCase):
    def test_exact_partial_decisions(self):
        """Compare sign(S_partial) to exact sign((w_i−w_j)·z+b_i−b_j) including ties."""
        left, right = [1.0, -1.5, 0.125], [0.5, 0.0, -0.125]
        compiled = compile_rows(left, right, 0.25, 0.0)
        for state in product([-1., -0.5, 0., 0.5, 1.], repeat=3):
            score = Fraction(1,4) + sum((Fraction(a)-Fraction(b))*Fraction(x)
                                        for a,b,x in zip(left,right,state))
            self.assertEqual(navigate(compiled,state)[0], (score>0)-(score<0))
        self.assertEqual(navigate(compile_rows([9.,0.], [0.,0.], 0.,0.), [1.,0.]), (1,1))
        self.assertEqual(navigate(compile_rows([1.,1.], [0.,0.], 0.,0.), [1.,-1.])[0], 0)

    def test_search_tag_requires_descriptor_confirmation(self):
        """Tag membership k∈tags does not imply descriptor.model_type=k."""
        listings = [{"id":"test/model","sha":"a"*40,"config":{"model_type":"mamba"}}]
        responses = [json.dumps(listings).encode(), b'{"model_type":"llama","architectures":["LlamaModel"]}']
        with patch.object(survey, "fetch", side_effect=responses):
            row = survey.survey(("registry","mamba"))
        self.assertEqual(row["status"], "unresolved")
        self.assertEqual(row["attempts"][0]["observed_type"], "llama")

    def test_registry_is_data(self):
        """K contains only literal mapping keys; no top-level source expression executes."""
        source = b'raise RuntimeError()\nCONFIG_MAPPING_NAMES = OrderedDict([("x","X"),("y","Y")])'
        self.assertEqual(survey.registry_keys(source), ["x","y"])
