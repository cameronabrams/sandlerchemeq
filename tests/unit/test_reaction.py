from unittest import TestCase
from sandlerchemeq.reaction import Reaction
from sandlerchemeq.component import Component
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

class TestReaction(TestCase):
    def setUp(self):
        from sandlerprops.properties import get_database
        self.db = get_database()

    def test_reaction_creation_and_str(self):
        ammonia = Component.from_compound(self.db.get_compound('ammonia'), T=298.15, P=1.0)
        nitrogen = Component.from_compound(self.db.get_compound('nitrogen'), T=298.15, P=1.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=298.15, P=1.0)

        rxn = Reaction(components=[ammonia, nitrogen, hydrogen])
        expected_stoichiometry = np.array([-2.0, 1.0, 3.0])
        assert_array_almost_equal(rxn.nu, expected_stoichiometry)

        expected_dGf = expected_stoichiometry[0]*ammonia.dGf + \
                        expected_stoichiometry[1]*nitrogen.dGf + \
                        expected_stoichiometry[2]*hydrogen.dGf
        assert_allclose(rxn.stoProps['dGf'], expected_dGf)
        expected_dHf = expected_stoichiometry[0]*ammonia.dHf + \
                        expected_stoichiometry[1]*nitrogen.dHf + \
                        expected_stoichiometry[2]*hydrogen.dHf
        assert_allclose(rxn.stoProps['dHf'], expected_dHf)
        expected_Cp = expected_stoichiometry[0]*ammonia.Cp + \
                        expected_stoichiometry[1]*nitrogen.Cp + \
                        expected_stoichiometry[2]*hydrogen.Cp
        assert_array_almost_equal(rxn.stoProps['Cp'], expected_Cp)

        expected_str = "2 NH3   <->   1 N2  +  3 H2"
        self.assertEqual(str(rxn), expected_str)

    def test_methane_combustion(self):
        methane = Component.from_compound(self.db.get_compound('methane'), T=298.15, P=1.0)
        oxygen = Component.from_compound(self.db.get_compound('oxygen'), T=298.15, P=1.0)
        carbon_dioxide = Component.from_compound(self.db.get_compound('carbon dioxide'), T=298.15, P=1.0)
        water = Component.from_compound(self.db.get_compound('water'), T=298.15, P=1.0)

        rxn = Reaction(components=[methane, oxygen, carbon_dioxide, water])
        expected_stoichiometry = np.array([-1.0, -2.0, 1.0, 2.0])
        assert_array_almost_equal(rxn.nu, expected_stoichiometry)

        expected_str = "1 CH4  +  2 O2   <->   1 CO2  +  2 H2O"
        self.assertEqual(str(rxn), expected_str)