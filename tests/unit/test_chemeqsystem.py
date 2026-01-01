from unittest import TestCase
from sandlerchemeq.reaction import Reaction
from sandlerchemeq.component import Component
from sandlerchemeq.chemeqsystem import ChemEqSystem
import numpy as np
from numpy.testing import assert_allclose

class TestChemEqSystem(TestCase):
    def setUp(self):
        from sandlerprops.properties import get_database
        self.db = get_database()
    
    def test_chemeqsystem_ammonia_synthesis_lagrange(self):
        ammonia = Component.from_compound(self.db.get_compound('ammonia'), T=500.0, P=10.0)
        nitrogen = Component.from_compound(self.db.get_compound('nitrogen'), T=500.0, P=10.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=500.0, P=10.0)

        system = ChemEqSystem(Components=[ammonia, nitrogen, hydrogen],
                              N0=np.array([0.0, 1.0, 3.0]),
                              T=500.0,
                              P=100.0)

        system.solve_lagrange()
        expected_N = np.array([1.682838, 0.158581, 0.475742])
        assert_allclose(system.N, expected_N, rtol=1e-3)
  
    def test_chemeqsystem_ammonia_synthesis_extents(self):
        ammonia = Component.from_compound(self.db.get_compound('ammonia'), T=500.0, P=10.0)
        nitrogen = Component.from_compound(self.db.get_compound('nitrogen'), T=500.0, P=10.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=500.0, P=10.0)

        rxn = Reaction(components=[nitrogen, hydrogen, ammonia])

        system = ChemEqSystem(Components=[nitrogen, hydrogen, ammonia],
                              Reactions=[rxn],
                              N0=np.array([1.0, 3.0, 0.0]),
                              T=500.0,
                              P=100.0)
        
        system.solve_implicit(Xinit=[0.76])
        expected_N = np.array([0.158581, 0.475742,1.682838])
        assert_allclose(system.N, expected_N, rtol=1e-3)

    def test_chemeqsystem_water_gas_shift_lagrange(self):
        carbon_monoxide = Component.from_compound(self.db.get_compound('carbon monoxide'), T=600.0, P=10.0)
        water = Component.from_compound(self.db.get_compound('water'), T=600.0, P=10.0)
        carbon_dioxide = Component.from_compound(self.db.get_compound('carbon dioxide'), T=600.0, P=10.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=600.0, P=10.0)

        system = ChemEqSystem(Components=[carbon_monoxide, water, carbon_dioxide, hydrogen],
                              N0=np.array([1.0, 1.0, 0.0, 0.0]),
                              T=600.0,
                              P=10.0)

        system.solve_lagrange()
        expected_N = np.array([0.165479, 0.165479, 0.834521, 0.834521])
        assert_allclose(system.N, expected_N, rtol=1e-3)

    def test_chemeqsystem_water_gas_shift_extents(self):
        carbon_monoxide = Component.from_compound(self.db.get_compound('carbon monoxide'), T=600.0, P=10.0)
        water = Component.from_compound(self.db.get_compound('water'), T=600.0, P=10.0)
        carbon_dioxide = Component.from_compound(self.db.get_compound('carbon dioxide'), T=600.0, P=10.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=600.0, P=10.0)
        
        rxn = Reaction(components=[carbon_monoxide, water, carbon_dioxide, hydrogen])

        system = ChemEqSystem(Components=[carbon_monoxide, water, carbon_dioxide, hydrogen],
                              Reactions=[rxn],
                              N0=np.array([1.0, 1.0, 0.0, 0.0]),
                              T=600.0,
                              P=10.0)

        system.solve_implicit(Xinit=[0.8])

        expected_N = np.array([0.165479, 0.165479, 0.834521, 0.834521])
        assert_allclose(system.N, expected_N, rtol=1e-3)