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
    
        ammonia = Component.from_compound(self.db.get_compound('ammonia'), T=500.0, P=10.0)
        nitrogen = Component.from_compound(self.db.get_compound('nitrogen'), T=500.0, P=10.0)
        hydrogen = Component.from_compound(self.db.get_compound('hydrogen (equilib)'), T=500.0, P=10.0)
        carbon_monoxide = Component.from_compound(self.db.get_compound('carbon monoxide'), T=600.0, P=10.0)
        water = Component.from_compound(self.db.get_compound('water'), T=600.0, P=10.0)
        carbon_dioxide = Component.from_compound(self.db.get_compound('carbon dioxide'), T=600.0, P=10.0)

        self.ammonia_syn_components = [nitrogen, hydrogen, ammonia]
        self.ammonia_syn_N0 = np.array([1.0, 3.0, 0.0])
        self.ammonia_syn_result = np.array([0.158581, 0.475742, 1.682838])

        self.water_gas_shift_components = [carbon_monoxide, water, carbon_dioxide, hydrogen]
        self.water_gas_shift_N0 = np.array([1.0, 1.0, 0.0, 0.0])
        self.water_gas_shift_result = np.array([0.165479, 0.165479, 0.834521, 0.834521])

    def test_chemeqsystem_ammonia_synthesis_lagrange(self):
        system = ChemEqSystem(components=self.ammonia_syn_components,
                              N0=self.ammonia_syn_N0,
                              T=500.0,
                              P=100.0)
        system.solve_lagrange()
        expected_N = self.ammonia_syn_result
        assert_allclose(system.N, expected_N, rtol=1e-3)
  
    def test_chemeqsystem_ammonia_synthesis_extents(self):
        rxn = Reaction(components=self.ammonia_syn_components)
        system = ChemEqSystem(components=self.ammonia_syn_components,
                              reactions=[rxn],
                              N0=self.ammonia_syn_N0,
                              T=500.0,
                              P=100.0)
        
        system.solve_implicit(Xinit=[0.76], simplified=False)
        expected_N = self.ammonia_syn_result
        assert_allclose(system.N, expected_N, rtol=1e-3)

    def test_chemeqsystem_water_gas_shift_lagrange(self):
        system = ChemEqSystem(components=self.water_gas_shift_components,
                              N0=self.water_gas_shift_N0,
                              T=600.0,
                              P=10.0)

        system.solve_lagrange()
        expected_N = self.water_gas_shift_result
        assert_allclose(system.N, expected_N, rtol=1e-3)

    def test_chemeqsystem_water_gas_shift_extents(self):
        rxn = Reaction(components=self.water_gas_shift_components)

        system = ChemEqSystem(components=self.water_gas_shift_components,
                              reactions=[rxn],
                              N0=self.water_gas_shift_N0,
                              T=600.0,
                              P=10.0)

        system.solve_implicit(Xinit=[0.8], simplified=False)

        expected_N = self.water_gas_shift_result
        assert_allclose(system.N, expected_N, rtol=1e-3)