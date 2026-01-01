from unittest import TestCase
from sandlerchemeq.component import Component
from sandlerprops.properties import PropertiesDatabase, get_database

class TestComponent(TestCase):

    def test_component_raw(self):
        comp = Component(Name='water', Formula='H2O', Molwt=18.01528, dHf=-285.83, dGf=-237.13,
                         CpA=33.58, CpB=0.213, CpC=-0.00065, CpD=0.000001)
        self.assertEqual(comp.Name, 'water')
        self.assertEqual(comp.Formula, 'H2O')
        self.assertEqual(comp.Molwt, 18.01528)
        self.assertEqual(comp.dHf, -285.83)
        self.assertEqual(comp.dGf, -237.13)
        self.assertEqual(comp.CpA, 33.58)
        self.assertEqual(comp.CpB, 0.213)
        self.assertEqual(comp.CpC, -0.00065)
        self.assertEqual(comp.CpD, 0.000001)

    def test_component_inheritance_from_compound(self):
        db = get_database()
        compound = db.get_compound('ethanol')
        comp = Component.from_compound(compound, T=350, P=5)
        self.assertEqual(comp.Name, compound.Name)
        self.assertEqual(comp.Formula, compound.Formula)
        self.assertEqual(comp.Molwt, compound.Molwt)
        self.assertEqual(comp.T, 350)
        self.assertEqual(comp.P, 5)
        self.assertEqual(comp.dHf, compound.dHf)
        self.assertEqual(comp.countAtoms('C'), 2)

        self.assertEqual(comp.dGf_T, -157108.75161900392)

    def test_component_no_compound(self):
        comp = Component(Name='unobtainium', Formula='X3Y4')
        self.assertEqual(comp.Name, 'unobtainium')
        self.assertEqual(comp.Formula, 'X3Y4')
        self.assertEqual(comp.Molwt, 0.0)
        self.assertEqual(comp.atomdict, {'X':3,'Y':4})