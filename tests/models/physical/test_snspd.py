import unittest
from piel.types import PhysicalComponent
from piel.models.physical.opto_electronic import (
    physical_snspd,
    photonspot_snspd,
)  # Replace `your_module` with the actual module name


class TestPhysicalSNSPD(unittest.TestCase):
    def test_physical_snspd_default(self):
        """
        Test the default values of the physical_snspd function.
        """
        expected = PhysicalComponent(name="", model="SNSPD", manufacturer="")
        result = physical_snspd()
        self.assertEqual(result, expected)

    def test_physical_snspd_custom_name(self):
        """
        Test physical_snspd with a custom name.
        """
        expected = PhysicalComponent(name="CustomSNSPD", model="SNSPD", manufacturer="")
        result = physical_snspd(name="CustomSNSPD")
        self.assertEqual(result, expected)

    def test_physical_snspd_custom_all_fields(self):
        """
        Test physical_snspd with custom name, model, and manufacturer.
        """
        expected = PhysicalComponent(
            name="MySNSPD", model="AdvancedSNSPD", manufacturer="QuantumInc"
        )
        result = physical_snspd(
            name="MySNSPD", model="AdvancedSNSPD", manufacturer="QuantumInc"
        )
        self.assertEqual(result, expected)

    def test_photonspot_snspd_default(self):
        """
        Test the photonspot_snspd partial function with default values.
        """
        expected = PhysicalComponent(name="", model="SNSPD", manufacturer="Photonspot")
        result = photonspot_snspd()
        self.assertEqual(result, expected)

    def test_photonspot_snspd_custom_name(self):
        """
        Test the photonspot_snspd partial function with a custom name.
        """
        expected = PhysicalComponent(
            name="PhotonSNSPD", model="SNSPD", manufacturer="Photonspot"
        )
        result = photonspot_snspd(name="PhotonSNSPD")
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
