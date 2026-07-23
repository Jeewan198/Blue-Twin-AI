import unittest
import numpy as np
import torch
from red_physics_engine import REDPhysicsEngine


class TestREDPhysicsEngine(unittest.TestCase):
    def setUp(self):
        # Initialize engine with dummy/test data
        # Ensure 'ARA24_Clean_Master.csv' exists or provide a mock path
        self.engine = REDPhysicsEngine(csv_path='ARA24_Clean_Master_Enhanced.csv', D=1.0e-9, T=300)

    def test_initialization(self):
        self.assertEqual(self.engine.T, 300)
        self.assertIsNotNone(self.engine.thermal_voltage)

    def test_residual_npp_output_shape(self):
        # Create x that requires gradients
        x = torch.rand((10, 1), requires_grad=True)
        # Create y that is a function of x so that gradients exist
        # This simulates the PINN model output
        y = torch.cat([torch.sin(x), torch.cos(x)], axis=1)

        # Now DeepXDE/PyTorch can calculate the Jacobian because y depends on x
        residuals = self.engine.residual_npp(x, y)

        self.assertEqual(len(residuals), 2)
        self.assertEqual(residuals[0].shape, (10, 1))

    def test_get_system_state(self):
        state = self.engine.get_system_state(500.0, 250.0)
        self.assertEqual(len(state), 3)
        self.assertLessEqual(state[0], 1.0)  # Check normalization


if __name__ == '__main__':
    unittest.main()