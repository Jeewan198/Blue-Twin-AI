import pandas as pd
import numpy as np
import deepxde as dde
from scipy.interpolate import interp1d


class REDPhysicsEngine:
    def __init__(self, csv_path, D=1.0e-9, mu=1.0, T=298.15):
        """
        Initializes the physics engine with transport parameters and temperature.
        D: Diffusion coefficient
        mu: Ionic mobility
        T: Temperature in Kelvin (for thermodynamic consistency)
        """
        self.csv_path = csv_path
        self.D = D
        self.mu = mu
        self.T = T
        self.thermal_voltage = (1.38e-23 * T) / 1.602e-19  # kB * T / e

        self.data = None
        self.interp_funcs = {}
        self.load_ara24_data()

    def load_ara24_data(self):
        """Loads dataset and creates temporal interpolation functions."""
        try:
            self.data = pd.read_csv(self.csv_path)
            months = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
            month_indices = np.linspace(0, 11, 12)

            for _, row in self.data.iterrows():
                river_id = str(row['River ID'])
                potentials = [row[f"{m}_Potential_MW"] for m in months]
                self.interp_funcs[river_id] = interp1d(month_indices, potentials, kind='cubic')
            print(f"✅ Successfully initialized interpolation for {len(self.interp_funcs)} rivers.")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def get_data_at_time(self, river_id, t):
        """Returns interpolated potential for a given river at time t (0-11)."""
        return self.interp_funcs[river_id](t) if river_id in self.interp_funcs else None

    def get_data_by_date(self, river_id, day_of_year):
        """Helper to map calendar day (1-365) to time index (0-11)."""
        t = ((day_of_year - 1) / 365) * 11
        return self.get_data_at_time(river_id, t)

    def residual_npp(self, x, y):
        """
        Computes the temperature-dependent Nernst-Planck-Poisson residuals.
        x: spatial coordinates
        y: network outputs [Concentration, Potential]
        """
        C = y[:, 0:1]
        Phi = y[:, 1:2]

        # Compute derivatives
        dC_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dPhi_dx = dde.grad.jacobian(y, x, i=1, j=0)
        d2C_dx2 = dde.grad.hessian(y, x, i=0, j=0)
        d2Phi_dx2 = dde.grad.hessian(y, x, i=1, j=0)

        # NPP Residuals with thermal voltage coupling
        res_c = d2C_dx2 + (dC_dx * dPhi_dx / self.thermal_voltage)
        res_phi = d2Phi_dx2 + C

        return [res_c, res_phi]

    def get_system_state(self, concentration, potential):
        """
        Returns a normalized vector for the RL agent's observation space.
        """
        # Simple normalization: values scaled against engine parameters
        norm_c = concentration / 1000.0
        norm_phi = potential / 500.0
        return np.array([norm_c, norm_phi, self.T / 300.0])