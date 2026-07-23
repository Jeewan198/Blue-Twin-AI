import pandas as pd
import numpy as np

try:
    import deepxde as dde
except ImportError:
    dde = None
from scipy.interpolate import interp1d


class REDPhysicsEngine:
    """
    Physics engine for Blue-Twin AI modeling Reverse Electrodialysis (RED) dynamics.
    Incorporates Nernst-Planck-Poisson transport and thermodynamic properties.
    """

    def __init__(self, csv_path='ARA24_Clean_Master_Enhanced.csv', D=1.0e-9, T=298.15):
        self.csv_path = csv_path
        self.D = D
        self.T = T
        self.F = 96485.33  # Faraday constant (C/mol)[cite: 1]
        self.R = 8.314  # Universal gas constant (J/(mol·K))[cite: 1]
        self.thermal_voltage = (self.R * T) / self.F

        # Enforce Einstein Relation: mu = (z * F * D) / (R * T), assuming z=1[cite: 1]
        self.mu = (1.0 * self.F * self.D) / (self.R * self.T)

        self.interp_funcs = {}
        self.load_ara24_data()

    def load_ara24_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            df.columns = df.columns.str.strip()
            months = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
            month_cols = [f"Theoretical_MW_{m}" for m in months]
            month_indices = np.linspace(0, 11, 12)

            for _, row in df.iterrows():
                river_id = str(row['River ID'])
                potentials = pd.to_numeric(row[month_cols], errors='coerce').values
                if not np.isnan(potentials).any():
                    self.interp_funcs[river_id] = interp1d(month_indices, potentials, kind='cubic')
            print(f"✅ Successfully initialized interpolation for {len(self.interp_funcs)} rivers.")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def residual_npp(self, x, y):
        """
        Computes the corrected Nernst-Planck-Poisson residuals.
        Includes cross-term C * d2Phi/dx2 and physical constants D, mu.[cite: 1]
        """
        if dde is None:
            raise ImportError("DeepXDE is required for residual_npp computation.")
        C = y[:, 0:1]
        Phi = y[:, 1:2]

        dC_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dPhi_dx = dde.grad.jacobian(y, x, i=1, j=0)

        d2C_dx2 = dde.grad.hessian(y, x, component=0, i=0, j=0)
        d2Phi_dx2 = dde.grad.hessian(y, x, component=1, i=0, j=0)

        res_c = (self.D * d2C_dx2) + self.mu * ((dC_dx * dPhi_dx) + (C * d2Phi_dx2))
        res_phi = d2Phi_dx2 + C

        return [res_c, res_phi]

    def nernst_potential(self, c_high, c_low, alpha=1.0, z=1.0):
        """
        Calculates E_cell = alpha * (RT/zF) * ln(c_high / c_low).
        """
        return alpha * (self.thermal_voltage / z) * np.log(c_high / c_low)

    def internal_resistance(self, concentration, thickness=0.001, z=1.0):
        """
        Models internal cell-pair resistance (R_lc) in Ω·cm².
        """
        conductivity = z * self.F * self.mu * concentration
        resistance_m2 = thickness / conductivity
        return resistance_m2 * 10000

    def get_data_at_time(self, river_id, t):
        return self.interp_funcs[river_id](t) if river_id in self.interp_funcs else None

    def get_data_by_date(self, river_id, day_of_year):
        t = ((day_of_year - 1) / 365) * 11
        return self.get_data_at_time(river_id, t)

    def get_system_state(self, concentration, potential):
        norm_c = concentration / 1000.0
        norm_phi = potential / 500.0
        return np.array([norm_c, norm_phi, self.T / 300.0], dtype=np.float32)


if __name__ == "__main__":
    engine = REDPhysicsEngine()
    print("\n--- Blue-Twin AI Physics Engine Sanity Check ---")
    print(f"Diffusion Coefficient (D): {engine.D} m²/s")
    print(f"Derived Ionic Mobility (mu): {engine.mu:.2e} m²/(V·s)")
    print(f"Thermal Voltage (RT/F): {engine.thermal_voltage * 1000:.2f} mV")