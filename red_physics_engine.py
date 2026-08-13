import pandas as pd
import numpy as np
try:
    import deepxde as dde
except ImportError:
    dde = None
from scipy.interpolate import interp1d

class REDPhysicsEngine:
    def __init__(self, csv_path='ARA24_Clean_Master_Enhanced.csv', D=1.0e-9, T=298.15):
        self.csv_path = csv_path
        self.D = D
        self.T = T
        self.F = 96485.33
        self.R = 8.314
        self.thermal_voltage = (self.R * T) / self.F
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
            print(f"Successfully initialized interpolation for {len(self.interp_funcs)} rivers.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def nernst_potential(self, c_high, c_low, alpha=1.0, z=1.0):
        return alpha * (self.thermal_voltage / z) * np.log(c_high / c_low)

    def internal_resistance(self, concentration, thickness=0.0002, z=1.0, r_mem=5.6):
        """
        Models internal cell-pair resistance in Ohm*cm^2, following Literature
        Review Equation 2.2: R_stack = R_mem + R_el + R_lc.

        R_lc (this compartment's own ionic resistance) is calculated from
        conductivity as before. R_mem (combined cation + anion exchange
        membrane resistance) is now added as a real, literature-grounded
        constant: commercial NaCl-compatible ion-exchange membranes commonly
        report ~2.8 Ohm*cm^2 each for CEM and AEM (~5.6 Ohm*cm^2 combined per
        cell pair) -- consistent with this model's NaCl-based system.

        R_el (electrode resistance) is deliberately NOT included as a separate
        additive term. Electrode resistance is a fixed, whole-stack quantity
        (two electrodes regardless of stack size), so its contribution per
        cell pair shrinks as the number of cell pairs increases -- and is
        reported in the literature as "not predominant" for realistically-sized
        stacks (hundreds of cell pairs, matching real demonstration plants
        such as REDstack and the REAPower pilot). Since this model has no
        stack-size parameter to divide a fixed R_el across, adding an
        ungrounded per-cell-pair R_el value would be less defensible than
        omitting it with this documented justification.
        """
        conductivity = z * self.F * self.mu * concentration
        r_lc = (thickness / conductivity) * 10000
        return r_mem + r_lc

    def get_data_at_time(self, river_id, t):
        return self.interp_funcs[river_id](t) if river_id in self.interp_funcs else None

    def get_data_by_date(self, river_id, day_of_year):
        t = ((day_of_year - 1) / 365) * 11
        return self.get_data_at_time(river_id, t)

    def get_system_state(self, concentration, potential):
        norm_c = concentration / 1000.0
        norm_phi = potential / 500.0
        return np.array([norm_c, norm_phi, self.T / 300.0], dtype=np.float32)
