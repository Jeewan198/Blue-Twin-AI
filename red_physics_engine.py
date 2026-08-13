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
        self.F = 96485.33  # Faraday constant (C/mol)
        self.R = 8.314      # Universal gas constant (J/(mol*K))
        self.thermal_voltage = (self.R * T) / self.F

        # Enforce Einstein Relation: mu = (z * F * D) / (R * T), assuming z=1
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

    def residual_npp(self, x, y):
        """
        Computes the Nernst-Planck-Poisson residuals for PINN training.
        Includes cross-term C * d2Phi/dx2 and physical constants D, mu.
        NOTE: not currently used in the RL training pipeline (project scope
        decision: RL is trained directly against this analytic physics engine,
        not against a trained PINN -- see Design chapter for rationale). Kept
        here as it was already implemented and may be useful for future work.
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
        alpha=1.0 assumes ideal membrane selectivity (documented simplifying
        assumption -- see Design chapter sensitivity analysis for the effect
        of realistic alpha < 1.0).
        """
        return alpha * (self.thermal_voltage / z) * np.log(c_high / c_low)

    def internal_resistance(self, concentration, thickness=0.0002, z=1.0, r_mem=5.6):
        """
        Models internal cell-pair resistance in Ohm*cm^2, following Literature
        Review Equation 2.2: R_stack = R_mem + R_el + R_lc.

        R_lc (this compartment's own ionic resistance) is calculated from
        conductivity. thickness=0.0002 m (200 um) reflects real RED spacer
        thickness reported in the literature (previously 0.001 m / 1mm, a
        generic electrodialysis approximation not specific to RED).

        R_mem (combined cation + anion exchange membrane resistance) is added
        as a real, literature-grounded constant: commercial NaCl-compatible
        ion-exchange membranes commonly report ~2.8 Ohm*cm^2 each for CEM and
        AEM (~5.6 Ohm*cm^2 combined per cell pair).

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


if __name__ == "__main__":
    engine = REDPhysicsEngine()
    print("\n--- Blue-Twin AI Physics Engine Sanity Check ---")
    print(f"Diffusion Coefficient (D): {engine.D} m^2/s")
    print(f"Derived Ionic Mobility (mu): {engine.mu:.2e} m^2/(V*s)")
    print(f"Thermal Voltage (RT/F): {engine.thermal_voltage * 1000:.2f} mV")

    # REDstack Afsluitdijk plant real concentrations (see internal_resistance docstring)
    c_sea, c_river = 479.0, 6.0
    v_nernst = engine.nernst_potential(c_sea, c_river)
    print(f"\nNernst Potential (REDstack real concentrations): {v_nernst * 1000:.2f} mV")

    r_dilute = engine.internal_resistance(concentration=c_river)
    r_concentrated = engine.internal_resistance(concentration=c_sea)
    print(f"Internal Resistance - Dilute Compartment (C={c_river}): {r_dilute:.2f} Ohm*cm^2")
    print(f"Internal Resistance - Concentrated Compartment (C={c_sea}): {r_concentrated:.2f} Ohm*cm^2")
    print(f"Ratio (Dilute / Concentrated): {r_dilute / r_concentrated:.1f}x")

    power_density_W_cm2 = (v_nernst ** 2) / r_dilute
    power_density_W_m2 = power_density_W_cm2 * 10000
    print(f"\nPredicted power density: {power_density_W_m2:.3f} W/m^2")
    print("Real REDstack measured (Tedesco et al., 2022): gross ~0.35 W/m^2, "
          "net ~0.10-0.25 W/m^2")
