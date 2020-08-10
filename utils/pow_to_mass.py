import pykep as pk
import numpy as np

SOLAR_CONSTANT = 1379 # Mean solar irradiance at mars
POWER_ANCILLARY = 0.02 # Mass of ancillary equipment as % of total power provided (0.02 kg per WATT).
avg_eff_loss_year = 0.025 # Efficiency loss on panels from one year of ageing
min_panel_mass = 45.636 # Minimum panel mass stemming from other requirements
inherent_degradation = 0.23 # Inherent loss of efficiency from construction losses
packaging_loss = 0.1 # Losses from imperfect packaging
surface_density = 5.06 # kg/m2

def pow_to_mass(power, t, d=None, planet='mars'):
    """
    Given a certain power, a time, and a distance, returns the mass of the associated power systems.
    If a planet name is passed, the distance of said planet to the sun at that time will be calculated.
    Electric storage is considered fixed and included in the original SC mass
        power: double, required power in W
        x: double, distance to sun in AU 
        t: Time in mjd2000 
        planet: Planet being orbited. Defaults to mars
     
    """
    m_ancillary = POWER_ANCILLARY*power
    if d:
        return m_ancillary + pow_to_panel_mass(power, d, t) # 
    else:
        planet = pk.planet.jpl_lp(planet)
        x = planet.eph(t)[0]
        d = np.linalg.norm(x)/pk.AU
        return m_ancillary + pow_to_panel_mass(power, d, t) # 


def pow_to_panel_mass(power, x, t):
    '''
    Returns the amount of solar panel mass required to supply power p at time t
    Args:
        power: Required power
        t: date in mjd2000
        planet: pykep planet
    '''
    ageing_eff_loss = 1 - (1-avg_eff_loss_year)**t # Loss of avg_eff_loss every year
    total_eff = 1*(1-ageing_eff_loss)*(1-inherent_degradation)*(1-packaging_loss)
    pflux_t = SOLAR_CONSTANT/(x**2)
    eff_pflux_t = pflux_t*total_eff
    panel_area = power/eff_pflux_t
    panel_mass = panel_area*surface_density
    return max(min_panel_mass, panel_mass)

    