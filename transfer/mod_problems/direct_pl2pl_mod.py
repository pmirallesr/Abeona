from pykep.trajopt._direct import _direct_base
import pykep as pk
import numpy as np
from utils.pow_to_mass import pow_to_mass as pw2m

defaults = {
    "w_mass": 0.5, "w_tof":0.5,
    "prop_eff": 0.5}

class direct_pl2pl_mod(_direct_base):
    """Represents a direct transcription transfer between solar system planets.

    This problem works by manipulating the starting epoch t0, the transfer time T the final mass mf and the controls 
    The dicision vector is::

        z = [t0, T, mf, Vxi, Vyi, Vzi, Vxf, Vyf, Vzf, controls]
    """

    def __init__(self,
                 p0="earth",
                 pf="mars",
                 mass=1000,
                 thrust=0.3,
                 isp=3000,
                 power=None,
                 nseg=20,
                 t0=[500, 1000],
                 tof=[200, 500],
                 vinf_dep=1e-3,
                 vinf_arr=1e-3,
                 hf=False,
                 **kwargs):
        """Initialises a direct transcription orbit to orbit problem.

        Args:
            - p0 (``str``): Departure planet name. (will be used to construct a planet.jpl_lp object)
            - pf (``str``): Arrival planet name. (will be used to construct a planet.jpl_lp object)
            - mass (``float``, ``int``): Spacecraft wet mass [kg].
            - thrust (``float``, ``int``): Spacecraft maximum thrust [N].
            - isp (``float``, ``int``): Spacecraft specific impulse [s].
            - nseg (``int``): Number of colocation nodes.
            - t0 (``list``): Launch epochs bounds [mjd2000].
            - tof (``list``): Transfer time bounds [days].
            - vinf_dep (``float``): allowed launch DV [km/s] 
            - vinf_arr (``float``): allowed arrival DV [km/s]
            - hf (``bool``): High-fidelity. Activates a continuous representation for the thrust.
        """
        # Init args
        self.args = defaults
        for arg in self.args:
            if arg in kwargs:
                self.args[arg] = kwargs[arg]
        # Init optim weights
        self.w_mass = self.args["w_mass"]
        self.w_tof = self.args["w_tof"]
        # Init power
        if power:
            self.power = power
        else:
            self.power = 0.5*thrust*isp/self.args["prop_eff"]
        # initialise base
        _direct_base.__init__(self, mass, thrust, isp, nseg, pk.MU_SUN, hf)

        # planets
        if all([isinstance(pl, str) for pl in [p0, pf]]):
            self.p0 = pk.planet.jpl_lp(p0)
            self.pf = pk.planet.jpl_lp(pf)
        else:
            raise TypeError("Planet names must be supplied as str.")

        # bounds 
        assert t0[1] - t0[0] >= tof[0]
        assert all(t > 0 for t in tof)
        assert tof[1] > tof[0]
        self.t0 = t0
        self.tof = tof

        # boundary conditions on velocity
        self.vinf_dep = vinf_dep * 1000  # (in m)
        self.vinf_arr = vinf_arr * 1000  # (in m)

        # The class is built around solar system planets hence mu is always the
        # SUN
        self.mu = pk.MU_SUN

    def fitness(self, z):
        # z = t0, tof, mf, [vinf_0], [vinf_f], [u]
        # epochs (mjd2000)
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # final mass
        mf = z[2]

        # controls: 60 element vector, probably containing ux, uy, uz for each segment
        u = z[9:]

        # compute Cartesian states of planets
        r0, v0 = self.p0.eph(t0)
        rf, vf = self.pf.eph(tf)

        # add the vinfs from the chromosome
        v0 = [a + b for a, b in zip(v0, z[3:6])]
        vf = [a + b for a, b in zip(vf, z[6:9])]

        # spacecraft states
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        xf = pk.sims_flanagan.sc_state(rf, vf, mf)

        # set leg
        self.leg.set(t0, x0, u, tf, xf)

        # compute equality constraints
        ceq = np.asarray(self.leg.mismatch_constraints(), np.float64)

        # nondimensionalise equality constraints
        ceq[0:3] /= pk.AU
        ceq[3:6] /= pk.EARTH_VELOCITY
        ceq[6] /= self.sc.mass

        # compute inequality constraints
        cineq = np.asarray(self.leg.throttles_constraints(), np.float64)
        
        
        # compute inequality constraints on departure and arrival velocities
        v_dep_con = (z[3] ** 2 + z[4] ** 2 + z[5] ** 2 - self.vinf_dep ** 2)
        v_arr_con = (z[6] ** 2 + z[7] ** 2 + z[8] ** 2 - self.vinf_arr ** 2)

        # nondimensionalize inequality constraints
        v_dep_con /= pk.EARTH_VELOCITY ** 2
        v_arr_con /= pk.EARTH_VELOCITY ** 2

        return np.hstack(([self.obj_func(z)], ceq, cineq, [v_dep_con, v_arr_con]))

    def obj_func(self, z):
        # z = t0, tof, mf, [vinf_0], [vinf_f], [u]
        tof, mf = z[1:3]
        tf = tof + z[0]
        mpow = pw2m(self.power, tf, planet='mars')
        mt = mf + mpow
        # Convert from power to general mass
        return -(self.w_mass*mt + self.w_tof*tof)
    
    def get_nic(self):
        return super().get_nic() + 2

    def get_bounds(self):
        lb = [self.t0[0], self.tof[0], self.sc.mass * 0.1] + \
            [-self.vinf_dep] * 3 + [-self.vinf_arr] * 3 + \
            [-1, -1, -1] * self.nseg
        ub = [self.t0[1], self.tof[1], self.sc.mass] + \
            [self.vinf_dep] * 3 + [self.vinf_arr] * 3 + \
            [1, 1, 1] * self.nseg
        return (lb, ub)

    def _plot_traj(self, z, axis, units):

        # times
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # plot Keplerian
        pk.orbit_plots.plot_planet(
            self.p0, t0, units=units, color=(0.8, 0.8, 0.8), axes=axis)
        pk.orbit_plots.plot_planet(
            self.pf, tf, units=units, color=(0.8, 0.8, 0.8), axes=axis)

    def _pretty(self, z):
        print("\nLow-thrust NEP transfer from " +
              self.p0.name + " to " + self.pf.name)
        print("\nLaunch epoch: {!r} MJD2000, a.k.a. {!r}".format(
            z[0], pk.epoch(z[0])))
        print("Arrival epoch: {!r} MJD2000, a.k.a. {!r}".format(
            z[0] + z[1], pk.epoch(z[0] + z[1])))
        print("Time of flight (days): {!r} ".format(z[1]))
        print("\nLaunch DV (km/s) {!r} - [{!r},{!r},{!r}]".format(np.sqrt(
            z[3]**2 + z[4]**2 + z[5]**2) / 1000, z[3] / 1000, z[4] / 1000, z[5] / 1000))
        print("Arrival DV (km/s) {!r} - [{!r},{!r},{!r}]".format(np.sqrt(
            z[6]**2 + z[7]**2 + z[8]**2) / 1000, z[6] / 1000, z[7] / 1000, z[8] / 1000))

    @staticmethod
    def _get_controls(z):
        return z[9:]