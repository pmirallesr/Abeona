import argparse
import logging
import pykep
import pykep.trajopt as tropt
from pykep.examples import add_gradient, algo_factory
import yaml
import time
import pygmo as pg
import math
import matplotlib.pyplot as plt
from utils.date_utils import mjd2000_to_date
import transfer.mod_problems.direct_pl2pl_mod as tropt_mod
import os
import numpy as np

months = {1:"January",
          2:"February",
          3:"March",
          4:"April",
          5:"May",
          6:"June",
          7:"July",
          8:"August",
          9:"September",
          10:"October",
          11:"November",
          12:"December"}

dep = "earth"
tgt = "mars"

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

possible_models = ['BHT-1500']
MAX_MASSF_DISCREPANCY = 0.1 # If the final mass encountered is within less than x per unit of a given final mass, we discard the iteration

def m_ratio_to_dv(m_ratio, isp=1915):
    return math.log(m_ratio)*pykep.G0*isp

def create_dir(output):
    try:
        os.makedirs(output)
        return output
    except:
        output += "bis"
        return create_dir(output)
class Engine:
    def __init__(self, name, yaml_engine):
        self.name = name
        self.modes = [mode_str.split() for mode_str in yaml_engine["modes"]]
        # Clean modes from string to number
        for mode in self.modes:
            for i, spec in enumerate(mode):
                mode[i] = float(spec)
        self.throttle = yaml_engine["throttle"]["range"] if "throttle" in yaml_engine else None
    def __str__(self):
        modes_str = ''
        if self.throttle:
            throttle_str = ('Throttle range: ' + str(self.throttle[0]*100) + '% - ' + str(self.throttle[1]*100) + '%')
        else:
            throttle_str = 'Not throttable'
        for i, mode in enumerate(self.modes):
            modes_str += f"Mode {i:2d} => Power: {mode[0]:5.0f} W, Thrust: {mode[1]:0.3f} N, Isp: {mode[2]:4.0f} s \n" 
        return (f"{self.name}, {throttle_str}. \n"
                f"Operation modes \n"
                f"{modes_str} ")

def load_engines(yaml_path):
    engines = {}
    with open(yaml_path, "r") as yaml_file:
        yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)
        for engine, specs in yaml_content.items():
            engines[engine] = Engine(engine, specs)
    return engines
        
def print_args(args, et_models):
    if args.model:
        engine = et_models[args.model]
        specs = engine.modes[args.mode]
        engine_str = f"{args.model}, at Power: {specs[0]:.0f} W, Thrust: {specs[1]:.3f} N, Isp: {specs[2]:.0f} s" 
    else:
        p = args.thrust*args.isp*9.81*0.5/0.5 #50% efficiency
        engine_str = f"Power: {p:.0f}W, Thrust: {args.thrust:.3f} N, Specific Impulse: {args.isp:.0f} s"
    dep_date = mjd2000_to_date(args.t0[0])
    arr_date = mjd2000_to_date(args.t0[1])
    
    print(f"-"*120 + "\n" + f"-"*120 + "\n" +
          f"Spacecraft wet mass: {args.mass:.0f} kg \n"
          f"Engine: {engine_str} \n"
          f"Dearture: {args.vinf_dep} km/s on {math.floor(dep_date[2])} of {months[dep_date[1]]}, {dep_date[0]} \n"
          f"Arrival: {args.vinf_arr} km/s on {math.floor(arr_date[2])} of {months[arr_date[1]]}, {arr_date[0]} \n"
          f"Time of flight from {args.tof[0]} to {args.tof[1]} days"
          + "\n" + f"-"*120 + "\n" + f"-"*120)
        
def main(args):
    if args.model:
        logger.debug(f"Models available: {et_models}")
        engine = et_models[args.model]
        et_mode = engine.modes[args.mode]
        logger.debug(f"You've chosen {args.model} at mode {et_mode}")
        _, args.thrust, args.isp = et_mode
    
    run_results = {}
    for n in range(args.n_runs):
        logger.info(f"Launching run {n}")
        og_output = args.output
        args.output += f"/run{n}"
        os.makedirs(args.output, exist_ok=True)
        done = False
        c=0 # Counter of discarded runs
        while not done:
            algo = algo_factory("slsqp")
            earth_to_mars = add_gradient(tropt_mod.direct_pl2pl_mod(p0=dep,
                                               pf=tgt,
                                               mass=args.mass,
                                               thrust=args.thrust,
                                               isp=args.isp,
                                               nseg=args.nseg,
                                               t0=args.t0,
                                               tof=args.tof,
                                               vinf_dep=args.vinf_dep,
                                               vinf_arr=args.vinf_arr,
                                               hf=False,
                                               w_mass=args.w_mass,
                                               w_tof=(1-args.w_mass)),
                                        with_grad=True)
            prob = pg.problem(earth_to_mars)
            prob.c_tol = [1e-5] * prob.get_nc()
            pop = pg.population(prob, 1)
            start = time.time()
            pop = algo.evolve(pop)
            logger.info(f"Time elapsed: {(time.time() - start):.3f}")
            if prob.feasibility_x(pop.champion_x):
                logger.info("OPTIMAL FOUND!")
                done = True
            else:
                logger.info("No solution found, trying again")
                c+=1
                logger.info(f"{c} runs discarded")
                continue
            results = earth_to_mars.udp_inner.pretty(pop.champion_x)
            
            tof, mf = pop.champion_x[1:3]
            #If the difference is too large, repeat. We only care about cases were mf < final_mass
            if args.final_mass - mf >= MAX_MASSF_DISCREPANCY*args.final_mass: 
                logger.info(f"-"*60 + "\n"
                            f"Run discarded due to large mass discrepancy. \n"
                            f"Initial mass: {args.mass} \n"
                            f"Final mass obtained: {mf}. Specified final mass: {args.final_mass}\n"
                            + results + f"-"*60)
                c+=1
                logger.info(f"{c} runs discarded")
                done = False
            else:
                true_initial_mass = args.final_mass*args.mass/mf
                logger.info(results)
                run_results[n] = (tof, true_initial_mass)
        f = open(args.output + "/solution.txt", 'w')
        f.write(results)
        f.write("-"*100)
        f.write(str(pop.champion_x))
        f.close()
        earth_to_mars.udp_inner.plot_traj(pop.champion_x)
        plt.title("The trajectory in the heliocentric frame")
        plt.savefig(args.output + "/trajectory.png")
        earth_to_mars.udp_inner.plot_control(pop.champion_x)
        plt.title("The control profile (throttle)")
        plt.savefig(args.output + "/throttle.png")
#         plt.show()
        
        args.output = og_output
        
       
    run_results_str = " run, tof, wet mass"
    run_results_list = [f"\n {n}, {tof}, {m0}" for n, (tof, m0) in run_results.items()]
    run_results_str += "".join(run_results_list)
    f = open(args.output + "/run_results.txt", 'w')
    f.write(run_results_str)
    results_array = np.array(list(run_results.values()))
    tof_avg, m0_avg = results_array.mean(0)
    f.write(f"\n mean {tof_avg}, {m0_avg}")
    tof_std, m0_std = results_array.std(0)
    f.write(f"\n std {tof_std}, {m0_std}")
    f.close()
        
if __name__ == "__main__":
    
    et_models = load_engines("data/engines.yaml")
    
    logger = logging.getLogger()
    # PROGRAM
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_level", default="info", type=str,
        help="Set to off, info, or debug to control verbosity"
    )
    parser.add_argument(
        "--output", default=f"output/{time.strftime('%m%d-%H%M', time.localtime())}", type=str,
        help="Output folder"
    )
    # SPACECRAFT
    spacecraft_options = parser.add_argument_group("Spacecraft options")
    spacecraft_options.add_argument("--mass", type=float, default=1000,
                        help="Wet mass of the spacecraft upon TMI")
    spacecraft_options.add_argument("--final_mass", type=float, default=425,
                        help="Final mass of the spacecraft after transfer")
    # ENGINE
    engine_options = parser.add_argument_group("Engine options")
    engine_options.add_argument("--thrust", type=float, default=0.102,
                    help="Thrust during transfer")
    engine_options.add_argument("--isp", type=float, default=1915,
                    help="Specific impulse during transfer")
    engine_options.add_argument("--model", default=None,
                    help=f"Select a model amongst: {list(et_models.keys())}. Overrides thrust and isp settings")
    engine_options.add_argument("--mode", type=int, default=0,
                    help="Engine mode during tranfer. Necessary only if a model was selected. 0 is the default mode")
    engine_options.add_argument("--list_modes", action='store_true',
                    help="List available modes for the selected model. Will do that instead of running program")
    # TRAJECTORY
    trajectory_options = parser.add_argument_group("Trajectory options")
    trajectory_options.add_argument("--t0", nargs=2, type=int, default=[9497, 10000],
                                    help="Start and end bound dates for the start of the trajectory." +
                                    " The optimization explores dates lying between the two.")
    trajectory_options.add_argument("--tof", nargs=2, type=int, default=[200, 500],
                                    help="Start and end bound dates for the time of flight. The optimization"
                                    + "explores times of flight lying between the two.")
    trajectory_options.add_argument("--vinf_dep", type=float, default=1e-3,
                                    help="Allowed v infinity for Earth departure")
    trajectory_options.add_argument("--vinf_arr", type=float, default=1e-3,
                                    help="Allowed v infinity for Mars arrival")
    # Optimization
    optimizer_options = parser.add_argument_group("Optimizer options")
    optimizer_options.add_argument("--nseg", type=int, default=20,
                                    help="Number of calculation segments")
    optimizer_options.add_argument("--n_runs", type=int, default=4,
                                    help="Number of simulations to be repeated")
    optimizer_options.add_argument("--w_mass", type=float, default=0.5,
                                    help="Weight of the mass objective relative to the total optimization. \n"\
                                    +"The weight of the time of flight is 1 - w_mass. Default 0.5")
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.setLevel(log_level)
    
    if args.list_modes:
        print(et_models[args.model])
        exit()
    
    args.output = create_dir(args.output)
    f = open(args.output + "/args.txt", 'w')
    f.write(str(vars(args)))
    f.close()

    main(args)
    print_args(args, et_models)
    

            
    
    
    