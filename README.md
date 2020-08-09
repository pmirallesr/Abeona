# Abeona
Design of low thrust transfers from Earth to Mars for SGAC's Small Satellite Project Group Mars Constellation

## Installation

Go the folder where you want your project to be i.e. ~/Documents/MarsStudy and run:

	```
	git clone https://github.com/pmirallesr/Abeona
	cd Abeona
	conda env create -f environment.yaml
	conda activate abeona
	python -m transfer.pl2pl_optim --help
	```
	
You should expect the following output:
	
	```
	/home/pmirallesr/Documents/MarsStudy/Abeona/transfer/pl2pl_optim.py:37: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
	  yaml_content = yaml.load(yaml_file)
	usage: pl2pl_optim.py [-h] [--log_level LOG_LEVEL] [--mass MASS] [--thrust THRUST] [--isp ISP] [--model MODEL] [--mode MODE] [--list_modes] [--t0 T0 T0] [--tof TOF TOF]
	                      [--vinf_dep VINF_DEP] [--vinf_arr VINF_ARR] [--nseg NSEG]
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --log_level LOG_LEVEL
	                        Set to off, info, or debug to control verbosity
	
	Spacecraft options:
	  --mass MASS           Wet mass of the spacecraft upon TMI
	
	Engine options:
	  --thrust THRUST       Thrust during transfer
	  --isp ISP             Specific impulse during transfer
	  --model MODEL         Select a model amongst: ['BHT-1500']. Overrides thrust and isp settings
	  --mode MODE           Engine mode during tranfer. Necessary only if a model was selected. 0 is the default mode
	  --list_modes          List available modes for the selected model. Will do that instead of running program
	
	Trajectory options:
	  --t0 T0 T0            Start and end bound dates for the start of the trajectory. The optimization explores dates lying between the two.
	  --tof TOF TOF         Start and end bound dates for the time of flight. The optimizationexplores times of flight lying between the two.
	  --vinf_dep VINF_DEP   Allowed v infinity for Earth departure
	  --vinf_arr VINF_ARR   Allowed v infinity for Mars arrival
	
	Optimizer options:
	  --nseg NSEG           Number of calculation segments
	```
	
## Folder structure:
- phasing, transfer, spiralling: Put files regarding these problems inside their respective folders
- mod_problems: There's one under phasing, transfer, and spiralling. Each of these hosts modified versions of pykep problems should you create any. e.g. direct_pl2pl_mod.py is an (unfinished) modification of pykep's direct_pl2pl algo that changes the objective function and adds a few constraints
- Runs: There's one under phasing, transfer, and spiralling. Use them to store simulation outputs. Keep it tidy!
- data: Hosts diverse data that might be useful. Right now only hosts a yaml file with EP thruster data
- utils: Hosts tools that can be useful across the repo. date_utils include tools for transforming from the mjd2000 format to normal human-readable dates and back.
