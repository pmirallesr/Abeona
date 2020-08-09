# Abeona
Design of low thrust transfers from Earth to Mars for SGAC's Small Satellite Project Group Mars Constellation

Folder structure:
- phasing, transfer, spiralling: Put files regarding these problems inside their respective folders
- mod_problems: There's one onder phasing, transfer, and spiralling. Each of these hosts modified versions of pykep problems should you create any. e.g. direct_pl2pl_mod.py is an (unfinished) modification of pykep's direct_pl2pl algo that changes the objective function and adds a few constraints
- Runs: There's one onder phasing, transfer, and spiralling. Use them to store simulation outputs. Keep it tidy!
- data: Hosts diverse data that might be useful. Right now only hosts a yaml file with EP thruster data
- utils: Hosts tools that can be useful across the repo. date_utils include tools for transforming from the mjd2000 format to normal human-readable dates and back.
