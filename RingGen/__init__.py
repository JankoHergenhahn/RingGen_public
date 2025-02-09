"""RingGen Module

This module has been designed to quickly generate coordinates for complex 
cyclic structures. Structures are minimized by molecular mechanics calculations
and can be used as starting points for higher level calculations or molecular
dynamics simulations. Coordinates are created based on a defined set of rules 
and topologies are generated based on the GAFF force field and AM1-BCC charges.
"""

from .ComplexStructure import ComplexStructure
from .Coordinates import Coordinates
from .Defaults import Defaults
from .DockingStructure import DockingStructure
from .PorphyrinStructure import PorphyrinStructure as prePorphyrinStructure
from .Structure import Structure
from .Tools import *
from .Topology import TopFile as TopFile
from .Topology import Topology
from .TopologyGenerator import TopologyGenerator
try:
    from .Minimizer import Minimizer
except:
    print("OpenMM missing, fall back to simplified structure optimizer.")
    from .Minimizer_SD import Minimizer  

# Default topology fragments can be customized, they are loaded here and 
# added to the prePorphyrinStructure so it doesn't have to be done manually 
# when using the programme.
# Existing defaults are for porphyrinstructures
defaults = Defaults()
class PorphyrinStructure(prePorphyrinStructure):
    def __init__(self, name = ""):
        super().__init__()
        self.set_defaults(defaults)