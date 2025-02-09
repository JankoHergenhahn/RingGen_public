import numpy as np
from copy import deepcopy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import importlib

from .Topology import TopFile

class Minimizer:
    """
    Performs MM energy minimization on coordinates using topology.
    
    Attributes
    ----------
    topology: RingGen.Topology
        contains force field information
    coordinates: RingGen.Coordinates
        coordinates need to match order of topology        
    constrained_atoms: list of int
        atoms whose positions are constrained during minimizations
    transparent_atoms: list of int
        atoms for which non-bonded parameters are set to 0
    minimizer: openMM.Simulation object
        only created once minimize method is called
    """
    def __init__(self, topology, coordinates, 
                 constrained_atoms = [], transparent_atoms = []):
        self.topology = topology
        self.coordinates = coordinates
        self.constrained_atoms = constrained_atoms
        self.transparent_atoms = transparent_atoms
        self.minimizer = None
        
    def minimize(self, steps = 1000):
        """
        Performs geometry optimization, max number of iterations = steps(int).
        """
        #create temp files
        self.coordinates.write_pdb('temp.pdb', topology=self.topology)
        top_temp = deepcopy(self.topology)
        top_temp = self.set_constrained_atoms(top_temp, self.constrained_atoms)
        top_temp = self.set_transparent_atoms(top_temp, self.transparent_atoms)
        top_file = TopFile().create_file(top_temp)
        top_file.set_atom_types(self.get_default_atom_types())
        top_file.write_file("temp.top")
        
        #set up
        pdb = PDBFile('temp.pdb')
        top = GromacsTopFile('temp.top')
        system = top.createSystem(nonbondedMethod=NoCutoff, 
                                  nonbondedCutoff=1*nanometer,
                                  constraints=None)
        integrator = LangevinIntegrator(300*kelvin, 
                                        1/picosecond, 
                                        0.002*picoseconds)
        #minimization
        self.minimizer = Simulation(top.topology, system, integrator)
        self.minimizer.context.setPositions(pdb.positions)
        self.minimizer.minimizeEnergy(
            maxIterations=steps,
            tolerance=Quantity(value=0.1, unit=kilojoule/mole))
        
    def get_coordinates(self):
        return np.array(self.minimizer.context.getState(getPositions=True).getPositions(asNumpy=True) * 10) #convert nm to Ang
    
    
    @staticmethod
    def set_constrained_atoms(topology, constrained_atoms):
        """
        Sets masses of constrained atoms to 0. Handled as constraint by openMM.
        """
        for i,atom in enumerate(topology.atoms):
            if i in constrained_atoms:
                topology.atoms[i][7] = 0.0
        return topology
    
    @staticmethod
    def set_transparent_atoms(topology, transparent_atoms):
        """
        Sets atom_types of transparent atoms to "XX" with LJ_eps = 0.
        """
        for i,atom in enumerate(topology.atoms):
            if i in transparent_atoms:
                topology.atoms[i][1] = "XX"
                topology.atoms[i][6] = 0.0
        return topology
    
    @staticmethod
    def get_default_atom_types():
        """
        Obtains atom types from default location top parse to topology file.
        """
        atom_types = []
        with importlib.resources.open_text('RingGen.DefaultFiles',
                                           'default_atom_types.txt') as f:
            for line in f:
                atom_types.append(line.strip('\n'))
        return atom_types