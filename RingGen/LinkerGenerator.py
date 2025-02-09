import numpy as np
from copy import deepcopy

from .ParaTools.ParameterDeterminator import ParameterDeterminator
from .ParaTools.RingTyper import RingTyper
from .ParaTools.GAFFpara import GAFFpara
from .TopologyGenerator import TopologyGenerator

class LinkerGenerator(TopologyGenerator):
    """
    Version of the TopologyGenerator class specialized to generate bonded 
    paramters to connect up disconnected topology fragments.
    """
    def generate_linker_topology(self, topology, connectivity, connection):
        """
        Generates bonded parameters involving the connecting atoms given in
        "connection".

        Parameters
        ----------
        topology : RingGen.Topology
        connectivity : NxN np.array
            Connectivity matrix.
        connection : list of int
            Atom indices that form the connecting bond.

        Returns
        -------
        new_topology : RingGen.Topology
            Edited version of the input topology, with additional parameters.
        """
        self.force_field = GAFFpara()
        atoms = self._find_linker_atoms(connectivity, connection)
        self.parameter_determinator = \
            self._find_parameter_indeces(atoms, topology)
        atom_types = self._determine_atom_types(
            topology, connectivity, atoms)
        self.atom_types = [atom_types[x]
                           if x in atom_types.keys()
                           else 0
                           for x,_ in enumerate(topology.atoms)]
        new_topology = self._build_topology(topology)
        
        return new_topology
    
    def _build_topology(self, topology):
        """
        Manages calls to generate bonded parameters.
        """
        new_topology = deepcopy(topology)
        new_topology = deepcopy(topology)        
        new_topology.bonds.extend(self._get_bonds())
        new_topology.angles.extend(self._get_angles())
        new_topology.dihedrals.extend(self._get_dihedrals())
        new_topology.dihedrals.extend(self._get_impropers())
        new_topology.pairs.extend(self._get_pairs())
        return new_topology
    
    
    def _find_linker_atoms(self, connectivity, connection):
        """
        Determines atoms that are involved in bonded interactions between the 
        disconnected fragments. Need multiple levels to be able to generate 
        bonds, angles and dihedral angles.
        """
        atoms = {"a0": connection[0], "b0": connection[1]}
    
        atoms.update({"a1": [x for x in np.where(connectivity[atoms["a0"]])[0]
                             if x != atoms["b0"]]})
        
        atoms.update({"b1": [x for x in np.where(connectivity[atoms["b0"]])[0]
                             if x != atoms["a0"]]})
        
        atoms.update({"a2": [[x for x in np.where(connectivity[y])[0]
                              if x != atoms["a0"]]
                             for y in atoms["a1"]]})
        
        atoms.update({"b2": [[x for x in np.where(connectivity[y])[0]
                              if x != atoms["b0"]]
                             for y in atoms["b1"]]})
        return atoms
    
    
    def _determine_atom_types(self, topology, connectivity, atoms):
        """
        Determine atom types of the relevant atoms from existing topology. 
        Some atom types need to be adjusted because of the new environment;
        this is currently only handled for a few cases.
        """
        #create atom_types dictionary for all atoms relevant for linker
        atom_types = {atoms["a0"]: topology.atoms[atoms["a0"]][1],
                      atoms["b0"]: topology.atoms[atoms["b0"]][1]}
        [atom_types.update({x: topology.atoms[x][1]}) for x in atoms["a1"]]
        [atom_types.update({x: topology.atoms[x][1]}) for x in atoms["b1"]]
        [[atom_types.update({x: topology.atoms[x][1]}) for x in y] 
         for y in atoms["a2"]]
        [[atom_types.update({x: topology.atoms[x][1]}) for x in y] 
         for y in atoms["b2"]]
        
        #check if atoms are phenyl connections, need to adjust atom types
        rings = RingTyper([x[4][0] for x in topology.atoms], connectivity)
        if (len(rings.ring_ids[atoms["a0"]]) > 0
              and len(rings.ring_ids[atoms["b0"]]) > 0):
            if all([x not in rings.ring_ids[atoms["b0"]] 
                    for x in rings.ring_ids[atoms["a0"]]]):
                if (atom_types[atoms["a0"]] == "ca" and 
                    atom_types[atoms["b0"]] == "ca"):
                    atom_types[atoms["a0"]] = "cp"
                    atom_types[atoms["b0"]] = "cp"
        
        if (atom_types[atoms["a0"]] == "ha"
              and atom_types[atoms["b0"]] in ["cg", "ch"]):
            atom_types[atoms["a0"]] = "ha"
            atom_types[atoms["b0"]] = "c1"
        if (atom_types[atoms["a0"]] in ["cg", "ch"] 
              and atom_types[atoms["b0"]] == "ha"):
            atom_types[atoms["a0"]] = "c1"
            atom_types[atoms["b0"]] = "ha"
            
        return atom_types 
    
    def _find_parameter_indeces(self, atoms, topology):
        """
        Determines indices of all new bonded parameters and stores them in 
        a ParameterDeterminator object.
        """
        parameter_determinator = ParameterDeterminator([])
        
        #bonds
        parameter_determinator.bond_indeces.append([atoms["a0"], atoms["b0"]])
        
        #angles
        for atom_a1 in atoms["a1"]:
            parameter_determinator.angle_indeces.append(
                [atom_a1, atoms["a0"], atoms["b0"]])
        for atom_b1 in atoms["b1"]:
            parameter_determinator.angle_indeces.append(
                [atoms["a0"], atoms["b0"], atom_b1])
            
        #dihedrals
        for atom_a1 in atoms["a1"]:             
            for atom_b1 in atoms["b1"]:
                parameter_determinator.dihedral_indeces.append(
                    [atom_a1, atoms["a0"], atoms["b0"], atom_b1])
                parameter_determinator.pair_indeces.append(
                    [atom_a1, atom_b1])
                
        for i, atom_a1 in enumerate(atoms["a1"]):
            for atom_a2 in atoms["a2"][i]:
                atom_list = [atom_a2 + 1, 
                             atom_a1 + 1, 
                             atoms["a0"] + 1, 
                             atoms["b0"] + 1]
                dihedral_exists = any([all([x in y[:4] for x in atom_list])
                                       for y in topology.dihedrals])
                if not dihedral_exists:
                    atom_list = [atom_a2, 
                                 atom_a1, 
                                 atoms["a0"], 
                                 atoms["b0"]]
                    parameter_determinator.dihedral_indeces.append(atom_list)
                    pair = [atom_a2, atoms["b0"]]
                    parameter_determinator.pair_indeces.append(pair)
                
        for i, atom_b1 in enumerate(atoms["b1"]):
            for atom_b2 in atoms["b2"][i]:
                atom_list = [atoms["a0"] + 1, 
                             atoms["b0"] + 1, 
                             atom_b1 + 1, 
                             atom_b2 + 1]
                dihedral_exists = any([all([x in y[:4] for x in atom_list])
                                       for y in topology.dihedrals])
                if not dihedral_exists:
                    atom_list = [atoms["a0"], atoms["b0"], atom_b1, atom_b2]
                    parameter_determinator.dihedral_indeces.append(atom_list)
                    pair = [atoms["a0"], atom_b2]
                    parameter_determinator.pair_indeces.append(pair)
                    
        #impropers
        if len(atoms["a1"]) == 2:
            parameter_determinator.improper_indeces.append(
                {"centre": atoms["a0"], 
                 "vertices": [atoms["a1"][0], atoms["a1"][1], atoms["b0"]]})
        if len(atoms["b1"]) == 2:
            parameter_determinator.improper_indeces.append(
                {"centre": atoms["b0"], 
                 "vertices": [atoms["b1"][0], atoms["b1"][1], atoms["a0"]]})            
    
        return parameter_determinator 
                    