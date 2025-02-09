import numpy as np
import itertools
from copy import deepcopy

class ParameterDeterminator:
    """
    Finds indices for bonded parameters based on connectivity matrix.
    
    Attributes
    ----------
    connectivity : NxN np.array
    bond_indeces : list of list of int
    angle_indeces : list of list of int
    dihedral_indeces : list of list of int
    improper_indeces : list of list of int
    pair_indeces : list of list of int
    """
    def __init__(self, connectivity):
        self.connectivity = connectivity
        self.bond_indeces = self.get_bond_indeces()
        self.angle_indeces = self.get_angle_indeces()
        self.dihedral_indeces = self.get_dihedral_indeces()
        self.improper_indeces = self.get_improper_indeces()
        self.pair_indeces = self.get_pair_indeces()

    def get_bond_indeces(self):
        """
        Finds pairs of indeces that have a 1 in the connectivity matrix 
        (that are bonded)
        """
        bond_indeces = []
        for i,a in enumerate(self.connectivity):
            for j,b in enumerate(a):
                if i<j:
                    if b == 1:
                        bond_indeces.append([i,j])
        return bond_indeces

    def get_angle_indeces(self):
        """
        Finds triples of indeces for all atoms that are connected to at least 
        two atoms (that form an angle)
        """
        angle_indeces = []
        for i,connections in enumerate(self.connectivity):
            other_two_atoms_of_angle = \
                list(itertools.combinations(np.where(connections==1)[0],2))
            for others in other_two_atoms_of_angle:
                angle_indeces.append([others[0],i,others[1]])
        return angle_indeces

    def get_dihedral_indeces(self):
        """
        Find dihedrals based on connected angles (one arm of angle overlaps 
        with another angle)
        """
        dihedral_indeces=[]
        for i,angle1 in enumerate(self.angle_indeces):
            for j,angle2 in enumerate(self.angle_indeces):
                if i < j:
                    if angle1[1] == angle2[0] and angle1[2] == angle2[1]:
                        dihedral_indeces.append(
                            [angle1[0],angle1[1],angle1[2],angle2[2]])
                    elif angle1[1] == angle2[2] and angle1[2] == angle2[1]:
                        dihedral_indeces.append(
                            [angle1[0],angle1[1],angle1[2],angle2[0]])
                    elif angle1[0] == angle2[1] and angle1[1] == angle2[2]:
                        dihedral_indeces.append(
                            [angle1[2],angle1[1],angle1[0],angle2[0]])
                    elif angle1[0] == angle2[1] and angle1[1] == angle2[0]:
                        dihedral_indeces.append(
                            [angle1[2],angle1[1],angle1[0],angle2[2]])
        return dihedral_indeces
    
    def get_improper_indeces(self):
        """
        Finds atoms that have three bonds and can form an improper dihedral.
        """
        improper_indeces = []
        for i, connections in enumerate(self.connectivity):
            if sum(connections) == 3:
                vertices = np.where(connections == 1)[0]
                improper_indeces.append({"centre": i, "vertices": vertices})
        return improper_indeces
        
    def get_pair_indeces(self):
        """
        Finds pairs based on terminal atoms of dihedral angles
        """
        pairs=[]
        for dihedral in self.dihedral_indeces:
            pair = [dihedral[0],dihedral[3]]
            pair.sort()
            # only add if not a duplicate
            if not any([(pair[0] == x[0] and pair[1] == x[1]) for x in pairs]):
                pairs.append(pair)
        
        # remove atoms that have a closer connection than 3 bonds
        # e.g. happens in small rings
        pairs_raw = deepcopy(pairs)
        pairs = []
        for pair in pairs_raw:
            well_spaced = True
            for dihedral in self.dihedral_indeces:
                if (all([x in dihedral[0:3] for x in pair]) or 
                    all([x in dihedral[1:4] for x in pair])):
                    well_spaced = False
            if well_spaced:
                pairs.append(pair)        
        return pairs