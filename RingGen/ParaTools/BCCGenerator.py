from .BCC_corrections import bcc_correction_dictionary

import numpy as np

class BCCGenerator:
    """
    Generates bond charge corrections to be used with AM1 charges (AM1-BCC).
    Uses AtomTypeDeterminator object created by TopologyGenerator to set all 
    properties needed to assign BCC.
    
    Check https://doi.org/10.1002/jcc.10128 for the general method.
    """
    def __init__(self, atom_type_determinator):
        self.coordinates = atom_type_determinator.coordinates
        self.rings = atom_type_determinator.rings
        self.bonds = atom_type_determinator.bonds
        self.bond_orders = atom_type_determinator.bond_orders
        self.valence_state = atom_type_determinator.valence_state

        self.nb_atoms = len(self.coordinates.elements)
        self.elements = self.coordinates.elements
        self.connectivity = self.coordinates.create_connectivity().connectivity
     
    def get_bcc_charges(self):
        """
        Processes properties to determine BCC-atom-types and assign BCC

        Returns
        -------
        charge_corrections : list of float
            List of charge corrections for each atom.
        """
        self.formal_charges = self.get_formal_charges()
        self.valency = self.get_valency()
        self.aromaticity = self.get_aromaticity()
        self.atom_bond_type_list = self.get_atom_bond_type_list()
        self.atom_types = self.get_atom_types()
        self.bcc_corrections = self.get_charge_correction()

        return self.bcc_corrections

    def get_formal_charges(self):
        # No formal charges allowed so far,
        # might be implemented later
        formal_charges = [0]*self.nb_atoms
        return formal_charges

    def get_valency(self):
        valency = []
        charge_descriptor_dictionary = {0:"", 1:"+", -1:"-", 2:"+", -2:"-"}
        for i,element in enumerate(self.elements):
            charge = int(self.formal_charges[i])
            charge_descriptor = charge_descriptor_dictionary[charge]
            valency.append(f"{element}"
                           +f"{charge_descriptor}"
                           +f"(x{int(sum(self.connectivity[i]))})")
        return np.array(valency)

    def get_aromaticity(self):

        aromaticity = np.array(["non"] * self.nb_atoms)
        aromaticity_assigned = np.zeros(self.nb_atoms)

        atom_class_X = ["C(x3)", "N(x2)", "P(x2)", 
                        "N+(x3)", "P+(x3)", "O+(x2)", "S+(x2)"]
        atom_class_Y = ["C-(x3)", "N-(x2)", "O(x2)", "S(x2)", "N(x3)", "P(x3)"]


        #test 1
        for ring in self.rings.rings:
            #check if all 6 atoms unassigned
            if len(ring) == 6 and sum(aromaticity_assigned[ring]) == 0:
                #check if all atoms are the correct type
                if all([x in atom_class_X 
                        for x in np.array(self.valency)[ring]]):
                    aromaticity[ring] = "ar6"
                    aromaticity_assigned[ring] = 1
        #test 2
        for ring in self.rings.rings:
            #check if 4 atoms unassigned
            if len(ring) == 6 and sum(aromaticity_assigned[ring]) == 2:
                assigned_atoms = [atom for atom in ring 
                                  if aromaticity_assigned[atom] == 1]
                unassigned_atoms = [atom for atom in ring 
                                    if aromaticity_assigned[atom] == 0]
                #check that unassigned atoms are the correct type and 
                # assigned atoms are adjacent
                if (all([x in atom_class_X 
                         for x in np.array(self.valency)[unassigned_atoms]]) 
                   and 
                   (assigned_atoms[1] in 
                      np.where(self.connectivity[assigned_atoms[0]]==1)[0])):
                    aromaticity[unassigned_atoms] = "ar6"
                    aromaticity_assigned[ring] = 1
        #test 3
        for ring in self.rings.rings:
            # check if 2 atoms unassigned
            if len(ring) == 6 and sum(aromaticity_assigned[ring]) == 2: 
                assigned_atoms = [atom for atom in ring 
                                  if aromaticity_assigned[atom] == 1]
                unassigned_atoms = [atom for atom in ring 
                                    if aromaticity_assigned[atom] == 0]
                # check that unassigned atoms are the correct type and 
                # unassigned atoms are adjacent
                if (all([x in atom_class_X 
                         for x in np.array(self.valency)[unassigned_atoms]]) 
                    and 
                    (unassigned_atoms[1] in 
                     np.where(self.connectivity[unassigned_atoms[0]]==1)[0])):
                    aromaticity[unassigned_atoms] = "ar6"
                    aromaticity_assigned[ring] = 1
        #test 4
        for ring in self.rings.rings:
            if len(ring) == 7:
                #check carbocation present
                if "C+(x3)" in self.valency[ring]:
                    other_atoms = [x for x in ring 
                                   if self.valency[x] != "C+(x3)"]
                    #check that unassigned atoms are the correct type
                    if all([x in atom_class_X 
                            for x in np.array(self.valency)[other_atoms]]):
                        aromaticity[other_atoms] = "ar7"
                        aromaticity_assigned[ring] = 1
        #test 5
        for ring in self.rings.rings:
            if len(ring) == 5:
                if any([x in atom_class_Y 
                        for x in np.array(self.valency)[ring]]):
                    if sum(aromaticity_assigned[ring]) == 0:
                        aromaticity[ring] = "ar5"
                        aromaticity_assigned[ring] = 1

        return aromaticity

    def get_atom_bond_type_list(self):
        atom_bond_type_list = []
        for i,_ in enumerate(self.elements):
            active_bonds = np.where(self.bonds == i)[0]    
            atom_bond_type_list.append([self.get_bond_type(bond_idx)
                                        for bond_idx in active_bonds])
        return atom_bond_type_list
    
    def get_bond_type(self, bond_idx):
        bond_order = self.bond_orders[bond_idx]
        bond_ind = self.bonds[bond_idx]
        bond_dict = {bond_ind[0]: bond_ind[1], bond_ind[1]: bond_ind[0]}
        
        if bond_order == 3:
            bond_type = "03"
        if bond_order == 2:
            if self._check_aromatic_bond(bond_ind):
                bond_type = "08"
            elif self._check_delocalized_bond(bond_ind):
                bond_type = "09"
            else:
                bond_type = "02"
        if bond_order == 1:
            if self._check_aromatic_bond(bond_ind):
                bond_type = "01"
            elif self._has_neg_charge(bond_ind):
                bond_type = "01"
            else:
                bond_type = "01"
        
        return [bond_ind, bond_dict, bond_type]
    
    def _check_aromatic_bond(self, bond_ind):
        return all([self.aromaticity[x] != "non" for x in bond_ind])
    
    def _check_delocalized_bond(self, bond_ind):
        return ((self.valence_state["con_val"][bond_ind[0]][1] >= 5 and 
                 self.valence_state["con_val"][bond_ind[1]][0] == 1) or
                (self.valence_state["con_val"][bond_ind[1]][1] >= 5 and
                 self.valence_state["con_val"][bond_ind[0]][0] == 1))
    def _has_neg_charge(self, bond_ind):
        return any([self.formal_charges[x] == -1 for x in bond_ind])

    def get_atom_types(self):
        atom_types = []
        for i,atom in enumerate(self.elements):
            atom_types.append(self.get_atom_type(i,atom))
        return atom_types
    
    def _has_Xaro_bond(self, idx):
        for bond in self.atom_bond_type_list[idx]:
            bonded_idx = bond[1][idx]
            bonded_element = self.elements[bonded_idx]
            bond_type = bond[2]
            if self.aromaticity[bonded_idx] == "non":
                continue
            if bonded_element == "O" and bond_type in ["01", "07"]:
                return True
            if bonded_element == "N" and bond_type in ["07", "08"]:
                return True
        return False

    def get_atom_type(self, i, atom):
        if atom == "C":
            if self.valency[i] == "C(x4)":
                return 11
            if self.valency[i] != "C(x3)":
                return 15
            #if self.aromaticity[i] == "non":
            if not any([bond[2] in ["07", "08"] 
                        for bond in self.atom_bond_type_list[i]]):
                double_bond = [x for x in self.atom_bond_type_list[i] 
                               if x[2] == "02"][0]
                double_bonded_element = self.elements[double_bond[1][i]]
                if double_bonded_element == "C":
                    return 12
                if double_bonded_element in ["N", "P"]:
                    return 13
                if double_bonded_element in ["O", "S"]:
                    return 14
            else:
                if self._has_Xaro_bond(i):
                    return 17
                else:
                    return 16
                
        if atom == "N":
            if self.valency[i] in ["N(x4)", "N+(x4)"]:
                return 21
            elif self.valency[i] == "N(x3)":
                carbo_cation = any([self.formal_charges[bond[1][i]] == 1
                                    for bond in self.atom_bond_type_list[i]])
                aromatic = (self.aromaticity[i] == "aro5")
                on_aromatic_ring = \
                    any([any([beta_bond[2] in ["07", "08"] 
                              for beta_bond 
                                  in self.atom_bond_type_list[bond[1][i]]
                              if beta_bond[1][bond[1][i]] != i])
                         for bond in self.atom_bond_type_list[i]])
                on_other_double_bond = \
                    any([any([beta_bond[2] == "02"
                              for beta_bond 
                                  in self.atom_bond_type_list[bond[1][i]]
                              if beta_bond[1][bond[1][i]] != i])
                         for bond in self.atom_bond_type_list[i]])
                if carbo_cation or aromatic or on_aromatic_ring:
                    return 23
                elif on_other_double_bond:
                    return 22
                else:
                    return 21
            elif self.valency[i][-4:] == "(x2)":
                if self.valency[i] == "N-(x2)":
                    near_double_bond = \
                        any([any([beta_bond[2] in ["02", "07", "08"] 
                                  for beta_bond 
                                      in self.atom_bond_type_list[bond[1][i]]
                                  if beta_bond[1][bond[1][i]] != i])
                             for bond in self.atom_bond_type_list[i]])
                    if near_double_bond:
                        return 22
                    else:
                        return 21                    
                else:
                    if self.formal_charges[i] == 0:
                        return 24
                    else:
                        return 25
            else:
                return 25
            
        if atom == "O":
            if self.valency[i] == "O(x2)":
                return 31
            else:
                bond = self.atom_bond_type_list[i][0]
                other_atom = bond[1][i]
                other_atom_in_ring = (len(self.rings.ring_ids[other_atom])!=0)
                other_atom_on_N = \
                    (sum([self.elements[x[1][other_atom]] == "N"
                          for x in self.atom_bond_type_list[other_atom]])
                     >= 1)
                other_atom_on_O = \
                    (sum([self.elements[x[1][other_atom]] == "O"
                          for x in self.atom_bond_type_list[other_atom]])
                     >= 2)
                if other_atom_in_ring and (other_atom_on_N or other_atom_on_O):
                    return 33
                elif other_atom_on_O:
                    return 32
                else:
                    return 31

        if atom == "P":
            if self.valency[i][-4:] == "(x4)":
                return 42
            if self.valency[i][-4:] == "(x3)":
                for bond in self.atom_bond_type_list[i]:
                    if bond[2] == "02":
                        return 42
            else:
                return 41
            
        if atom == "S":
            if self.valency[i][-4:] == "(x4)":
                return 53
            if self.valency[i][-4:] == "(x3)":
                return 52
            else:
                return 51
            
        if atom == "Si":
            return 61
        if atom == "F":
            return 71
        if atom == "Cl":
            return 72
        if atom == "Br":
            return 73
        if atom == "I":
            return 74
        if atom == "H":
            return 91

        print("Couldn't assign atom type.")
        return 0

    def get_charge_correction(self):
        charge_corrections = []
        for i,atom in enumerate(self.elements):
            bcc_correction = 0
            for bond in self.atom_bond_type_list[i]:
                bcc_type = (f"{self.atom_types[i]}"
                            +f"{bond[2]}"
                            +f"{self.atom_types[bond[1][i]]}")
                bcc_correction += bcc_correction_dictionary.get(bcc_type, 0)
                if bcc_type not in bcc_correction_dictionary.keys():
                    print(f"Bond type {bcc_type} not defined!")
            charge_corrections.append(bcc_correction)
        return charge_corrections
