import numpy as np
from copy import deepcopy

class ValenceStateSampler:
    """
    Tool to assign valence state of a molecule based on coordinates.
    
    See https://doi.org/10.1016/j.jmgm.2005.12.005 for method.
    """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.valence_states = []

    def sample(self, N):
        tps_scorer = TotalPenaltyScorer()
        min_ps_valency =tps_scorer.get_min_ps_valency(self.coordinates)
        bond_orders_0 = self.coordinates.connectivity
        con_val_0 = np.vstack([np.sum(bond_orders_0, axis = 0),
                               np.sum(bond_orders_0, axis = 0)]).T

        # establish which bonds should be modified, if valency is already 
        # larger than minimum, it should not be modified
        possible_bond_increments_raw = np.where(bond_orders_0 == 1)
        possible_bond_increments_0 = []
        for idx1, idx2 in zip(possible_bond_increments_raw[0], 
                              possible_bond_increments_raw[1]):
            if idx1 < idx2:
                possible_bond_increments_0.append([idx1, idx2])
        possible_bond_increments_0 = self.remove_aps0_atoms(
            possible_bond_increments_0, con_val_0, min_ps_valency)
                
        #sample
        for i in range(N):
            con_val = deepcopy(con_val_0)
            possible_bond_increments = deepcopy(possible_bond_increments_0)
            increments = 0
            while len(possible_bond_increments) > 0 and increments < 2000:
                increments +=1
                
                #check preferred bond_increments:
                preferred_increments = self.get_preferred_bonds(
                    possible_bond_increments, con_val_0, min_ps_valency)
                if preferred_increments:
                    inc_ind = np.random.choice(preferred_increments)
                else:
                    inc_ind = np.random.randint(len(possible_bond_increments))
                
                bond_increment = possible_bond_increments[inc_ind]
                con_val[bond_increment,1] += 1
                
                
                possible_bond_increments = self.remove_aps0_atoms(
                    possible_bond_increments, con_val, min_ps_valency)
            
            tps = tps_scorer.score(self.coordinates, con_val)
            self.valence_states.append({"con_val": con_val,
                                        "tps": tps})
            
        print("Minimum valence penalty score: ",
              f"{min(x['tps'] for x in self.valence_states)}", sep = "")
        return self.valence_states
    
    @staticmethod
    def remove_aps0_atoms(possible_bond_increments, con_val, min_ps_valency):
        good_bonds = []
        for i, ind in enumerate(possible_bond_increments):
            if any([(con_val[ind[x],1] >= min_ps_valency[ind[x]]) for x in range(2)]):
                good_bonds.append(i)
        for i in sorted(good_bonds, reverse=True):
            del possible_bond_increments[i]
        return possible_bond_increments
                
    @staticmethod
    def get_preferred_bonds(possible_bond_increments,
                            con_val_0,
                            min_ps_valency):
        preferred_increments = []
        for i, ind in enumerate(possible_bond_increments):
            if (len(np.where(possible_bond_increments == ind[0])[0]) == 1 and
                len(np.where(possible_bond_increments == ind[1])[0]) <= 2 and
                (con_val_0[ind[0],1]) < min_ps_valency[ind[0]]):
                preferred_increments.append(i)
            if (len(np.where(possible_bond_increments == ind[1])[0]) == 1 and
                len(np.where(possible_bond_increments == ind[0])[0]) <= 2 and
                (con_val_0[ind[1],1]) < min_ps_valency[ind[1]]):
                preferred_increments.append(i)
        return preferred_increments
    
class TotalPenaltyScorer:
    def __init__(self):
        self.aps_list = \
           [AtomicPenaltyScoreAtom("H", [1], {1:0, 2:64}),        
            AtomicPenaltyScoreAtom("F", [1], {1:0, 2:64}),        
            AtomicPenaltyScoreAtom("Cl", [1], {1:0, 2:64}),        
            AtomicPenaltyScoreAtom("Br", [1], {1:0, 2:64}),        
            AtomicPenaltyScoreAtom("I", [1], {1:0, 2:64}),     
            AtomicPenaltyScoreAtom_CNR1("C", [1], {3:0, 4:1, 5:32}),
            AtomicPenaltyScoreAtom("C", [1], {3:1, 4:0, 5:32}),
            AtomicPenaltyScoreAtom_XNO("C", [3], {4:32, 5:0, 6:32}, 2),        
            AtomicPenaltyScoreAtom("C", [2, 3, 4, 5, 6], 
                                      {2:64, 3:32, 4:0, 5:32, 6:64}), 
            AtomicPenaltyScoreAtom("C", [4], {4:0}),
            AtomicPenaltyScoreAtom_NNR1("N", [1], {2:0, 3:0}),
            AtomicPenaltyScoreAtom("N", [1], {2:3, 3:0, 4:32}),
            AtomicPenaltyScoreAtom_NNR2("N", [2], {3:1, 4:0}),
            AtomicPenaltyScoreAtom("N", [2], {2:4, 3:0, 4:2}),
            AtomicPenaltyScoreAtom_XNO("N", [3], 
                                          {3:64, 4:32, 5:0, 6:32}, 2),
            AtomicPenaltyScoreAtom_XNO("N", [3], {3:1, 4:0}, 1),
            AtomicPenaltyScoreAtom("N", [3], {2:32, 3:0, 4:1, 5:2}),
            AtomicPenaltyScoreAtom("N", [4], {2:64, 3:0, 4:64}),
            AtomicPenaltyScoreAtom_X1N("O", [1], {1:0, 2:1}),
            AtomicPenaltyScoreAtom("O", [1], {1:1, 2:0, 3:64}),
            AtomicPenaltyScoreAtom("O", [2], {1:32, 2:0, 3:64}),
            AtomicPenaltyScoreAtom("P", [1], {2:2, 3:0, 4:32}),
            AtomicPenaltyScoreAtom("P", [2], {2:4, 3:0, 4:2}),
            AtomicPenaltyScoreAtom("P", [3], {2:32, 3:0, 4:1}),
            AtomicPenaltyScoreAtom_XNO("P", [4], {5:32, 6:0, 7:32}, 2),
            AtomicPenaltyScoreAtom_XNO("P", [4], {6:32, 7:0}, 3),
            AtomicPenaltyScoreAtom("P", [4], {3:64, 4:1, 5:0, 6:32}),
            AtomicPenaltyScoreAtom_X1N("S", [1], {1:0, 2:1}),
            AtomicPenaltyScoreAtom("S", [1], {1:2, 2:0, 3:64}),
            AtomicPenaltyScoreAtom("S", [2], {1:32, 2:0, 3:32, 4:1}),
            AtomicPenaltyScoreAtom("S", [3], {3:1, 4:0, 5:2, 6:2}),
            AtomicPenaltyScoreAtom_XNO("S", [4], {6:0, 7:32}, 2),
            AtomicPenaltyScoreAtom_XNO("S", [4], {6:32, 7:0}, 3),
            AtomicPenaltyScoreAtom_XNO("S", [4], {6:32, 7:0}, 4),
            AtomicPenaltyScoreAtom("S", [4], {4:4, 5:2, 6:0})]

    def score(self, coordinates, con_val):
        tps = 0
        for i,_ in enumerate(coordinates.elements):
            for aps in self.aps_list:
                if aps.check_if_applies(i, coordinates):
                    tps += aps.apss[con_val[i,1]]
                    break
        return tps
    
    def get_min_ps_valency(self, coordinates):
        min_ps_valency = []
        for i,_ in enumerate(coordinates.elements):
            for aps in self.aps_list:
                if aps.check_if_applies(i, coordinates):
                    min_ps_valency.append(min(aps.apss, key = aps.apss.get))
                    break
        return min_ps_valency

class AtomicPenaltyScoreAtom:
    def __init__(self, element, cons, apss):
        self.element = element
        self.cons = cons
        self.apss = apss
        self.avs = apss.keys()
        
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
                sum(coordinates.connectivity[index]) in self.cons):
            return True
        else:
            return False
        
    def get_penalty_score(self, av):
        return self.apss[av]
    
    
class AtomicPenaltyScoreAtom_XNO(AtomicPenaltyScoreAtom):
    def __init__(self, element, cons, apss, attached_SO):
        self.element = element
        self.cons = cons
        self.apss = apss
        self.avs = apss.keys()
        self.attached_SO = attached_SO
    
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
            sum(coordinates.connectivity[index]) in self.cons):
            attached_atoms_ind = \
                np.where(coordinates.connectivity[index] == 1)[0]
            is_SO = [(coordinates.elements[x] in ["O", "S"])
                     for x in attached_atoms_ind]
            is_mono_con = [(sum(coordinates.connectivity[x]) == 1)
                           for x in attached_atoms_ind]
            if (np.array(is_SO)*np.array(is_mono_con)).sum()==self.attached_SO:
                return True
        else:
            return False

class AtomicPenaltyScoreAtom_X1N(AtomicPenaltyScoreAtom):
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
            sum(coordinates.connectivity[index]) in self.cons):
            attached_atom_idx = \
                np.where(coordinates.connectivity[index] == 1)[0][0]
            if coordinates.elements[attached_atom_idx] == "N":
                return True
        else:
            return False    

class AtomicPenaltyScoreAtom_CNR1(AtomicPenaltyScoreAtom):
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
            sum(coordinates.connectivity[index]) in self.cons):
            attached_atom_idx = \
                np.where(coordinates.connectivity[index] == 1)[0][0]
            if (coordinates.elements[attached_atom_idx] == "N" and 
                sum(coordinates.connectivity[attached_atom_idx]) == 2):
                return True
        else:
            return False        

class AtomicPenaltyScoreAtom_NNR1(AtomicPenaltyScoreAtom):
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
                sum(coordinates.connectivity[index]) in self.cons):
            attached_atom_idx = \
                np.where(coordinates.connectivity[index] == 1)[0][0]
            if (coordinates.elements[attached_atom_idx] == "N" and 
                sum(coordinates.connectivity[attached_atom_idx]) == 2):
                return True
        else:
            return False  
      
class AtomicPenaltyScoreAtom_NNR2(AtomicPenaltyScoreAtom):
    def check_if_applies(self, index, coordinates):
        if (coordinates.elements[index] == self.element and
                sum(coordinates.connectivity[index]) in self.cons):
            attached_atom_ind = \
                np.where(coordinates.connectivity[index] == 1)[0]
            if ((coordinates.elements[attached_atom_ind[0]] == "N" and 
                 sum(coordinates.connectivity[attached_atom_ind[0]]) == 1) or
                (coordinates.elements[attached_atom_ind[1]] == "N" and 
                 sum(coordinates.connectivity[attached_atom_ind[1]]) == 1)):
                return True
        else:
            return False  