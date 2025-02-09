import numpy as np
from copy import deepcopy

class BondTyper:
    """
    Tool to assign bond orders of all bonds in a molecule.
    """
    def __init__(self, connectivity, valence_states):
        self.connectivity = connectivity
        self.valence_states = valence_states
    
    def bond_order_assignment(self):
        """
        Determines bond orders based on bond locations and valence state
        (describes how many bonds an atom needs).
        """
        #get list of bonds from connectivity matrix
        cols, rows = np.meshgrid(range(self.connectivity.shape[0]), 
                                 range(self.connectivity.shape[0]))
        connectivity_ = self.connectivity * (rows < cols)
        bonds = np.array(np.where(connectivity_ == 1)).T
        
        tpss = [x["tps"] for x in self.valence_states]
        priority_order = np.argsort(tpss)
        
        for i, vs_index in enumerate(priority_order):
            print(f"Assigning bond orders on valence structure {i}.",
                  f"(PS: {self.valence_states[vs_index]['tps']})")
            bond_orders = self.get_bond_orders(
                self.valence_states[vs_index]["con_val"], bonds)
            if bond_orders is not None:
                print("....Success")
                break
            else:
                print("....Fail")
        return bonds, bond_orders, self.valence_states[vs_index]
        
    @staticmethod
    def get_bond_orders(con_val, bonds):
        """
        Recursive function to assign bond orders. Guesses location of double
        bonds/triple bonds and moves on. If it fails it turns back to a point
        where it had to guess and retry with a differnt guess.
        
        Algorithm is described in https://doi.org/10.1016/j.jmgm.2005.12.005
        
        Return: None if assignment failed or
                np.array of int if successful
        """
        con_val = deepcopy(con_val)
        bond_orders = np.zeros(len(bonds))
        assigned_bonds = np.zeros(len(bonds))
        
        change = True
        while change:
            change = False
            for atom_idx,(con,val) in enumerate(con_val):
                if val == 0:
                    continue
                # if valence equals number of bonds, set all to BO=1
                if con == val:
                    change = True
                    active_bonds = np.where(bonds == atom_idx)[0]
                    active_bonds = [x for x in active_bonds 
                                    if assigned_bonds[x] == 0]
                    bond_orders[active_bonds] = 1
                    for bond_idx in active_bonds:
                        con_val[bonds[bond_idx][0]] -= 1
                        con_val[bonds[bond_idx][1]] -= 1
                        
                    
                    assigned_bonds[active_bonds] = 1
                    continue
                    
                # if only one connection, set that connection equal to valence
                if con == 1:
                    change = True
                    active_bonds = np.where(bonds == atom_idx)[0]
                    active_bonds = [x for x in active_bonds 
                                    if assigned_bonds[x] == 0]
                    bond_orders[active_bonds] = val
                    for bond_idx in active_bonds:
                        con_val[bonds[bond_idx][0]][0] -= 1
                        con_val[bonds[bond_idx][1]][0] -= 1
                        con_val[bonds[bond_idx][0]][1] -= val
                        con_val[bonds[bond_idx][1]][1] -= val
                        
                    assigned_bonds[active_bonds] = 1
                    continue
            
            error = [(x[0] < 0 or x[1] < 0) for x in con_val]
            if any(error):
                return None
            error = [((x[0] == 0 and x[1] != 0) or (x[1] == 0 and x[0] != 0)) 
                     for x in con_val]
            if any(error):
                return None
            
            if not change and (0 in assigned_bonds):
                #try setting first bond to BO = 1, then 2, then 3
                trial_bond = np.where(assigned_bonds == 0)[0][0]
                for BO_trial in range(1,4):
                    trial_con_val = deepcopy(con_val)
                    bond_orders[trial_bond] = BO_trial
                    assigned_bonds[trial_bond] = 1
                    trial_con_val[bonds[trial_bond][0]][0] -= 1
                    trial_con_val[bonds[trial_bond][1]][0] -= 1
                    trial_con_val[bonds[trial_bond][0]][1] -= BO_trial
                    trial_con_val[bonds[trial_bond][1]][1] -= BO_trial
                    
                    atom_mapping = {}
                    unassigned_bonds = np.where(assigned_bonds == 0)[0]
                    for sub_idx, main_idx in enumerate(unassigned_bonds):
                        atom_mapping.update({main_idx: sub_idx})
                    
                    #start next step in recursion
                    bond_orders_sub = BondTyper.get_bond_orders(
                        trial_con_val, bonds[unassigned_bonds])
                    #if subroutine was successful, can exit here
                    #   otherwise, try higher bond order for trial bond
                    if bond_orders_sub is not None:
                        for main_idx in atom_mapping:
                            bond_orders[main_idx] = \
                                bond_orders_sub[atom_mapping[main_idx]]
                        return bond_orders
                
            if np.sum(con_val) == 0:
                return bond_orders 
                
        return None

        




        