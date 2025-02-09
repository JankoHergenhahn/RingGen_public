import importlib
import numpy as np
from RingGen.Coordinates import Coordinates
from .ValenceStateSampler import ValenceStateSampler
from .BondTyper import BondTyper
from .RingTyper import RingTyper


class AtomTypeDeterminator:
    """
    Tool to assign GAFF atom types to atoms of a structure.
    
    Atom types are assigned based on structural features of the local 
    environements such as number of connected atoms, rings, bond types etc.
    
    This class implements algorithms of the Antechamber program, for details 
    of the assignment method, see: https://doi.org/10.1016/j.jmgm.2005.12.005
    """
    def __init__(self, coordinates, easy_valence = False):
        self.coordinates = coordinates
        
        #get properties
        if easy_valence:
            valence_states = self._get_easy_valence_state(coordinates)
        else:
            val_sampler = ValenceStateSampler(coordinates)
            valence_states = val_sampler.sample(len(coordinates.elements))
        b_ty = BondTyper(coordinates.connectivity, valence_states)
        bonds, bond_orders, active_valence_state = b_ty.bond_order_assignment()
        rings = RingTyper(coordinates.elements, 
                          coordinates.connectivity, 
                          bonds, 
                          bond_orders)
        bond_types = self._get_bond_types(bonds, 
                                          bond_orders, 
                                          rings, 
                                          active_valence_state)
        
        #get features
        features = self._get_features(coordinates, bonds, bond_types, rings)
        self.elements = coordinates.elements
        self.rings = rings
        self.bonds = bonds
        self.bond_orders = bond_orders
        self.bond_types = bond_types
        self.valence_state = active_valence_state
        
        self.atom_types = self._get_atom_types(features)
        self.refine_atom_types(coordinates.connectivity, 
                               bonds, 
                               bond_orders, 
                               rings)
        
    def get_atom_types(self):
        return self.atom_types
        
    def _get_coordinates_from_file(self, file):
        with open(file) as f:
            content = f.readlines()

        elements = []
        coordinates = []
        for i,line in enumerate(content):
            if "Empirical Formula:" in line:
                N_atoms = int(line.split()[-2])
            if line == "                             CARTESIAN COORDINATES\n":
                start_index = i+2
        for line in content[start_index:start_index + N_atoms]:
            elements.append(line.split()[1])
            coordinates.append([float(line.split()[2]), 
                                float(line.split()[3]), 
                                float(line.split()[4])])
        coordinates = np.array(coordinates)
        return Coordinates().set(elements, coordinates).create_connectivity()
    
    def _get_charges_from_file(self, file):
        with open(file) as f:
            content = f.readlines()

        am1_charges = []
        for i,line in enumerate(content):
            if "Empirical Formula:" in line:
                N_atoms = int(line.split()[-2])
            if "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS" in line:
                start_index = i+3
        for line in content[start_index:start_index + N_atoms]:
            am1_charges.append(float(line.split()[2]))
        return am1_charges
    
    def _get_easy_valence_state(self, coordinates):
        easy_val_dict = {"H": 1, "C": 4, "N": 3, "O": 2, "S": 2}
        con_val = []
        for i, element in enumerate(coordinates.elements):
            con_val.append([np.sum(coordinates.connectivity[i]),
                            easy_val_dict[element]])
        con_val = np.array(con_val)
        return [{"con_val": con_val, "tps": 0}]
        
    def _get_features(self, coordinates, bonds, bond_types, rings):
        atom_features = []
        # atomic number; 
        # number of atoms connected; 
        # number of attached hydrogen atoms; 
        # number of electron-withdrawal atoms (N, O, F, Cl and Br) 
        #                   binding to the immediately connected atom; 
        # atomic property; 
        # chemical environment definition
        
        atom_dict = {"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,
                     "Si":14,"P": 15,"S":16,"Cl":17,"Br":35,"I":53}
        elements_ = np.array(coordinates.elements)
        for atom_idx, element in enumerate(elements_):
            #Basic features
            atomic_nb = atom_dict[element]
            attached_atoms = \
                np.where(coordinates.connectivity[atom_idx] == 1)[0]
            nb_attached_atoms = len(attached_atoms)
            nb_attached_H = np.sum(elements_[attached_atoms] == "H")
            if element == "H":                
                attached_atom = \
                    np.where(coordinates.connectivity[atom_idx])[0][0]
                attached_atoms_2 = \
                    np.where(coordinates.connectivity[attached_atom] == 1)[0]
                nb_connected_EWG = \
                    np.sum(elements_[x] 
                           in ["O", "N", "F", "Cl", "Br", "I", "S"]
                           for x in attached_atoms_2)
            else:
                nb_connected_EWG = 0
            #Atomic properties
            atomic_properties = []
            active_bonds = np.where(bonds == atom_idx)[0]
            for bond in active_bonds:
                atomic_properties.append(bond_types[bond])
                if bond_types[bond] == "SB":
                    atomic_properties.append("sb")
                if bond_types[bond] == "DB":
                    atomic_properties.append("db")
            for ring in rings.ring_ids[atom_idx]:
                atomic_properties.append(rings.ring_types[ring])
                atomic_properties.append(f"RG{len(rings.rings[ring])}")
            for bond_type in ["sb", "db", "tb", "SB", "DB", "DL"]:
                atomic_properties.append(
                    f"{atomic_properties.count(bond_type)}{bond_type}")
            for attached_atom in attached_atoms:
                common_rings = np.intersect1d(rings.ring_ids[attached_atom],
                                              rings.ring_ids[atom_idx])
                common_AR1_rings = any([rings.ring_types[x] == "AR1"
                                        for x in common_rings])
                if common_AR1_rings:
                    continue
                for attached_atom_ring in rings.ring_ids[attached_atom]:
                    if attached_atom_ring not in rings.ring_ids[atom_idx]:
                        atomic_properties.append(
                            f"1RG{len(rings.rings[attached_atom_ring])}")
            
            #Enter features
            feature_list = [str(atomic_nb), 
                            str(nb_attached_atoms), 
                            str(nb_attached_H), 
                            str(nb_connected_EWG),
                            atomic_properties]
            atom_features.append(feature_list)
        #Need features above first before chemical environments is determined
        for atom_idx, element in enumerate(elements_):
            chemical_environment = \
                self._get_chemical_environments(atom_idx, 
                                                coordinates, 
                                                bonds, 
                                                bond_types, 
                                                atom_features)
            atom_features[atom_idx].append(chemical_environment)
        return atom_features
    
    def _get_chemical_environments(self, 
                                   atom_idx, 
                                   coordinates, 
                                   bonds, 
                                   bond_types, 
                                   atom_features):
        elements_ = np.array(coordinates.elements)
        connectivity_ = coordinates.connectivity
        paths = self._get_paths(atom_idx, connectivity_, depth = 3)
        chemical_environments = []
        for path in paths:
            chemical_environments.append(
                [self._get_environment_string(idx_1, idx_2,
                                              elements_, connectivity_,
                                              bonds, bond_types, atom_features)
                 for idx_1, idx_2 in zip([atom_idx] + path[:-1],path)])
        return chemical_environments
            
    def _get_environment_string(self, idx_1, idx_2, elements, connectivity, 
                                bonds, bond_types, atom_features):
        bond_idx = np.intersect1d(np.where(bonds == idx_1)[0], 
                                  np.where(bonds == idx_2)[0])[0]
        return (f"{elements[idx_2]}"
                +f"{sum(connectivity[idx_2])}"
                +f"[{bond_types[bond_idx]}',"
                +f"{','.join(atom_features[idx_2][4])}]")
    
    def _get_paths(self, atom_idx, connectivity, depth, blocked_atom = -1):
        attached_atoms = np.where(connectivity[atom_idx] == 1)[0]
        attached_atoms = [x for x in attached_atoms if x!= blocked_atom]
        
        if depth == 1:
            return [[x] for x in attached_atoms]
        
        paths = []
        for attached_atom in attached_atoms:
            sub_paths = self._get_paths(attached_atom, 
                                        connectivity, 
                                        depth - 1, 
                                        atom_idx)
            paths.append([attached_atom])
            for sub_path in sub_paths:
                paths.append([attached_atom] + sub_path)
        
        return paths
    
            
    def _get_bond_types(self, bonds, bond_orders, rings, active_valence_state):
        bond_types = []
        for i, bond_order in enumerate(bonds):
            bond_atoms = bonds[i]
            bond_order = bond_orders[i]
            #Check if aromatic bond possible
            #   both atoms need to be in aromatic rings
            if all([any([rings.ring_types[x] in ["AR1", "AR2"] 
                         for x in rings.ring_ids[bond_atom]]) 
                    for bond_atom in bond_atoms]):
                aromatic = True
            else:
                aromatic = False
            #check if delocalized bond possible
            #   one atom needs extended valence 
            #   and other needs to have only one connection
            if ((active_valence_state["con_val"][bond_atoms[0]][1] >= 5 and
                 active_valence_state["con_val"][bond_atoms[1]][0] == 1) or
                (active_valence_state["con_val"][bond_atoms[1]][1] >= 5 and
                 active_valence_state["con_val"][bond_atoms[0]][0] == 1)):
                delocalized = True
            else:
                delocalized = False
            #Assign bond type
            if bond_order == 3:
                bond_types.append("tb")
            elif bond_order == 2:
                if delocalized:
                    bond_types.append("DL")
                elif aromatic:
                    bond_types.append("db")
                else:
                    bond_types.append("DB")
            elif bond_order == 1:
                if aromatic:
                    bond_types.append("sb")
                else:
                    bond_types.append("SB")
        return bond_types
        
    def _get_atom_types(self, features):
        """
        Assigns atom types based on features by comparing them to atom type
        definitions stored in a 'ATOMTYPE_GFF.dat' file.
        """
        atom_type_definitions = self._get_atom_type_definitions()
        atom_types = []
        for i, atom_features in enumerate(features[:]):
            correct_atom_type = False
            for definition in atom_type_definitions:
                #print(definition[0])
                for f_idx, feature in enumerate(definition[2:]):
                    if feature == "&":
                        correct_atom_type = True
                        break
                    if feature == "*":
                        continue
                    #Check basic features
                    if 0 <= f_idx <= 3:
                        feature_correct = (feature == atom_features[f_idx])
                    #Check atomic properties
                    elif f_idx == 4:
                        properties = feature.strip("[]").split(",")
                        or_statements = [("." in prop) for prop in properties]
                        feature_correct = \
                            all([any([prop_2 in atom_features[f_idx] 
                                      for prop_2 in prop.split(".")]) 
                                 if or_statements[x] 
                                 else (prop in atom_features[f_idx]) 
                                 for x,prop in enumerate(properties)])
                        #print(properties, atom_features[f_idx])
                        #print(f"   {feature_correct}")
                    #Check chemical environment
                    elif f_idx == 5:
                        feature_correct = \
                            self._check_paths(feature, atom_features[f_idx])
                        #print(f"   {feature_correct}")

                    if feature_correct:
                        continue
                    else:
                        break
                if correct_atom_type:
                    atom_types.append(definition[0])
                    break
                else:
                    continue
        return atom_types                    
    
    def _get_atom_type_definitions(self):
        atom_definitions = []
        atom_definition_file = 'ATOMTYPE_GFF.dat'
        with importlib.resources.open_text(
                'RingGen.ParaTools',atom_definition_file) as f:
            content = f.readlines()
        for line in content[14:247]:
            if not line.startswith("//"):
                atom_definitions.append(line.split()[1:])
        return atom_definitions
        
    def _check_paths(self, target_path, path_list):
        target_paths = self._expand_target_paths(target_path)
        path_matches = []
        for path_def in target_paths:
            matches = []
            for i, path_ato in enumerate(path_list):
                if self._check_path_match(path_def, path_ato):
                    matches.append(i)
            path_matches.append(matches)
        #print(path_matches)
        return self._check_matches(path_matches)
        
    def _check_matches(self, matches):
        if len(matches) == 1:
            if len(matches[0]) >= 1:
                return True
            else:
                return False
        
        matches_lengths = [len(x) for x in matches]
        order = np.argsort(matches_lengths)
        
        if matches_lengths[order[0]] == 1:
            new_matches = [x for i,x in enumerate(matches) if i != order[0]]
            for i, match in enumerate(new_matches):
                if matches[order[0]][0] in match:
                    new_matches[i].remove(matches[order[0]][0])
            if any([len(x) == 0 for x in new_matches]):
                return False
            else:
                return self._check_matches(new_matches)
        else:
            for guess in range(matches_lengths[order[0]]):
                new_matches = [x 
                               for i,x in enumerate(matches) 
                               if i != order[0]]
                for i, match in enumerate(new_matches):
                    if matches[order[0]][guess] in match:
                        new_matches[i].remove(matches[order[0]][guess])
                if any([len(x) == 0 for x in new_matches]):
                    return False
                else:
                    return self._check_matches(new_matches)
        
    def _expand_target_paths(self, path):
        target_path = path.strip("()")
        target_paths = [target_path]
        change = True
        while change:
            change = False
            if any(["," in x for x in target_paths]):
                for i, temp in enumerate(target_paths):
                    pos_com = [i for i, c in enumerate(temp) if c == ","]
                    pos_br1 = [i for i, c in enumerate(temp) if c == "("]
                    pos_br2 = [i for i, c in enumerate(temp) if c == ")"]
                    
                    for pos in pos_com:
                        nb_br1 = sum([x<pos for x in pos_br1])
                        nb_br2 = sum([x<pos for x in pos_br2])
                        
                        if nb_br1 == nb_br2:
                            change = True
                            target_paths.pop(i)
                            target_paths.extend([temp[:pos], temp[pos+1:]])
                            break
                        if nb_br1 == nb_br2 + 1:
                            target_paths.pop(i)
                            change = True
                            target_paths.extend(
                                [temp[:pos] + ")", 
                                 temp[:pos_br1[0]+1] + temp[pos+1:]])
                            break
                        if nb_br1 == nb_br2 + 2:
                            target_paths.pop(i)
                            change = True
                            target_paths.extend(
                                [temp[:pos] + ")", 
                                 temp[:pos_br1[1]+1] + temp[pos+1:]])
                            break
                    if change:
                        break
        for i,temp in enumerate(target_paths):
            target_paths[i] = temp.strip(")").split("(")
            
        return target_paths
        
    def _check_path_match(self, path_def, path_ato):
        wild_card_atoms = {"XX": ["C", "N", "O", "S", "P"],
                           "XA": ["S", "O"],
                           "XB": ["N", "P"],
                           "XC": ["F", "Cl", "Br", "I"],
                           "XD": ["S", "P"]}
        if len(path_def) != len(path_ato):
            return False
        #print(path_def, path_ato)
        for i,atom_definition in enumerate(path_def):
            #Get properties of atom in definition
            if "[" in atom_definition:
                if atom_definition.split("[")[0][-1].isnumeric():
                    atom_type_d = atom_definition.split("[")[0][:-1]
                    nb_connections_d = atom_definition.split("[")[0][-1]
                else:
                    atom_type_d = atom_definition.split("[")[0]
                    nb_connections_d = ""
                properties_d = \
                    atom_definition[atom_definition.find("[")+1:-1].split(",")
            else:
                if atom_definition[-1].isnumeric():
                    atom_type_d = atom_definition[:-1]
                    nb_connections_d = atom_definition[-1]
                else:
                    atom_type_d = atom_definition
                    nb_connections_d = ""
                properties_d = None
            #Get properties of atom in path to be checked
            atom_from_path = path_ato[i]
            atom_type_a = atom_from_path.split("[")[0][:-1]
            nb_connections_a = atom_from_path.split("[")[0][-1]
            properties_a = \
                atom_from_path[atom_from_path.find("[")+1:-1].split(",")

            #print(atom_type_d, nb_connections_d, properties_d)
            #print(atom_type_a, nb_connections_a, properties_a)
            
            if atom_type_d not in wild_card_atoms:
                if atom_type_d != atom_type_a:
                    return False
            else:
                if not any([f"{x}" == f"{atom_type_a}" 
                            for x in wild_card_atoms[atom_type_d]]):
                    return False
            if nb_connections_d != "" and nb_connections_d != nb_connections_a:
                return False
            if properties_d is not None:
                if not all([x in properties_a for x in properties_d]):
                   return False
        return True


    #Function is very wide, need to break up method into smaller functions
    def refine_atom_types(self, connectivity, bonds, bond_orders, rings):
        special_atoms = ["cc", "ce", "cg", "nc", "ne", "cp",
                         "cd", "cf", "ch", "nd", "nf", "cq"]
        group_types = {"cc":"g1", "cd":"g2",
                       "ce":"g1", "cf":"g2",
                       "cg":"g1", "ch":"g2",
                       "nc":"g1", "nd":"g2",
                       "ne":"g1", "nf":"g2",
                       "cp":"g1", "cq":"g2"}
        group_det_dict = {"cc":{"g1":"cc", "g2":"cd"}, 
                          "cd":{"g1":"cc", "g2":"cd"}, 
                          "ce":{"g1":"ce", "g2":"cf"}, 
                          "cf":{"g1":"ce", "g2":"cf"}, 
                          "cg":{"g1":"cg", "g2":"ch"}, 
                          "ch":{"g1":"cg", "g2":"ch"}, 
                          "nc":{"g1":"nc", "g2":"nd"}, 
                          "nd":{"g1":"nc", "g2":"nd"}, 
                          "ne":{"g1":"ne", "g2":"nf"}, 
                          "nf":{"g1":"ne", "g2":"nf"}, 
                          "cp":{"g1":"cp", "g2":"cq"}, 
                          "cq":{"g1":"cp", "g2":"cq"}} 

        checked_atoms = []
        for i, atom_type in enumerate(self.atom_types):
            if atom_type not in special_atoms:
                checked_atoms.append(i)
        next_atoms = []
        next_next_atoms = []
        while len(checked_atoms) < len(self.atom_types):
            for i, atom_type in enumerate(self.atom_types):
                if i in checked_atoms:
                    continue
                
                if next_atoms == [] and next_next_atoms != []:
                    next_atoms = next_next_atoms
                    next_next_atoms = []
                    
                if next_atoms == [] and next_next_atoms == []:
                    connected_atoms = np.where(connectivity[i] == 1)[0]
                    group_atoms = [x 
                                   for x in connected_atoms 
                                   if self.atom_types[x] in special_atoms]
                    next_atoms = [x 
                                  for x in group_atoms 
                                  if x not in checked_atoms]
                    if not any([x in checked_atoms for x in group_atoms]):
                        checked_atoms.append(i)
                    else:
                        reference_atom = \
                            np.intersect1d(checked_atoms, group_atoms)[0]
                        bond_idx = \
                            np.intersect1d(
                                np.where(bonds == i)[0], 
                                np.where(bonds == reference_atom)[0])[0]
                        if atom_type in ["cc", "ce", "cg", "ne", "nc"]:
                            if bond_orders[bond_idx] == 1:
                                group_type = \
                                   group_types[self.atom_types[reference_atom]]
                                self.atom_types[i] = \
                                   group_det_dict[atom_type][group_type]
                                checked_atoms.append(i)
                            else:
                                group_type = \
                                   group_types[self.atom_types[reference_atom]]
                                if group_type == "g1":
                                    group_type = "g2"
                                elif group_type == "g2":
                                    group_type = "g1"
                                self.atom_types[i] = \
                                   group_det_dict[atom_type][group_type]
                                checked_atoms.append(i)
                        elif atom_type in ["cp"]:
                            common_rings = \
                                np.intersect1d(rings.ring_ids[i],
                                               rings.ring_ids[reference_atom])
                            common_AR1_rings = any([rings.ring_types[x]=="AR1"
                                                    for x in common_rings])
                            if common_AR1_rings:
                                group_type = \
                                   group_types[self.atom_types[reference_atom]]
                                if group_type == "g1":
                                    group_type = "g2"
                                elif group_type == "g2":
                                    group_type = "g1"
                                self.atom_types[i] = \
                                   group_det_dict[atom_type][group_type]
                                checked_atoms.append(i)
                            else:
                                group_type = \
                                   group_types[self.atom_types[reference_atom]]
                                self.atom_types[i] = \
                                   group_det_dict[atom_type][group_type]
                                checked_atoms.append(i)
                            
                        
                elif i in next_atoms:
                    next_atoms.remove(i)
                    connected_atoms = np.where(connectivity[i] == 1)[0]
                    group_atoms = ([x 
                                    for x in connected_atoms 
                                    if self.atom_types[x] in special_atoms])
                    next_next_atoms.extend([x 
                                            for x in group_atoms 
                                            if (x not in checked_atoms and 
                                                x not in next_next_atoms)])
                    
                    reference_atom = \
                        np.intersect1d(checked_atoms, group_atoms)[0]
                    bond_idx = \
                        np.intersect1d(np.where(bonds == i)[0], 
                                       np.where(bonds == reference_atom)[0])[0]
                    if atom_type in ["cc", "ce", "cg", "ne", "nc"]:
                        if bond_orders[bond_idx] == 1:
                            group_type = \
                                group_types[self.atom_types[reference_atom]]
                            self.atom_types[i] = \
                                group_det_dict[atom_type][group_type]
                            checked_atoms.append(i)
                        else:
                            group_type = \
                                group_types[self.atom_types[reference_atom]]
                            if group_type == "g1":
                                group_type = "g2"
                            elif group_type == "g2":
                                group_type = "g1"
                            self.atom_types[i] = \
                                group_det_dict[atom_type][group_type]
                            checked_atoms.append(i)
                    elif atom_type in ["cp"]:
                        common_rings = \
                            np.intersect1d(rings.ring_ids[i],
                                           rings.ring_ids[reference_atom])
                        common_AR1_rings = any([rings.ring_types[x] == "AR1"
                                                for x in common_rings])
                        if common_AR1_rings:
                            group_type = \
                                group_types[self.atom_types[reference_atom]]
                            if group_type == "g1":
                                group_type = "g2"
                            elif group_type == "g2":
                                group_type = "g1"
                            self.atom_types[i] = \
                                group_det_dict[atom_type][group_type]
                            checked_atoms.append(i)
                        else:
                            group_type = \
                                group_types[self.atom_types[reference_atom]]
                            self.atom_types[i] = \
                                group_det_dict[atom_type][group_type]
                            checked_atoms.append(i)
