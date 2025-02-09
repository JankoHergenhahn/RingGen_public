import numpy as np
import re
from copy import deepcopy
import importlib

from .Structure import Structure, Coordinates, Topology, TopologyGenerator
from .CoordinatesGenerator import CoordinatesGenerator
from .Tools import find_sub_systems, get_COM

class PorphyrinStructure(Structure):
    """
    Structure with special operations for structures containing porphyrins.

    Attributes
    ----------
    name : str
    coordinates : RingGen.Coordinates
        Contains spatial coordinates of molecule.
    topology : RingGen.Topology
        Contains force field parameters of molecule.
    elements : list of str
        List of elements of the atoms.
    connectivity : NxN np.array
        Connectivity C of molecular graph. C[i,j] = 1 if nodes i and j are 
        connected and 0 otherwise. C[i,i] should be 0.
    
    """
    def __init__(self, name = "PorphyrinStructure"):
        """
        Function initilizes empty object
        No inputs, no returns.
        """
        super().__init__()
        self.side_chain = None
        
  #Main methods
    def define_pattern(self, 
                       input_string, 
                       side_chain = "H"):
        """
        Function to initialize structure for automatic generation
        
        Parameters
        ----------
        input_string : str
            String that lists names of fragments, separated by "-". If the last
            character is "-", the last fragment connects to the first and 
            the structure is cyclic.
        side_chain : str
            Side chain for 5,15 meso positions, needs to be present in default
            files: "H", "Phe", "tBu", "Oct", "THS". Default is "H"
        
        Returns self: Allows method chaining.
        """
        self.pattern, self.is_cyclic = \
            self._understand_input_string(input_string)
        self.side_chain = side_chain
        return self

    def build_topology(self, 
                       restrain_aro_dihedrals = False, 
                       restrain_dihedral_angles = [90, 90]):
        """
        Generate topology from pattern and fragments.
        
        Returns self: Allows method chaining.
        """
        # Adds additional building blocks to the fragment_dictionary depending
        # on the side chain
        fragments = self._prep_top_fragments(self.side_chain,
                                             self._default_topology_fragments,
                                             restrain_aro_dihedrals,
                                             restrain_dihedral_angles)
        #Combines certain building blocks together
        pattern = self._process_pattern(self.pattern, self.is_cyclic)
        
        connections, fused_connections = self._get_connections(
            pattern, self.is_cyclic, self._default_topology_fragments)
        self._construct_topology(pattern, connections,fragments)
        
        #Need to handle fused porphyrin connection separately
        fused_connections_res_ids = \
            self._get_fused_porphyrin_res_ids(pattern, 
                                              fused_connections, 
                                              fragments)
        self._add_fused_connections(fused_connections_res_ids)
        
        #Butadiyne-linked porphyrins need sihedral angle correction
        self._add_dihedral_angle_parameters()
        self._charge_correction(pattern)
        self.create_elements()
        return self
    
    def build_coordinates(self,**kwargs):
        """
        Generate topology from pattern and fragments.
        
        Returns self: Allows method chaining.
        """
        #Combines certain building blocks together
        if "custom_twist" in kwargs:
            pattern, kwargs["custom_twist"] = self._process_custom_twist(
                self.pattern, self.is_cyclic, kwargs["custom_twist"])
        else:
            pattern = self._process_pattern(self.pattern, self.is_cyclic)
        
        # Adds additional building blocks to the fragment_dictionary depending
        # on the side chain
        fragments = self._prep_coord_fragments(
            self.side_chain, self._default_coordinate_fragments)
        
        coordinates_generator = CoordinatesGenerator(pattern, 
                                                     self.is_cyclic, 
                                                     fragments)
        self.coordinates = coordinates_generator.build(**kwargs)
        return self
    
    def generate_topology_from_coordinates(self):
        """
        Function to generate topology from coordinates.
        
        Zn-Porphyrin motifs are assigned directly, the rest is parameterized
        with the GAFF force field and AM1-BCC charges.
        
        Returns self: Allows method chaining.
        """
        ZnPor_maps = self._find_ZnPor_structure(self.coordinates)
        sub_coordinates, connections = \
            self._find_sub_coordinates(self.coordinates, ZnPor_maps)
        sub_topologies = self._get_sub_topologies(
            sub_coordinates, self._default_topology_fragments)
        self._recover_topology(sub_coordinates, sub_topologies, connections)
        return self    
    
    
  #Local functions
    @staticmethod
    def _understand_input_string(inputString):
        """
        Processes input_string. Returns pattern (list of str) and if structure 
        is cyclic (bool).
        """
        if inputString[0] == "H":
            is_cyclic = False
        elif inputString[-1] in ["-", ":"]:
            is_cyclic = True
        else:
            is_cyclic = False
            
        #Get seperate array about blocks (list) and connections (string)
        pattern = re.split("-|:",inputString)
        connection_characters = re.sub('[^:-]', '', inputString)
        if pattern[-1] == '':
            pattern.pop(-1)

        #Get correct structure for each block
        FusedStructures={"none":"Por",
                         "right":"PorFr",
                         "left":"PorFl",
                         "double":"PorFd"}
        for i,block in enumerate(pattern):
            if block == "Por":
                connectivity = \
                    PorphyrinStructure._Por_type(connection_characters,i)
                pattern[i] = FusedStructures[connectivity]
              
        return pattern, is_cyclic

    @staticmethod
    def _Por_type(connection_characters, i):
        """
        Determines type of porphyrin (fused, meso-connected, mixed) based on 
        connection characters.
        """
        if len(connection_characters) <= i:
            return "none"
        if connection_characters[i]=='-' and connection_characters[i-1]=='-':
            return "none"
        if connection_characters[i]==':' and connection_characters[i-1]=='-':
            return "right"
        if connection_characters[i]=='-' and connection_characters[i-1]==':':
            return "left"
        if connection_characters[i]==':' and connection_characters[i-1]==':':
            return "double"

    
    @staticmethod
    def _process_pattern(pattern, is_cyclic):
        """
        Replaces two subsequent Alk with butadiyne fragment and
        a terminal H+Alk with a dedicated H_Alk fragment.
        """
        if is_cyclic:
            if pattern[0] == "Alk" and pattern[-1] == "Alk":
                pattern = pattern[1:] + ["Alk"]
        else:
            if pattern[-1] == "H" and pattern[-2] == "Alk":
                pattern = pattern[:-2] + ["Alk_H"]            
            if pattern[0] == "H" and pattern[1] == "Alk":
                pattern = ["H_Alk"] + pattern[2:]
                    
        counter = 0
        new_pattern = []
        while counter < len(pattern):
            if pattern[counter] == "Alk" and pattern[counter+1] == "Alk":
                new_pattern.append("Butadiyne")
                counter += 2
            else:
                new_pattern.append(pattern[counter])
                counter += 1
            if counter == len(pattern) - 1:
                new_pattern.append(pattern[counter])
                counter += 1
        return new_pattern

    @staticmethod
    def _process_custom_twist(pattern, is_cyclic, custom_twist):
        """
        Replaces two subsequent Alk with butadiyne fragment and
        a terminal H+Alk with a dedicated H_Alk fragment.
        """
        if is_cyclic:
            if pattern[0] == "Alk" and pattern[-1] == "Alk":
                pattern = pattern[1:] + ["Alk"]
                custom_twist = custom_twist[1:] + custom_twist[0]
        else:
            if pattern[-1] == "H" and pattern[-2] == "Alk":
                pattern = pattern[:-2] + ["Alk_H"]
                custom_twist = custom_twist[:-1]
            if pattern[0] == "H" and pattern[1] == "Alk":
                pattern = ["H_Alk"] + pattern[2:]
                custom_twist = custom_twist[1:]
                    
        counter = 0
        new_pattern = []
        new_custom_twist = []
        while counter < len(pattern):
            if pattern[counter] == "Alk" and pattern[counter+1] == "Alk":
                new_pattern.append("Butadiyne")
                new_custom_twist.append(custom_twist[counter])
                counter += 2
            else:
                new_pattern.append(pattern[counter])
                new_custom_twist.append(custom_twist[counter])
                counter += 1
            if counter == len(pattern) - 1:
                new_pattern.append(pattern[counter])
                new_custom_twist.append(custom_twist[counter])
                counter += 1
        return new_pattern, new_custom_twist
    
    @staticmethod
    def _prep_top_fragments(side_chain, 
                            default_fragments, 
                            restrain_aro_dihedrals=False, 
                            restrain_dihedral_angles = [90, 90]):
        """
        Creates topology fragments with correct sidechains and with variations 
        of connections
        
        Porphyrin topology fragments are build at run time to allow for more 
        flexibility in available side chains and to enable restricting the 
        dihedral angle to phenyl sidechains.
        """
        fragments = deepcopy(default_fragments)
        moleculetypes = default_fragments.moleculetypes
        sc_name = f"side_chain_{side_chain}"
        # construct topology for POR fragment 
        # (and POR fragments for different connectivities)
        for top in ["Por","PorFr","PorFl","PorFd"]:
            connections = [[0, 1, [moleculetypes[top].con_atoms[2],
                                   moleculetypes[sc_name].con_atoms[0]]], 
                           [0, 2, [moleculetypes[top].con_atoms[3],
                                   moleculetypes[sc_name].con_atoms[0]]]]
            
            temp_top = Structure()
            temp_top._construct_topology([top, 
                                          f"side_chain_{side_chain}", 
                                          f"side_chain_{side_chain}"],
                                         connections,
                                         default_fragments)
            temp_top.topology.set_name(f"{top}")
            fragments.add_moleculetype(temp_top.topology)
        return fragments

    @staticmethod
    def _prep_coord_fragments(side_chain, default_fragments):
        """
        Creates coordinate fragments with correct sidechains and with 
        variations of connections
        """
        fragments = deepcopy(default_fragments)
        if side_chain == "H":
            side_chain_coords = fragments["H"]
        else:
            side_chain_coords = fragments[f"sidechain_{side_chain}"]
        
        side_chain_coords_1 = deepcopy(side_chain_coords)
        side_chain_coords_1.rotate([0,1,0], -np.pi/2)
        side_chain_coords_1.rotate([0,0,1], np.pi/2)
        side_chain_coords_1.translate([3.42, 0, -4.8])
        
        side_chain_coords_2 = deepcopy(side_chain_coords)
        side_chain_coords_2.rotate([0,1,0], +np.pi/2)
        side_chain_coords_2.rotate([0,0,1], np.pi/2)
        side_chain_coords_2.translate([3.42, 0, +4.8])
        
        Por_coords = fragments["Por_raw"]
        Por_coords.add(side_chain_coords_1)
        Por_coords.add(side_chain_coords_2)
        
        for name, del_ind in zip(["Por", "PorFr", "PorFl", "PorFd"],
                                 [[], [11, 16], [2,25], [2,11,16,25]]):
            temp_coords = deepcopy(Por_coords)
            temp_coords.remove_coordinates(del_ind)
            temp_coords.connecting_atoms = [x - sum([y<x for y in del_ind])
                                        for x in temp_coords.connecting_atoms] 
            fragments.update({name: temp_coords})
        
        return fragments
    
    @staticmethod
    def _get_connections(pattern, is_cyclic, default_topology_fragments):
        """
        Creates connection list that described which atoms of adjacent 
        fragments are connected and which adjacent fragments need a edge-fused
        porphyrin connection.

        Parameters
        ----------
        pattern : list of str
            Contains names of building blocks.
        is_cyclic : bool
            Determines if last fragment is connected to first.
        default_topology_fragments : dict
            Contains topologies of building blocks.

        Returns
        -------
        connections : list of lists
            Lists the connections and connecting atoms. Each list entry is a 
            list with the form:
                [idx of res 1, idx of res 2, [atom in res 1, atom in res 2]].
        fused_connections : list of list of int
            List of pairs of fragment indices that need fused connection.
        """
        connections = []
        fused_connections = []
        for i, (frag1, frag2) in enumerate(zip(pattern[:-1], pattern[1:])):
            if PorphyrinStructure._is_fused_con(frag1, frag2): 
                fused_connections.append([i, i+1])
            else:
                linking_atoms = \
                    Structure._get_linking_atoms(frag1, 
                                                 frag2, 
                                                 default_topology_fragments)
                connections.append([i, i+1, linking_atoms])
        if is_cyclic:
            if PorphyrinStructure._is_fused_con(pattern[-1], pattern[0]): 
                fused_connections.append([len(pattern)-1, 0])
            else:
                linking_atoms = \
                    Structure._get_linking_atoms(pattern[-1], 
                                                 pattern[0], 
                                                 default_topology_fragments)
                connections.append([len(pattern)-1, 0, linking_atoms])
        return connections, fused_connections
    
    @staticmethod
    def _is_fused_con(frag1, frag2):
        """
        Determines if the connection between frag1 and frag2 needs edge-fused 
        linker (based on types of connected porphyrins).
        """
        if ((frag1 == "PorFr" and frag2 == "PorFd") or
            (frag1 == "PorFr" and frag2 == "PorFl") or
            (frag1 == "PorFd" and frag2 == "PorFd") or
            (frag1 == "PorFd" and frag2 == "PorFl")):
            return True
        else:
            return False
        
    @staticmethod
    def _get_fused_porphyrin_res_ids(pattern, fused_connections, fragments):
        """
        Finds pairs of residue indices that need fused connections.
        
        Need to convert from list of fragments that are fused to list of
        residues that are fused (fragment may contain more than one residue).
        
        Parameters
        ----------
        pattern : list of str
            List of names of residues.
        fused_connections : list of lists of int
            Each list entry is a list with two fragment indices.
        fragments : TopFile
            Contains topology fragments.

        Returns
        -------
        fused_connections_res_ids : list of lists of int
            Each list entry is a list with two residue indices.
        """
        #determine the number of residues in a fragment based on larges res_id
        fused_connections_res_ids = []
        fragment_lengths = []
        for fragment in pattern:
            fragment_lengths.append(
                fragments.moleculetypes[fragment].atoms[-1][2])
        
        # convert frag indices to res indices
        for fused_connection in fused_connections:
            res_id_1 = sum(fragment_lengths[:fused_connection[0]]) + 1
            res_id_2 = sum(fragment_lengths[:fused_connection[1]]) + 1
            fused_connections_res_ids.append([res_id_1, res_id_2])
            
        return fused_connections_res_ids
        
    
    def _add_fused_connections(self, fused_connections_res_ids):
        """
        Adds edge-fused linker topologies to the residue pairs specified in
        fused_connections_res_ids.
        """
        moleculetypes = self._default_topology_fragments.moleculetypes
        for res_id_1, res_id_2 in fused_connections_res_ids:
            atom_ind = []
            #determine which sides of porphyrins are facing each other
            atom_names = self._get_fuse_atom_names(res_id_1, res_id_2)
            for res_id,atom_name in zip([res_id_1]*3 + [res_id_2]*3,
                                        atom_names):
                atom_ind.append(
                    self.topology.find_atoms(atom_name = atom_name, 
                                             res_id = res_id)[0][0] - 1)
            
            # adjust the connectivity matrix to include the 3 new bonds
            # necessary for the _add_linker method
            connectivity = deepcopy(self.connectivity)
            for i in range(3):
                connectivity[atom_ind[i], atom_ind[i]+3] = 1
                connectivity[atom_ind[i+3], atom_ind[i]] = 1

            
            linker = moleculetypes["LINKER_POR_POR_fused"]
            self.topology = self._add_linker(self.topology, 
                                             connectivity, 
                                             [atom_ind[1], atom_ind[4]], 
                                             linker)
        
    def _get_fuse_atom_names(self, res_id_1, res_id_2):
        """
        Determine which sides of porphyrins are facing each other based on how
        close they are in the rough coordinates that were generated.
        """
        atom_ind = []
        for res_id,atom_name in zip([res_id_1]*4 + [res_id_2]*4,
                                    ["C05", "C10", "C15", "C20"]*2):
            atom_ind.append(
                self.topology.find_atoms(atom_name = atom_name, 
                                         res_id = res_id)[0][0] - 1)
        
        if len(self.coordinates.coordinates) == 0:
            #default is C10 to C20
            return ["C08", "C10", "C12", "C02", "C20", "C18"]
        else:
            atom_pos = self.coordinates.coordinates[atom_ind]
            COM_1 = get_COM(atom_pos[:4])
            COM_2 = get_COM(atom_pos[4:])
            
            Cmeso_1 = np.linalg.norm(atom_pos[:4] - COM_2, axis = 1).argmin()
            Cmeso_2 = np.linalg.norm(atom_pos[4:] - COM_1, axis = 1).argmin()
        
        indices_dict = {0:["C03", "C05", "C07"],
                        1:["C08", "C10", "C12"],
                        2:["C13", "C15", "C17"],
                        3:["C02", "C20", "C18"]}
        
        
        idx_A1 = self.topology.find_atoms(atom_name = indices_dict[Cmeso_1][0], 
                                          res_id = res_id_1)[0][0] - 1
        idx_A2 = self.topology.find_atoms(atom_name = indices_dict[Cmeso_1][2], 
                                          res_id = res_id_1)[0][0] - 1
        idx_B1 = self.topology.find_atoms(atom_name = indices_dict[Cmeso_2][0], 
                                          res_id = res_id_2)[0][0] - 1
        idx_B2 = self.topology.find_atoms(atom_name = indices_dict[Cmeso_2][2], 
                                          res_id = res_id_2)[0][0] - 1
        
        coord_A1 = self.coordinates.coordinates[idx_A1]
        coord_A2 = self.coordinates.coordinates[idx_A2]
        coord_B1 = self.coordinates.coordinates[idx_B1]
        coord_B2 = self.coordinates.coordinates[idx_B2]
        
        if ((np.linalg.norm(coord_A1 - coord_B1) 
             + np.linalg.norm(coord_A2 - coord_B2)) 
           <(np.linalg.norm(coord_A1 - coord_B2) 
             + np.linalg.norm(coord_A2 - coord_B1))):
            return indices_dict[Cmeso_1] + indices_dict[Cmeso_2]
        else:
            return indices_dict[Cmeso_1]+list(reversed(indices_dict[Cmeso_2])) 
    
    
    
    
    def _charge_correction(self, pattern):
        """
        Sets partial charges of porphyrin units to values obtained from 
        QM calculations and adjusts the partial of the remaining residues
        accoridngly by distributing the excess charge over them.
        """
        self.create_connectivity()
        if not any(["Por" in x for x in pattern]):
            return 0
        
        POR_charges = \
        {'C01': 0.142003, 'C02':-0.090229, 'H02': 0.149241, 'C03':-0.315665, 
         'H03': 0.169634, 'C04': 0.383305, 'C05':-0.473016, 'C06': 0.383305, 
         'C07':-0.315665, 'H07': 0.169634, 'C08':-0.090229, 'H08': 0.149241, 
         'C09': 0.142003, 'C10': 0.103492, 'C11': 0.142003, 'C12':-0.090229, 
         'H12': 0.149241, 'C13':-0.315665, 'H13': 0.169634, 'C14': 0.383305, 
         'C15':-0.473016, 'C16': 0.383305, 'C17':-0.315665, 'H17': 0.169634, 
         'C18':-0.090229, 'H18': 0.149241, 'C19': 0.142003, 'C20': 0.103492,
         'N21':-0.57875, 'N22':-0.57875, 'N23':-0.57875, 'N24':-0.57875}
        ALK_charges = \
        {"attached_to_POR": -0.198167, 
         "not_attached_to_POR":-0.002727}
        
        top = deepcopy(self.topology)
        #cycle through porphyrins and apply reference charges
        Por_residues = list(set([x[2] for x in top.atoms if x[3] == "POR"]))
        Por_residues.sort()
        assigned_atoms = []
        for i,atom in enumerate(top.atoms):
            if atom[3] == "POR":
                charge = POR_charges[atom[4]]
                #add charge of hydrogen to carbon if it is a fused porphyrin
                if top.atoms[i][4] in ["C02", "C03", "C07", "C08", 
                                       "C12", "C13", "C17", "C18"]:
                    H_name = f"H{top.atoms[i][4][1:]}"
                    has_H = any([True for x in top.atoms
                                if x[2] == top.atoms[i][2] and x[4] == H_name])
                    if not has_H:
                        charge = np.round(charge + POR_charges[H_name], 6)
                top.atoms[i][6] = charge
                assigned_atoms.append(i)
            elif atom[3] == "ALK":
                connected_atoms = np.where(self.connectivity[i] == 1)[0]
                if any([top.atoms[x][3] == "POR" for x in connected_atoms]):
                    top.atoms[i][6] = ALK_charges["attached_to_POR"]
                else:
                    top.atoms[i][6] = ALK_charges["not_attached_to_POR"]
                assigned_atoms.append(i)
        #change charges for terminal Alk-H groups
        for i,atom in enumerate(top.atoms):
            if atom[3] != "HTE":
                continue
            connected_atom_1 = np.where(self.connectivity[i] == 1)[0][0]
            if top.atoms[connected_atom_1][3] != "ALK":
                continue
            connected_atom_2 = \
                [x 
                 for x in np.where(self.connectivity[connected_atom_1] == 1)[0]
                 if x not in [i, connected_atom_1]][0]
            connected_atom_3 = \
                [x 
                 for x in np.where(self.connectivity[connected_atom_2] == 1)[0]
                 if x not in [i, connected_atom_1, connected_atom_2]][0]
            top.atoms[i][6] = 0.306794
            top.atoms[connected_atom_1][6] = -0.417549
            top.atoms[connected_atom_2][6] = -0.023994
            top.atoms[connected_atom_3][6] = 0.037347
        
        #calculate excess_charge_per_atom
        excess_charge = sum([x[6] for x in top.atoms])
        ecpa = excess_charge/(len(top.atoms)-len(assigned_atoms))
        for i,_ in enumerate(top.atoms):
            if i not in assigned_atoms:
                top.atoms[i][6] = np.round(top.atoms[i][6] - ecpa,6)
        top.atoms[0][6] = np.round(top.atoms[0][6] 
                                   - sum([x[6] for x in top.atoms]),6)
        self.topology = top
        
        return self
    
    def _add_dihedral_angle_parameters(self):
        """
        Adds additional force field parameters to butadiyne-linked porphyrins
        to ensure they favour co-planar geometries.
        """
        dihedrals = [["C09", "C10", "C20", "C01"],
                     ["C09", "C10", "C20", "C19"],
                     ["C11", "C10", "C20", "C01"],
                     ["C11", "C10", "C20", "C19"]]
        resids = list(set([x[2] for x in self.topology.atoms]))
        
        #
        res_list = []
        last_res_idx = 0
        for atom in self.topology.atoms:
            if atom[2] != last_res_idx:
                res_list.append(atom[3])
                last_res_idx = atom[2]
                
        #
        for res in resids:
            if res_list[res-1] != "POR":
                continue
            C10_idx = [i for i,x in enumerate(self.topology.atoms)
                       if x[2] == res and x[4] == "C10"][0]
            con_atoms = [x 
                         for x in np.where(self.connectivity[C10_idx] == 1)[0]
                         if self.topology.atoms[x][2] != res]
            if len(con_atoms) > 0:
                con_atom = con_atoms[0]
            else:
                continue
            nb_atoms = len(self.topology.atoms)
            if (self.topology.atoms[(con_atom+0)%nb_atoms][3] == "ALK" and
                self.topology.atoms[(con_atom+1)%nb_atoms][3] == "ALK" and
                self.topology.atoms[(con_atom+2)%nb_atoms][3] == "ALK" and
                self.topology.atoms[(con_atom+3)%nb_atoms][3] == "ALK" and
                self.topology.atoms[(con_atom+4)%nb_atoms][3] == "POR"):
                res2 = self.topology.atoms[(con_atom+4)%nb_atoms][2]
                for dihedral in dihedrals:
                    dihedral_indeces = []
                    for atom_id, res_id in zip(dihedral, 
                                               [res, res, res2, res2]):
                        for atom in self.topology.atoms:
                            if (atom[2] == res_id and atom[4] == atom_id):
                                dihedral_indeces.append(atom[0])
                    #print("Added dihedral angle")
                    self.topology.dihedrals.append(dihedral_indeces 
                                        +[1, 180.00008, 0.6, 2.0])
            
    @staticmethod
    def _find_ZnPor_structure(self, coordinates):
        """
        Finds patterns in the connectivity of the coordinates that match the 
        connectivity of a metallo porphyrin.
        """
        with importlib.resources.open_text('RingGen.DefaultFiles',
                                           'Por_pattern.txt') as f:
            Por_pattern = f.readlines()
        Por_pattern_connectivity = np.array([x.split() 
                                             for x in Por_pattern[1:]], int)
        Por_pattern_elements = list(Por_pattern[0].split())
        
        ZnPor_maps = []
        if "Zn" in coordinates.elements:
            zn_indices = [i for i,x in enumerate(coordinates.elements)
                          if x == "Zn"]
            for zn_idx in zn_indices:
                matches = \
                Structure.match_patterns(Por_pattern_connectivity, 
                                         Por_pattern_elements,
                                         coordinates.connectivity, 
                                         coordinates.elements,
                                         matches = {32: zn_idx})
                ZnPor_maps.append(matches)
        return ZnPor_maps

    @staticmethod
    def _find_sub_coordinates(coordinates, ZnPor_maps):
        """
        Separated the coordinates into fragments by removing sections that are
        described by zn-porphyrin topology.
        
        Returns list of  fragments (list of RingGen.Coordinates) and list of
        connections (list of pairs of atom indices)
        """
        #create list of indices that are part of Zn porphyrins
        ii = []
        for ZnPor_map in ZnPor_maps:
            ii.extend(ZnPor_map.values())
        
        #create a connectivity matrix where connections to Zn porphyrins
        #have been deleted
        broken_connectivity = deepcopy(coordinates.connectivity)
        connections_by_atom_idx = []
        for i, con in enumerate(coordinates.connectivity):
            for j in np.where(con)[0]:
                if (i in ii and j not in ii) or (i not in ii and j in ii):
                    broken_connectivity[i, j] = 0
                    connections_by_atom_idx.append([i,j])
        sub_systems = find_sub_systems(broken_connectivity)
        
        #create a list of coordinates of the subsystems created by cutting 
        #bonds to zinc porphyrins
        sub_coordinates = []
        for sub_system in sub_systems:
            if "Zn" in np.array(coordinates.elements)[sub_system]:
                temp = Coordinates()
                temp.set(list(np.array(coordinates.elements)[sub_system]), 
                         coordinates.coordinates[sub_system])
            elif len(sub_system) == 1:
                idx = sub_system[0]
                temp = Coordinates()
                temp.set([coordinates.elements[idx]],
                         np.array([coordinates.coordinates[idx]]))
                temp.con_atoms = [0]
            else:
                connecting_points = []
                for i, idx in enumerate(sub_system):
                    con_atoms = np.where(coordinates.connectivity[idx])[0]
                    if any([x not in sub_system for x in con_atoms]):
                        connecting_points.append(i)
    
                temp = PorphyrinStructure._get_sub_coordinates(
                    coordinates, sub_system, connecting_points)
                temp.con_atoms = connecting_points
            sub_coordinates.append(temp)
                
        connections = PorphyrinStructure._get_ZnPor_connections(
            connections_by_atom_idx, sub_systems)
        
        return sub_coordinates, connections

    @staticmethod
    def _get_ZnPor_connections(connections_by_atom_idx, sub_systems):
        """
        Translates the absolute atom indices to atom indices inside the 
        fragmentes subsystems. 
        
        Returns connections (list of lists) which lists the connections 
        and connecting atoms. Each list entry is a list with the form:
                [idx of res 1, idx of res 2, [atom in res 1, atom in res 2]].
        """
        connections = []
        for i, cbai in enumerate(connections_by_atom_idx):
            for j, sub_system in enumerate(sub_systems):
                if cbai[0] in sub_system:
                    ss_1 = j 
                    idx_1 = np.where(np.array(sub_systems[j]) == cbai[0])[0][0]
                if cbai[1] in sub_system:
                    ss_2 = j 
                    idx_2 = np.where(np.array(sub_systems[j]) == cbai[1])[0][0]
            if not any([(x[0] == ss_1 and x[1] == ss_2) or 
                        (x[0] == ss_2 and x[1] == ss_1)
                        for x in connections]):
                connections.append([ss_1, ss_2, [idx_1, idx_2]])
        return connections
    
    @staticmethod
    def _get_sub_coordinates(coordinates, subsystem, connecting_points):
        sub_coordinates = Coordinates()
        sub_coordinates.set(list(np.array(coordinates.elements)[subsystem]), 
                            coordinates.coordinates[subsystem])
        sub_coordinates.create_connectivity()
        
        #add hydrogen where bonds where cut
        for atom in connecting_points:
            connected_atoms = np.where(sub_coordinates.connectivity[atom])[0]
            c1 = sub_coordinates.coordinates[atom]
            c2 = get_COM(sub_coordinates.coordinates[connected_atoms])
            vector = c1 - c2
            vector /= np.linalg.norm(vector)
            new_H = Coordinates().set(["H"], c1 + vector *1.09)
            sub_coordinates.add(new_H)
        return sub_coordinates
    
    @staticmethod
    def _get_sub_topologies(sub_coordinates, fragments):
        """
        Generates topologies for each subsystem and returns a list of 
        topologies (list of RingGen.Topology).
        """
        sub_topologies = []
        for coords in sub_coordinates:
            if "Zn" in coords.elements:
                temp = \
                deepcopy(fragments.moleculetypes["Por"])
                sub_topologies.append(temp)
            elif len(coords.elements) == 1:
                temp = Topology()
                temp.atoms =  [[1, 'ha', 1, 'HTE', 'H01', 1, 0.2009, 1.008]]
                sub_topologies.append(temp)                
            else:
                generator = TopologyGenerator()
                generator.generate_topology(coords, input_type = "Coordinates")
                temp = generator.get_topology()
                for atom in coords.con_atoms:
                    if temp.atoms[atom][1] == "c1":
                        temp.atoms[atom][1] = "ch"
                
                for i,_ in enumerate(coords.con_atoms):
                    temp.atoms[-(1+i)][2] = 2
                temp.reduce([1])
                
                sub_topologies.append(temp)
        return sub_topologies
           
    def _recover_topology(self, sub_topologies, connections):
        """
        Uses sub_topologies (list of RingGen.Topology) and connections (list
        of lists) to build the overall topology.

        No return value.
        """
        fragments_dict = deepcopy(self._default_topology_fragments)
        for i, top in enumerate(sub_topologies):
            top.set_name(f"{i}")
            fragments_dict.add_moleculetype(top)
        
        # _construct_topology sets the topology to self automatically
        self._construct_topology([f"{i}" for i,_ in enumerate(sub_topologies)],
                                 connections,
                                 fragments_dict)