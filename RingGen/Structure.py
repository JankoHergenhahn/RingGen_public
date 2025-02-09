import numpy as np
from copy import deepcopy

from .Coordinates import Coordinates
from .Topology import Topology
from .CoordinatesGenerator import CoordinatesGenerator
from .TopologyGenerator import TopologyGenerator
from .LinkerGenerator import LinkerGenerator
try:
    from .Minimizer import Minimizer
except:
    from .Minimizer_SD import Minimizer  
    
class Structure():
    """
    Structure object for managing and using a topology and coordinate objects.

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
    def __init__(self, name="MOL"):
        self.name = name
        self.topology = Topology()
        self.coordinates = Coordinates()
        self.elements = []
        self.connectivity = np.empty((0,0))

        #Internal
        self._pattern = []
        self._default_topology_fragments = None
        self._default_coordinate_fragments = None
        self._atom_key = {}

  #methods to set or update properties
    def set_name(self, name):
        """
        Sets name of structure and topology.
        
        Returns self: Allows method chaining.
        """
        self.name = name
        self.topology.name = name
        return self

    def set_coordinates(self, coordinates):
        """
        Sets coordinates from file (str) or coordinates (RingGen.Coordinates).
        
        Returns self: Allows method chaining.
        """
        if isinstance(coordinates, Coordinates):
            self.coordinates = coordinates
        else:
            self.coordinates = Coordinates().read_file(coordinates)
        return self
    
    def set_topology(self, topology):
        """
        Sets topology from file (str) or topology (RingGen.Topology).
        
        Returns self: Allows method chaining.
        """
        if isinstance(topology,Topology):
            self.topology = topology
        else:
            self.topology = Topology().read_file(topology)
        return self
    
    def set_defaults(self, defaults):
        """
        Defines defaults for automatic coordinate and topology generators.
        
        Parameters
        ----------
        defaults : Default object
            Needs have attributes default_topology_fragments and 
            default_coordinate_fragments.

        Returns
        -------
        self: Allows method chaining.
        """
        self._default_topology_fragments = defaults.topology_fragments
        self._default_coordinate_fragments = defaults.coordinate_fragments
        return self

    def set_topology_defaults(self, default_topology_fragments):
        """
        Defines default_topology_fragments (dict) for topology generation.
        
        Returns self: Allows method chaining.
        """
        self._default_topology_fragments = default_topology_fragments
        return self

    def set_coordinates_defaults(self, default_coordinate_fragments={}):
        """
        Defines default_coordinate_fragments (dict) for coordinates generation.
        
        Returns self: Allows method chaining.
        """
        self._default_coordinate_fragments = default_coordinate_fragments
        return self

  # Main methods
    def define_pattern(self, input_string):
        """
        Function to initialize structure for automatic generation
        
        Parameters
        ----------
        input_string : str
            String that lists names of fragments, separated by "-". If the last
            character is "-", the last fragment connects to the first and 
            the structure is cyclic.
        
        Returns self: Allows method chaining.
        """
        self._pattern, self.is_cyclic = \
            self._understand_input_string(input_string)
        return self
    
    def build(self):
        """
        Generates coordinates, topology and minimizes structure.
        
        Returns self: Allows method chaining.
        """
        self.build_coordinates()
        self.build_topology()
        self.minimize()
        return self
    
    def build_topology(self):
        """
        Generate topology from pattern and fragments.
        
        Returns self: Allows method chaining.
        """
        connections = self._get_connections(self._pattern,
                                            self.is_cyclic,
                                            self._default_topology_fragments)
        self._construct_topology(self._pattern, 
                                 connections,
                                 self._default_topology_fragments)
        self.create_elements()
        return self
    
    def build_coordinates(self, **kwargs):
        """
        Generate rough coordinates from pattern and fragments.
        
        Parameters
        ----------
        custom_twist : list, optional
            Determines the relative rotation of the coordinate fragments. Needs
            to be same length as pattern. If not specified, 
            rotations will be determined based on a set of rules.
        end_to_end_distance : float, optional
            Adds a curvature to linear chains. Terminal fragments are placed
            "end_to_end_distance" away from each other.
        radius : float, optional
            Adds a curvature to linear chains. Chain is warped around a 
            cylinder with radius "radius".
        
        Returns self: Allows method chaining.
        """
        coordinates_generator  = CoordinatesGenerator(
            self._pattern, self.is_cyclic, self._default_coordinate_fragments)
        coordinates_generator.build(**kwargs)
        self.coordinates = coordinates_generator.get_coordinates()
        return self

    def generate_topology_from_coordinates(self, easy_valence=False):
        """
        Function to generate topology from coordinates.
        
        This function can be used to parameterize a molecule with the GAFF
        force field and AM1-BCC charges.
        
        Parameters
        ----------
        easy_valence : bool, optional
            Should be True if there are no formal charges or delocalized bonds,
            speeds up parameterization of large structures.
        
        Returns self: Allows method chaining.
        """
        topology_generator = TopologyGenerator()
        topology_generator.generate_topology(self.coordinates, 
                                             input_type="Coordinates",
                                             easy_valence = easy_valence)
        self.topology = topology_generator.get_topology()
        return self

    def create_connectivity(self):
        """
        Creates connectivity C matrix based on bonds in topology.
        
        C[i,j] = 1 if nodes i and j are connected and 0 otherwise. 
        C[i,j] is 0.
        
        Returns self: Allows method chaining.
        """
        self.connectivity = self._create_connectivity(self.topology)
        return self

    def create_elements(self):
        """
        Function to create elements list from topology
        
        Returns self: Allows method chaining.
        """
        elements = []
        for atom in self.topology.atoms:
            if atom[4].startswith("H"):
                elements.append("H")
            elif atom[4].startswith("C"):
                elements.append("C")
            elif atom[4].startswith("N"):
                elements.append("N")
            elif atom[4].startswith("O"):
                elements.append("O")
            elif atom[4].startswith("Z"):
                elements.append("Zn")
            elif atom[4].startswith("Si"):
                elements.append("Si")
            else:
                elements.append("X")
                print("Found unknown atom type.")
        self.elements=elements
        return self

    def minimize(self, steps = 1000):
        """
        MM-Optimizes the geometry in coordinates using force field in topology.
        
        Returns self: Allows method chaining.
        """
        my_minimizer = Minimizer(self.topology, self.coordinates)
        my_minimizer.minimize(steps = steps)
        self.coordinates.set_coordinates(my_minimizer.get_coordinates())
        return self
    
    def write_coordinates(self, file_name = "out", file_format = "xyz"):
        """
        Writes coordinate file.
        
        Parameters
        ----------
        file_name : str
        fiel_format : str
            Only "xyz" or "pdb" supported.
        
        Returns self: Allows method chaining.
        """
        file_name = file_name.split(".")[0]
        full_name = f"{file_name}.{file_format}"
        if file_format == "xyz":
            self.coordinates.write_xyz(output_name = full_name)
        elif file_format == "pdb":
            self.coordinates.write_pdb(output_name = full_name,
                                       topology = self.topology)
        return self

    def reorder_coordinates(self, first_atom_index = 0):
        """
        Changes order of coordinates to match the topology. 
        
        Useful when preparing structures for MD simulations where coordinates
        need to match the topology.

        Parameters
        ----------
        first_atom_index : int, optional
            Index of atom in coordinates that matches the first atom in the
            topology. If set to -1, algorithm will try to find the matches but
            might take longer. The default is 0.

        Returns self: Allows method chaining.
        """
        #Ensure that required patterns are defined
        if self.elements == []:
            self.create_elements()
        if len(self.connectivity) == 0:            
            self.create_connectivity()
        if len(self.coordinates.connectivity) == 0:            
            self.coordinates.create_connectivity()
            
        #Run matching algorithm
        if first_atom_index == -1:
            starting_match = {}
        else:
            starting_match = {0: first_atom_index}
        matches = self.match_patterns(self.connectivity, 
                                      self.elements, 
                                      self.coordinates.connectivity, 
                                      self.coordinates.elements,
                                      matches = starting_match)
        #Process matches output
        if matches is None:
            print("Could not match coordinates to topology.")
            return
        if len(matches) < len(self.elements):
            print("Could not match coordinates to topology.",
                  f"Only {len(matches)} out of {len(self.atoms)} were matched")
            return
        ordered_coordinates=[0]*len(self.elements)
        ordered_elements=[0]*len(self.elements)
        for i in matches:
            ordered_coordinates[i]=self.coordinates.coordinates[matches[i]]
            ordered_elements[i]=self.coordinates.elements[matches[i]]
        ordered_coordinates = np.array(ordered_coordinates)
        self.coordinates.set_coordinates(ordered_coordinates)
        self.coordinates.set_elements(ordered_elements)
        self._atom_key = matches
        
        return self
            
    def match_patterns(self, con1, ele1, con2, ele2, known_matches = {}):
        """
        Determines mapping of one object to another.
        
        The two objects can be a topology and a coordinate object or two of 
        the same type as long as a connectivity matrix and list of elements
        exist.
        
        Parameters
        ----------
        con1 : NxN np.array
            Connectivity matrix of object 1.
        ele1 : list of str (with length N)
            Elements of object 1.
        con2 : MxM np.array
            Connectivity matrix of object 2.
        ele2 : list of str (with length M)
            Elements of object 2.
        known_matches : dict, optional
            Known mappings from con1 to con2. The default is {}.

        Returns
        -------
        Atom_mapping : dict
            Atom mapping with {atom_in_object_1: atom_in_object_2}.
        """
        con1 = deepcopy(con1)
        con2 = deepcopy(con2)
        
        ele_dict = {x:2+i for i, x in enumerate(set(ele1 + ele2))}
        for i, e in enumerate(ele1):
            con1[i,i] = ele_dict[e]
        
        for i, e in enumerate(ele2):
            con2[i,i] = ele_dict[e]
        
        return self._match_networks(con1, con2, known_matches)

  #Local functions
    def _construct_topology(self, pattern, connections, fragments_dictionary):
        """
        Generates topologies based on fragments.
        
        First, overall topology is created by concatenating all fragments, but
        with no connetions. Afterwards, connections are added between the
        fradments using default linker topology fragments or by creating a
        linker from scratch using GAFF parameters and the existing atom types
        of the fragments.

        Parameters
        ----------
        pattern : list of str
            List of the names of topology fragments.
        connections : List of lists
            Lists the connections with indices of residues (i,j) and the atoms
            that form the connecting bonds (a_i, a_j)..Each entry is of the
            form: [i, j, [a_i, a_j]].
        fragments_dictionary : dict
            Contains topologies of the fragments and linkers between fragments. 
            Has the form {name: RingGen.Topology}.

        Returns self: Allows method chaining.
        """
        counter = 0
        connections_corrected = deepcopy(connections)

        #Build Topology
        self.topology = Topology()
        for i, block in enumerate(pattern):
            fragment = fragments_dictionary.moleculetypes[block]
            self.topology = self._combine_topologies(self.topology, fragment)
            
            #correct connection atom numbers
            for j, connection in enumerate(connections_corrected):
                if connection[0] == i:
                    connections_corrected[j][2][0] += counter
                if connection[1] == i:
                    connections_corrected[j][2][1] += counter
            counter += len(fragment.atoms)
        
        #Add connections
        for connection in connections_corrected:
            connectivity = self._create_connectivity(self.topology)
            connectivity[connection[2][0], connection[2][1]] = 1
            connectivity[connection[2][1], connection[2][0]] = 1
            
            con_name, res_ord = self._get_con_name(self.topology, connection[2])
            
            if con_name in fragments_dictionary.moleculetypes.keys():
                self.topology = self._add_linker(self.topology, 
                                                 connectivity, 
                                                 np.array(connection[2])[res_ord], 
                                                 fragments_dictionary.moleculetypes[con_name])
            else:
                self.topology = LinkerGenerator().generate_linker_topology(self.topology, connectivity, connection[2])
            
        #Check Pairs
        self.topology = self._correct_pairs(self.topology)
        
        self.topology.set_name(self.name)
        self.create_elements()
        self.create_connectivity()
        return self    
    
    @staticmethod
    def _create_connectivity(topology):
        """
        Creates connectivity C matrix based on bonds in topology.
        
        C[i,j] = 1 if nodes i and j are connected and 0 otherwise. 
        C[i,j] is 0.
        
        Returns
        -------
        connectivity : NxN np.array
        """
        nb_atoms = len(topology.atoms)
        connectivity = np.zeros((nb_atoms, nb_atoms))
        for bond in topology.bonds:
            connectivity[bond[0]-1, bond[1]-1] = 1
            connectivity[bond[1]-1, bond[0]-1] = 1
        return connectivity
    
    @staticmethod
    def _understand_input_string(input_string):
        """
        Processes input_string. Returns patter (list of str) and if structure 
        is cyclic (bool).
        """
        if input_string[0] == "H":
            is_cyclic=False
        elif input_string[-1] == "-":
            is_cyclic=True
        else:
            is_cyclic=False
            
        pattern = input_string.split("-")
        if pattern[-1]=='':
            pattern.pop(-1)

        return pattern, is_cyclic

    @staticmethod
    def _get_connections(pattern, is_cyclic, default_topology_fragments):
        """
        Creates connection list that described which atoms of adjacent 
        fragments are connected.

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
        """
        connections = []
        for i, (frag1, frag2) in enumerate(zip(pattern[:-1], pattern[1:])):
            linking_atoms = Structure._get_linking_atoms(
                    frag1, frag2, default_topology_fragments)
            connections.append([i, i+1, linking_atoms])
        if is_cyclic:
            linking_atoms = Structure._get_linking_atoms(
                    pattern[-1], pattern[0], default_topology_fragments)
            connections.append([len(pattern)-1, 0, linking_atoms])
        return connections
    
    @staticmethod
    def _get_linking_atoms(frag1, frag2, default_topology_fragments):
        """
        Obtains the connecting atom indices from default topology information.
        """
        # should connect second connection point of frag1 to first connection
        # point of frag2 to form a chain, unless frag1 only has a single 
        # connection point, e.g. for terminal coordinate fragments such as "H"
        conec_1 = default_topology_fragments.moleculetypes[frag1].con_atoms
        if len(conec_1) == 1:
            atom_1 = conec_1[0]
        else:
            atom_1 = conec_1[1]
        conec_2 = default_topology_fragments.moleculetypes[frag2].con_atoms
        atom_2 = conec_2[0]
        
        return [atom_1, atom_2]
    
    @staticmethod
    def _correct_pairs(topology):
        """
        Corrects pair paramters which should only apply to atoms 3 bonds away.
        """
        pairs = deepcopy([x[:2] for x in topology.pairs])
        # get pairs based on terminal atoms of dihedral angles
        for dihedral in topology.dihedrals:
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
            for dihedral in topology.dihedrals:
                if (all([x in dihedral[0:3] for x in pair]) or 
                    all([x in dihedral[1:4] for x in pair])):
                    well_spaced = False
            if well_spaced:
                pairs.append(pair)
        
        topology.pairs = [x + [1] for x in pairs]    
        return topology
    
    @staticmethod
    def _get_con_name(topology, connection):
        """
        Formats name of a connection to be able to look it up in the defaults.
        """
        res_name_1 = topology.atoms[connection[0]][3]
        res_name_2 = topology.atoms[connection[1]][3]
        
        res_names = np.array([res_name_1, res_name_2])
        if res_names[0] == res_names[1]:
            res_ord = [0, 1]
        else:
            res_ord = np.argsort(res_names)
        
        con_name = f"LINKER_{res_names[res_ord[0]]}_{res_names[res_ord[1]]}"
        return con_name, res_ord
    
    @staticmethod
    def _combine_topologies(top_1, top_2):
        """
        Concatenates two topologies. Automatically corrects all atom indices in
        the bonded paramters. 
        """
        if len(top_1.atoms) == 0:
            return top_2
        new_topology = deepcopy(top_1)
        last_res = new_topology.atoms[-1][2]
        last_atom = len(new_topology.atoms)
        new_topology = deepcopy(top_1)
        #Add atoms for top_2
        for atom in top_2.atoms:
            new_atom = deepcopy(atom)
            new_atom[0] += last_atom
            new_atom[2] += last_res
            new_topology.atoms.append(new_atom)
            
        #Add parameters for top_2
        para_idx_len = {"bonds": 2, "pairs": 2, "angles": 3, "dihedrals": 4}
        for para_name in ["bonds", "pairs", "angles", "dihedrals"]:
            for i, old_para in enumerate(top_2.__dict__[para_name]):
                new_para = deepcopy(old_para)
                for j in range(para_idx_len[para_name]):
                    new_para[j] += last_atom
                new_topology.__dict__[para_name].append(new_para)
        return new_topology

    @staticmethod
    def _add_linker(topology, connectivity, connection, linker_topology):
        """
        Applies a linker between two parts of a topology.
        
        Tries to apply a connection topology fragment to the current topology
        by mapping all of its atoms to those of the current topology and then
        copying over the parameters.

        Parameters
        ----------
        topology : RingGen.Topology
            Current topology with disconnected regions.
        connectivity : TYPE
            Connectivity matrix that describes the topology.
        connection : list of int
            Contains the indices of the atoms that form the connecting bond.
        linker_topology : RingGen.Topology
            Topology that contains bonded paramters to link up fragments.

        Returns
        -------
        new_topology : RingGen.Topology
            Current topology with new bonded parameters connecting fragments.
        """
        new_topology = deepcopy(topology)
        ele_dict = {"h": 2, "c": 6, "n": 7, "o": 8, "z": 26, "s": 14}
        
        #connectivity of the linker topology fragment
        con1 = deepcopy(linker_topology.connectivity)
        ele1 = [x[4][0].lower() for x in linker_topology.atoms]
        for i, e in enumerate(ele1):
            con1[i,i] = ele_dict[e]
        
        #connectivity of the current topology
        con2 = deepcopy(connectivity)
        ele2 = [''.join([x.lower() for x in y[4] if not x.isdigit()])
                for y in topology.atoms]
        ele2 = [x[4][0].lower() for x in topology.atoms]
        for i, e in enumerate(ele2):
            con2[i,i] = ele_dict[e]
            
        match = Structure._match_networks(
            con1, con2, matches={linker_topology.connection[0]: connection[0],
                                 linker_topology.connection[1]: connection[1]})
        if match is not None:
            para_idx_len = {"bonds": 2, 
                            "pairs": 2, 
                            "angles": 3, 
                            "dihedrals": 4}
            for para_name in ["bonds", "pairs", "angles", "dihedrals"]:
                for linker_para in linker_topology.__dict__[para_name]:
                    new_para = deepcopy(linker_para)
                    for i in range(para_idx_len[para_name]):
                        new_para[i] = match[new_para[i]-1]+1
                    new_topology.__dict__[para_name].append(new_para)
        else:
            print("Found topology in default library" ,
                  "but could not match it to structure. ",
                  "Generate topology instead.", sep = "")
            new_topology = LinkerGenerator().generate_linker_topology(
                topology, connectivity, connection)
            
        return new_topology

    @staticmethod
    def _match_networks(con1, con2, matches = {}):
        """
        Recursive matching algorithm to match nodes of con1 to nodes of con2.
        
        Usually used via match_patterns wrapper.

        Parameters
        ----------
        con1 : NxN np.array
            Modified connectivity matrix D. D[i,j] = 1 if nodes i and j are 
            connected and 0 otherwise. D[i,i] is determined by the element.
        con2 : MxM np.array
            Modified connectivity matrix D. D[i,j] = 1 if nodes i and j are 
            connected and 0 otherwise. D[i,i] is determined by the element.
        matches : dict, optional
            Known atom mappings from nodes of con1 to nodes of con2. 
            The default is {}.

        Returns
        -------
        matches : dict or None
            A dictionary if method could map all atoms of con1 and None if not.
        """
        #Previous iteration found last match
        if len(matches) == len (con1):
            return matches
       
        #get non-asigned atoms connected to atoms already asigned for con1
        connected_atoms = []
        for key in matches.keys():
            for atom in np.where(con1[key])[0]:
                if atom not in matches.keys() and atom not in connected_atoms:
                    connected_atoms.append(atom)
    
        #get non-asigned atoms connected to atoms already asigned for con2
        connected_atoms_2 = []
        for val in matches.values():
            for atom in np.where(con2[val])[0]:
                if (atom not in matches.values() and 
                    atom not in connected_atoms_2):
                    connected_atoms_2.append(atom)
    
        #try to guess a match between those groups
        for atom_idx_1 in connected_atoms:
            atom_1_cons = con1[atom_idx_1]    
            con_atoms_1 = np.where(atom_1_cons == 1)[0]
            
            
            for atom_idx_2 in connected_atoms_2:
                atom_2_cons = con2[atom_idx_2]    
                con_atoms_2 = np.where(atom_2_cons == 1)[0]
    
                if (atom_1_cons[atom_idx_1] != 0 and 
                    atom_1_cons[atom_idx_1] != atom_2_cons[atom_idx_2]):
                    continue
                if len(con_atoms_1) > len(con_atoms_2):
                    continue
                
                
                # for each connected atom in con1, it either needs to be not 
                # indexed yet, or if it is, the coresponding atom in con2 must 
                # also be connected
                valid_atom = all([matches[x] in con_atoms_2
                                  if x in matches.keys()
                                  else True
                                  for x in con_atoms_1])
                
                if not valid_atom:
                    continue
                    
                #move to next level of recursion
                matches_guess = deepcopy(matches)
                matches_guess.update({atom_idx_1 : atom_idx_2})
                new_matches = Structure._match_networks(con1, 
                                                        con2, 
                                                        matches_guess)
                
                if new_matches is not None:
                    matches = new_matches
                    break
    
            if len(matches) == len (con1):
                return matches
            else:
                return None
        return None
