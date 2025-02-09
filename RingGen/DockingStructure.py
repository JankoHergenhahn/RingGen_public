import itertools
import numpy as np
from copy import deepcopy

from .ComplexStructure import ComplexStructure
from .Coordinates import Coordinates
from .ParaTools.RingTyper import RingTyper
from .Tools import get_COM, get_normal_vector, rotate_array
try:
    from .Minimizer import Minimizer
except:
    from .Minimizer_SD import Minimizer  
    
class DockingStructure(ComplexStructure):
    """
    Object handling binding between template molecules and porphyrin molecules.
   
    Attributes
    ----------
    structures : list of RingGen.Structures
    host_index : int
    guest_index : int
    host_binding_sites : list
    guest_binding_sites : list
    used_host_binding_sites : list
    binding_pairs : list
   """
    def __init__(self, structures = []):
        self.structures = []
        for structure in structures:
            self.add_structure(structure)
        self.host_index = 0
        self.guest_index = 1
        self.host_binding_sites = []
        self.guest_binding_sites = []
        self.used_host_binding_sites = []
        self.binding_pairs = []

    def set_host_index(self, index):
        """
        Sets host_index (int), which specifies which molecule of the structures 
        list is the "host".
        
        Returns self: Allows method chaining.
        """
        self.host_index = index
        self.used_host_binding_sites = []
        return self

    def set_host_binding_sites(self, sites):
        """
        Sets host_binding_sites (list of int), which specify which atom indices
        are binding to the guest.
        
        Returns self: Allows method chaining.
        """
        self.host_binding_sites = sites
        nb_sites = len(self.host_binding_sites)
        print(f"Host structure has {nb_sites} binding sites.")
        return self

    def find_host_binding_sites(self):
        """
        Determines host binding sites, only used zn-atoms as binding sites.
        
        Returns self: Allows method chaining.
        """
        host_binding_sites = []
        for i,atom in enumerate(self.structures[self.host_index].elements):
            if atom == "Zn":
                if i not in self.used_host_binding_sites:
                    host_binding_sites.append(i)
        print(f"Host structure has {len(host_binding_sites)} binding sites.")
        self.host_binding_sites = np.array(host_binding_sites)
        return self
    
    def set_guest_index(self, index):
        """
        Sets guest_index (int), which specifies which molecule of the 
        structures list is the "guest".
        
        Returns self: Allows method chaining.
        """
        self.guest_index = index
        return self

    def set_guest_binding_sites(self, sites):
        """
        Sets guest_binding_sites (list of int), which specify which atom 
        indices are binding to the host.
        
        Returns self: Allows method chaining.
        """
        self.guest_binding_sites = sites
        nb_sites = len(self.guest_binding_sites)
        print(f"Guest structure has {nb_sites} binding sites.")
        return self

    def find_guest_binding_sites(self):
        """
        Determines guest binding sites, uses sp2 hybridized nitrogen atoms with
        only two connected atoms (have a lone pair).
        
        Returns self: Allows method chaining.
        """
        guest_binding_sites=[]
        for i,atom in enumerate(self.structures[self.guest_index].elements):
            if atom == "N":
                if sum(self.structures[self.guest_index].connectivity[i])==2:
                    guest_binding_sites.append(i)
        print(f"Guest structure has {len(guest_binding_sites)} binding sites.")
        self.guest_binding_sites = np.array(guest_binding_sites)
        return self

    def vernier_knot(self):
        """
        Manipulates coordinates to introduce "figure-of-eight" type motifs.
        Structure require some bound and unbound host binding sites.
        
        Returns self: Allows method chaining.
        """
        if len(self.used_host_binding_sites) == 0:
            return
        
        combined_coordinates = self.combine_coordinates().coordinates
        zn_sites, free_sites, bound_sites = self._get_sites_for_vernier_knot()
        
        #divide structure into domains by which zn the atoms are closest to
        distances_to_zn = np.zeros((len(combined_coordinates), len(zn_sites)))
        for i, site in enumerate(zn_sites):
            distances_to_zn[:,i] = np.linalg.norm(combined_coordinates 
                                                  -combined_coordinates[site], 
                                                  axis = 1)
        distances_to_bound = np.min(
            distances_to_zn[:,[i for i,x in enumerate(zn_sites) 
                               if x in bound_sites]],
            axis = 1)
        distances_to_free = np.min(
            distances_to_zn[:,[i for i,x in enumerate(zn_sites) 
                               if x in free_sites]],
            axis = 1)
        domain = (distances_to_free < distances_to_bound)
        
        #rotate domains of unbound zn atoms to induce Vernier knot
        vector = (get_COM(combined_coordinates[1-domain])
                  -get_COM(combined_coordinates[domain]))
        combined_coordinates[domain] = rotate_array(
            combined_coordinates[domain], vector, np.pi)
        
        temp_coordinates = self.combine_coordinates()
        temp_coordinates.set_coordinates(combined_coordinates)
        temp_topology = self._combine_topologies()
        my_minimizer = Minimizer(temp_topology, temp_coordinates)
        my_minimizer.minimize()
        self.update_coordinates(my_minimizer.get_coordinates())
        return self
      
    def dock(self, **kwa):
        """
        Function to dock template molecule to porphyrin structure.

        Parameters
        ----------
        binding_pairs : list of tuples of int, optional
            Specifies which host sites bind to which guest sites if specified.
            Default is [], which means sites will be determined automatically.
        keep_host_rigid : bool, optional
            Constrains host coordinates during optimizations. Useful for 
            stacked template complexes. 
            Default is False.
        first_site : list, optional
            Guess for first host_site-guest_site match. If it is set to [], 
            every combination will be tested to find best orientations.
            Default is [0,0].
        force_guest_positions : bool, optional
            Detaches guest binding sites and constrains them to the correct 
            host binding sites to ensure binding. 
            Default is False.
        force_host_positions : bool, optional
            Activates a different docking algorithm in which the host is 
            partitioned and host_sites are placed at constrained guest sites.
            Default is False.
        buckle_offset" : float, optional
            Determined initial domage when docking template. Useful for stacked
            template complexes. 
            Default is 0
        
        Returns self: Allows method chaining.
        """
        defaults = {"binding_pairs": [],
                    "keep_host_rigid": False, 
                    "first_site": [0,0], 
                    "force_guest_positions": False,
                    "force_host_positions": False,
                    "buckle_offset": 0}
        args = {**defaults, **kwa}
        
        if args["force_guest_positions"] and args["force_host_positions"]:
            print("Can't force guest and host positions simultaneously!")
            return self
        self.find_host_binding_sites()
        self.find_guest_binding_sites()
        host_coord = deepcopy(self.structures[self.host_index].coordinates)
        guest_coord = deepcopy(self.structures[self.guest_index].coordinates)
        
        #Docking mode B
        if args["force_host_positions"]:
            if len(self.structures) > 2:
                print("You are forcing the host binding site positions but",
                      "there are additional structures besides the host and", 
                      "guest. This might lead to unexpected behaviour.",
                      sep = "")
            host_coord = self._force_host_positions(host_coord, guest_coord,
                                                    self.host_binding_sites,
                                                    self.guest_binding_sites,
                                                    args["binding_pairs"])
            self.structures[self.host_index].set_coordinates(host_coord)
            atoms_before_guest = \
                sum([len(self.structures[i].coordinates.elements) 
                     for i in range(self.guest_index)])
                
            nb_at = len(self.structures[self.guest_index].coordinates.elements)
            constrained_atoms = np.arange(nb_at) + atoms_before_guest
            self.minimize(constrained_atoms=constrained_atoms)
        
        #Docking mode A
        else:
            self.binding_pairs = args["binding_pairs"]

            #place guest with host structure
            if len(args["binding_pairs"]) > 0:
                gs0_idx = args["binding_pairs"][0, 0]
                hs0_idx = args["binding_pairs"][0, 1]
                hs0_pos, hs0_vec = self.get_hs0_loc(host_coord, 
                                                    hs0_idx, 
                                                    self.host_binding_sites)
                guest_coord = self.place_guest(guest_coord, 
                                               hs0_pos, 
                                               hs0_vec, 
                                               gs0_idx)
                guest_coord = self.rotate_guest(guest_coord, 
                                                host_coord, 
                                                hs0_vec, 
                                                gs0_idx)
                binding_pairs = args["binding_pairs"]
            else:
                guest_coord = self._find_best_guest_position(
                    host_coord, 
                    guest_coord, 
                    self.host_binding_sites, 
                    self.guest_binding_sites, 
                    args["first_site"])
                binding_pairs = self.get_binding_pairs(host_coord, guest_coord)
                
            #optimize initial complex
            if args["force_guest_positions"]:
                guest_coord, guest_constrained_atoms = \
                    self.place_guest_binding_sites(guest_coord, 
                                                   host_coord, 
                                                   binding_pairs, 
                                                   args["buckle_offset"])
                guest_coord = self.relax_guest(
                    self.structures[self.guest_index].topology,
                    guest_coord, 
                    guest_constrained_atoms)
            else:            
                guest_constrained_atoms = []
            self.structures[self.guest_index].set_coordinates(guest_coord)
            self.relax_system(binding_pairs, args["keep_host_rigid"], 
                              guest_constrained_atoms=guest_constrained_atoms)
        return self
        
  #Docking Mode A
    def get_hs0_loc(self, host_coord, hs0_idx, host_binding_sites):
        """
        Finds positions and orientations of a specified host binding site.

        Parameters
        ----------
        host_coord : RingGen.Coordinates
        hs0_idx : int
        host_binding_sites : list of int
            Need to know about other sites to get orientation correctly.

        Returns
        -------
        hs0_pos : 1x3 np.array
        hs0_vec : 1x3 np.array
        """
        host_COM = get_COM(host_coord.coordinates[host_binding_sites])
        hs0_pos = host_coord.coordinates[hs0_idx]
        connected_atoms = np.where(host_coord.connectivity[hs0_idx] == 1)[0]
        hs0_vec = get_normal_vector(host_coord.coordinates[connected_atoms])
        #invert vector if pointing outside ring
        if np.dot(host_COM-hs0_pos, hs0_vec) < 0:
            hs0_vec *= -1
        return hs0_pos, hs0_vec
    
    def place_guest(self, guest_coord, hs0_pos, hs0_vec, gs0_idx):
        """
        Places guest in right location and orientation based on a given 
        host site position/vector and guest site index.

        Parameters
        ----------
        guest_coord : RingGen.Coordinates
        hs0_pos : 1x3 np.array
        hs0_vec : 1x3 np.array
        gs0_idx : int

        Returns
        -------
        guest_coord : RingGen.Coordinates
        """
        #place guest in right location
        guest_coord.translate(-guest_coord.coordinates[gs0_idx])
        #get guest orientation
        connected_atoms = np.where(guest_coord.connectivity[gs0_idx] == 1)[0]
        gs0_vec = get_COM(guest_coord.coordinates[connected_atoms])
        gs0_vec /= np.linalg.norm(gs0_vec)
        #rotate guest into right orientation
        if np.dot(gs0_vec, hs0_vec) < -0.9:
            #need to do differently to preserve chirality, use it for now
            guest_coord.coordinates *= -1
        else:
            guest_coord.rotate(hs0_vec + gs0_vec, np.pi)
        #Zn-N bond is about 2.2 Angstrom
        guest_coord.translate(hs0_pos + 2.2 * hs0_vec)
        
        return guest_coord
        
    def rotate_guest(self, guest_coord, host_coord, hs0_vec, gs0_idx):
        """
        Rotates guest around host_site-guest_site vector to minimize RMSD 
        distance between host sites and guest sites

        Parameters
        ----------
        guest_coord : RingGen.Coordinates
        host_coord : RingGen.Coordinates
        hs0_vec : 1x3 np.array
        gs0_idx : int

        Returns
        -------
        guest_coord : RingGen.Coordinates
        """
        gs0_pos = deepcopy(guest_coord.coordinates[gs0_idx])
        RMSD = []        
        angles = np.linspace(0, 2*np.pi, 100)
        for angle in angles:
            #need to move to origin temporarily for rotation
            temp_coord = deepcopy(guest_coord)
            temp_coord.translate(-gs0_pos)
            temp_coord.rotate(hs0_vec, angle)
            temp_coord.translate(gs0_pos)
            RMSD.append(self.BS_RMSD(host_coord, temp_coord))
        
        guest_coord.translate(-gs0_pos)
        guest_coord.rotate(hs0_vec, angles[np.argmin(RMSD)])
        guest_coord.translate(gs0_pos)

        return guest_coord

    def BS_RMSD(self, host_coord, guest_coord):
        """
        Determines RMSD distance between host and guest binding sites.
        
        Uses self.binding_pairs to match up pairs if defines, otherwise tries
        to find best combination of pairs using get_binding_pairs.
        """
        
        if len(self.binding_pairs) == 0:
            binding_pairs = self.get_binding_pairs(host_coord, guest_coord)
        else:
            binding_pairs = self.binding_pairs
        gs_coords = guest_coord.coordinates[binding_pairs[:,0]]
        hs_coords = host_coord.coordinates[binding_pairs[:,1]]
        return np.sqrt(np.mean((hs_coords - gs_coords)**2))

    def get_binding_pairs(self, host_coord, guest_coord):
        """
        Determins best combination of (host_site, guest_site) pairs to minimize
        RMSD distance between host and guest binding sites.
        """
        nb_gs = len(self.guest_binding_sites)
        nb_hs = len(self.host_binding_sites)
        distances = np.zeros((nb_gs, nb_hs))
        hs_coords = host_coord.coordinates[self.host_binding_sites]
        for i, idx in enumerate(self.guest_binding_sites):
            distances[i,:] = np.linalg.norm(
                hs_coords-guest_coord.coordinates[[idx]], axis = 1)
        
        #try to get all minima first
        binding_matches = np.argmin(distances, axis = 1)
        if len(set(binding_matches)) != len(binding_matches):
            binding_matches = [None for x in range(nb_gs)]
            #delete closest pair until all are asigned
            for i in range(nb_gs):
                min_pos = np.where(distances == distances.min())
                binding_matches[min_pos[0][0]] = min_pos[1][0]
                #set distances of asigned pairs high so they are not reassigned
                distances[min_pos[0][0], :] = 1000
                distances[:, min_pos[1][0]] = 1000
                
        #convert matches to indices
        binding_pairs = np.vstack([self.guest_binding_sites,
                                   [self.host_binding_sites[x] 
                                    for x in binding_matches]]).T
        return binding_pairs
    
    
    def _find_best_guest_position(self, 
                                  host_coord, 
                                  guest_coord, 
                                  host_sites, 
                                  guest_sites, 
                                  first_site):
        """
        Tries to position guest relative to host to minimize RMSD distance
        between host and guest binding site pairs. 
        
        If first_site is given, one binding site pair is fixed and guest is 
        only rotated around that point. Otherwise, all possible binding site 
        pair combinations are tested to find best match.
        """
        if first_site:
            gs0_idx = guest_sites[first_site[0]]
            hs0_idx = host_sites[first_site[1]]
            hs0_pos, hs0_vec = \
                self.get_hs0_loc(host_coord, hs0_idx, host_sites)
            guest_coord = \
                self.place_guest(guest_coord, hs0_pos, hs0_vec, gs0_idx)
            guest_coord = \
                self.rotate_guest(guest_coord, host_coord, hs0_vec, gs0_idx)
        else:
            RMSDs = np.zeros((len(guest_sites), len(host_sites)))
            for i, gs_idx in enumerate(guest_sites):
                for j, hs_idx in enumerate(host_sites):
                    hs_pos, hs_vec = \
                        self.get_hs0_loc(host_coord, hs_idx, host_sites)
                    guest_coord = \
                        self.place_guest(guest_coord,hs_pos,hs_vec,gs_idx)
                    guest_coord = \
                        self.rotate_guest(guest_coord,host_coord,hs_vec,gs_idx)
                    RMSDs[i,j] = self.BS_RMSD(host_coord, guest_coord)
            min_ind = np.where(RMSDs == RMSDs.min())
            gs0_idx = guest_sites[min_ind[0][0]]
            hs0_idx = host_sites[min_ind[1][0]]
            hs0_pos, hs0_vec = \
                self.get_hs0_loc(host_coord, hs0_idx, host_sites)
            guest_coord = \
                self.place_guest(guest_coord, hs0_pos, hs0_vec, gs0_idx)
            guest_coord = \
                self.rotate_guest(guest_coord, host_coord, hs0_vec, gs0_idx)
        return guest_coord
    
  #Forcing Binding sites
    def place_guest_binding_sites(self, 
                                  guest_coord, 
                                  host_coord, 
                                  binding_pairs, 
                                  buckle_offset):
        """
        Cuts off guest binding sites from the rest of the guest molecule and 
        places them explicitely at the correct host binding sites.

        Parameters
        ----------
        guest_coord : RingGen.Coordinates
        host_coord : RingGen.Coordinates
        binding_pairs : list of pairs of int
            Each list entry give a combination of host_site and guest_site that
            will be forced to be bound together.
        buckle_offset : float
            Distance between mean plane of host sites and centre of mass of 
            guest.

        Returns
        -------
        guest_coord : RingGen.Coordinates
        constrained_atoms : list of int
            Atom indices of extended binding sites that need to be fixed.
        """
        # Place guest molecule above host cavity
        constrained_atoms = []
        plane_vector = get_normal_vector(host_coord.coordinates)
        if np.dot(plane_vector, [0,0,1]) < 0:
            plane_vector *= -1
        guest_coord.coordinates += plane_vector * buckle_offset
        guest_coord.translate(get_COM(host_coord.coordinates) 
                              + plane_vector * buckle_offset 
                              - get_COM(guest_coord.coordinates))
        
        #Place each guest site
        for i, (gs0_idx, hs0_idx) in enumerate(binding_pairs):
            #find out orientation of porphyrin
            hs0_pos, hs0_vec  = self.get_hs0_loc(host_coord, 
                                                 hs0_idx, 
                                                 binding_pairs[:,1])
            host_com = np.mean(host_coord.coordinates, axis=0)
    
            #invert if pointing outwards
            if np.dot(hs0_vec,host_com-hs0_pos) < 0:
                hs0_vec *= -1

            #get extended guest binding site, e.g. pyridine motif
            gs0_pos = deepcopy(guest_coord.coordinates[gs0_idx])
            connected_atoms = np.where(guest_coord.connectivity[gs0_idx]==1)[0]
            gs0_vec = get_COM(guest_coord.coordinates[connected_atoms])-gs0_pos
            gs0_vec /= np.linalg.norm(gs0_vec)

            guest_descriptor = RingTyper(guest_coord.elements, 
                                         guest_coord.connectivity)
            ring_id = guest_descriptor.ring_ids[gs0_idx][0]
            ring_skeleton = guest_descriptor.rings[ring_id]
            binding_group_indeces = self._get_full_binding_site(ring_skeleton, 
                                                                guest_coord)
            connected_atoms = np.where(guest_coord.connectivity[gs0_idx]==1)[0]
    
            temp_fragment = Coordinates().set(
                np.array(guest_coord.elements)[binding_group_indeces],
                guest_coord.coordinates[binding_group_indeces])
            
            #allign and rotate extended guest binding site
            temp_fragment.translate(-gs0_pos)
            if np.dot(gs0_vec, hs0_vec) < -0.9:
                #need to do differently to preserve chirality
                temp_fragment.coordinates *= -1
            else:
                temp_fragment.rotate(hs0_vec + gs0_vec, np.pi)
            temp_fragment.translate(hs0_pos+2.1*hs0_vec)
            guest_coord.coordinates[binding_group_indeces] = \
                temp_fragment.coordinates
            constrained_atoms.extend(binding_group_indeces)       
        
        return guest_coord, constrained_atoms

    def relax_guest(guest_topology, guest_coord, constrained_atoms):
        """
        Performs minimization while constraining atoms belonging to extended 
        guest binding sites.
        """
        my_minimizer = Minimizer(guest_topology, 
                                 guest_coord, 
                                 constrained_atoms)
        my_minimizer.minimize()
        guest_coord.set_coordinates(my_minimizer.get_coordinates())
        return guest_coord

    def _get_full_binding_site(self, atom_indeces, guest_coord):
        """
        Find atoms belonging to extended guest binding site including hydrogen
        atoms from the ring skeleton given in atom_indeces (list of int).
        """
        full_site = []
        for i in atom_indeces:
            full_site.extend(np.where(guest_coord.connectivity[i] == 1)[0])
        full_site = list(set(full_site))

        full_site.sort()
        return full_site

  #Relaxing system
    def relax_system(self, 
                     binding_pairs, 
                     keep_host_rigid, 
                     guest_constrained_atoms = []):
        """
        Performs a series of optimizations aimed at slowely moving from a very 
        constrained structure in which binding sites have been enforced to 
        an optimized complex.
        """        
        steps = 1000
        
        #1st round: bonds in binding sites, keep host constrained
        temp_coordinates = self.combine_coordinates()
        temp_topology = self._combine_topologies()
        temp_topology = self._add_binding_site_bonds(temp_topology, 
                                                     binding_pairs)
        constrained_atoms = self.get_constrained_atoms(False,
                                                       guest_constrained_atoms)
        my_minimizer = Minimizer(temp_topology, 
                                 temp_coordinates, 
                                 constrained_atoms)
        my_minimizer.minimize(steps = steps)
        
        #2nd round: bonds in binding sites, no constraints (except maybe host)
        temp_coordinates.set_coordinates(my_minimizer.get_coordinates())
        if keep_host_rigid:
            constrained_atoms = self.get_constrained_atoms(just_host = True)
        else:
            constrained_atoms = []
        my_minimizer = Minimizer(temp_topology, 
                                 temp_coordinates, 
                                 constrained_atoms)
        my_minimizer.minimize(steps = steps)

        #3rd round: normal topologies, no modifications
        temp_coordinates.set_coordinates(my_minimizer.get_coordinates())
        temp_topology = self._combine_topologies()
        if keep_host_rigid:
            constrained_atoms = self.get_constrained_atoms(just_host = True)
        else:
            constrained_atoms = []
        my_minimizer = Minimizer(temp_topology, 
                                 temp_coordinates, 
                                 constrained_atoms)
        my_minimizer.minimize(steps = steps)

        #Output
        self.update_coordinates(my_minimizer.get_coordinates())
        return self

    def _add_binding_site_bonds(self, temp_topology, binding_pairs = None):
        """
        Adds bond parameters between atoms in bidning sites 
        (pairs of host_site, guest_site).
        """
        if binding_pairs is not None:
            #Need to correct atom indices in combined structure
            atoms_before_guest = \
                sum([len(self.structures[i].coordinates.elements) 
                     for i in range(self.guest_index)])
            atoms_before_host = \
                sum([len(self.structures[i].coordinates.elements) 
                     for i in range(self.host_index)])
                
            for bs_guest, bs_host in binding_pairs:
                idx_1 = bs_guest + atoms_before_guest + 1
                idx_2 = bs_host + atoms_before_host + 1
                host_connectivity = \
                    self.structures[self.host_index].coordinates.connectivity
                connected_N_atoms = np.where(host_connectivity[bs_host]==1)[0]
                
                self.used_host_binding_sites.append(bs_host)
                temp_topology.bonds.append([idx_1, idx_2, 1, 0.220, 500000])
                for N_atom in connected_N_atoms:
                    temp_topology.angles.append(
                        [idx_1, idx_2, N_atom + 1, 1, 90, 200])
        return temp_topology

    def get_constrained_atoms(self, 
                              just_host = False, 
                              guest_constrained_atoms = []):
        """
        Determines the corrected atom indices of the atoms that are to be 
        constrained during an optimization.
        
        Atom indices need to be corrected in combined structure when structures
        are concatenated.
        """
        constrained_atoms = []
        counter = 0
        if just_host:
            #only constrain the host
            for i, struc in enumerate(self.structures):
                if i == self.host_index:
                    for atom in struc.elements:
                        constrained_atoms.append(counter)
                        counter += 1
                else:
                    counter += len(struc.elements)
        else:
            #constrain all atoms other than the guest
            for i, struc in enumerate(self.structures):
                if i != self.guest_index:
                    for atom in struc.elements:
                        constrained_atoms.append(counter)
                        counter += 1
                else:
                    counter += len(struc.elements)
        atoms_before_guest = sum([len(self.structures[i].coordinates.elements) 
                                      for i in range(self.guest_index)])
        constrained_atoms += [x + atoms_before_guest 
                              for x in guest_constrained_atoms]
        return constrained_atoms
    
    def update_coordinates(self, coordinates):
        """
        Splits up coordinates of combined coordinates object and assigns them
        to the structures of self.structures.
        """
        atom_index_cum = 0
        for i,x in enumerate(self.structures):
            coord_length = len(self.structures[i].coordinates.elements)
            self.structures[i].coordinates.set_coordinates(
                coordinates[atom_index_cum:atom_index_cum+coord_length])
            atom_index_cum += coord_length

  #Docking mode B
    def _force_host_positions(self, 
                              host_coord, 
                              guest_coord, 
                              host_binding_sites, 
                              guest_binding_sites, 
                              binding_pairs):
        """
        Docking mode B, based on partitioning the host coordinates and placing 
        porphyrins in best places to bind to guest binding sites.
        """
        #Divide structure up into domains
        nb_domains = len(host_binding_sites)
        distance_matrix = [np.linalg.norm(host_coord.coordinates 
                                          - host_coord.coordinates[x], axis=1) 
                           for x in self.host_binding_sites]
        distance_matrix = np.array(distance_matrix)
        atom_domain_attribution = np.argmin(distance_matrix, axis=0)
        domain_list = [np.where(atom_domain_attribution == x)[0] 
                       for x in range(nb_domains)]
        
        #Place each domain at correct binding site
        host_coord = self._place_host_binding_sites(host_coord, 
                                                    guest_coord, 
                                                    host_binding_sites, 
                                                    guest_binding_sites, 
                                                    domain_list)

        #Find domain connections
        domain_connections_list = \
            self._find_domain_connections(nb_domains, 
                                          domain_list, 
                                          atom_domain_attribution, 
                                          host_coord)
        
        #Rotate domains
        host_coord = self._rotate_host_binding_sites(host_coord, 
                                                     guest_coord, 
                                                     host_binding_sites, 
                                                     guest_binding_sites, 
                                                     domain_list, 
                                                     domain_connections_list)
        
        return host_coord

    @staticmethod
    def _place_host_binding_sites(host_coord, 
                                  guest_coord, 
                                  host_binding_sites, 
                                  guest_binding_sites, 
                                  domain_list):
        """
        Positions each porphyrin (host) site at guest binding site.
        Returns the host coordinates as Nx3 np.array
        """
        for i, (zn_idx, N_idx) in enumerate(zip(host_binding_sites, 
                                                guest_binding_sites)):
            #get position and direction of guest binding site
            guest_binding_position = guest_coord.coordinates[N_idx]
            connected_atoms = np.where(guest_coord.connectivity[N_idx] == 1)[0]
            connected_atoms_com = np.mean(
                guest_coord.coordinates[connected_atoms], axis = 0)
            guest_binding_vector = guest_binding_position - connected_atoms_com
            guest_binding_vector /= np.linalg.norm(guest_binding_vector)

            #orientation of host structure binding site
            host_binding_position = host_coord.coordinates[zn_idx]
            connected_atoms = np.where(host_coord.connectivity[zn_idx] == 1)[0]
            for j,k in itertools.product(connected_atoms, connected_atoms):
                v1 = host_coord.coordinates[j] - host_binding_position
                v1 /= np.linalg.norm(v1)
                v2 = host_coord.coordinates[k] - host_binding_position
                v2 /= np.linalg.norm(v2)
                if abs(np.dot(v1, v2)) < 0.5:
                    host_plane_vector = np.cross(v1, v2)
                    host_plane_vector /= np.linalg.norm(host_plane_vector)
                    break

            #position host fragments
            temp_coords = Coordinates().set_coordinates(
                deepcopy(host_coord.coordinates[domain_list[i]]))
            temp_coords.translate(-host_binding_position)
            rotation_axis = host_plane_vector + guest_binding_vector
            temp_coords.rotate(rotation_axis, np.pi)
            trans_vector = guest_binding_position + 2*guest_binding_vector
            temp_coords.translate(trans_vector)
            host_coord.coordinates[domain_list[i]] = temp_coords.coordinates
        return host_coord

    @staticmethod
    def _find_domain_connections(nb_domains, 
                                 domain_list, 
                                 atom_domain_attribution, 
                                 host_coord):
        """
        Finds atom indices that connect the partitioned host molecule domains.
        Returns a list of pairs of int
        """
        domain_connections_list = []
        for i in range(nb_domains):
            connections = []
            for atom in domain_list[i]:
                connected_atoms = np.where(host_coord.connectivity[atom]==1)[0]
                for connected_atom in connected_atoms:
                    if (atom_domain_attribution[atom] 
                        != atom_domain_attribution[connected_atom]):
                        connections.append([atom, connected_atom])
            domain_connections_list.append(connections)
        return domain_connections_list

    @staticmethod
    def _rotate_host_binding_sites(host_coord, 
                                   guest_coord, 
                                   host_binding_sites, 
                                   guest_binding_sites, 
                                   domain_list, 
                                   domain_connections_list):
        """
        Rotates each porphyrin (host) site so that the distances between the 
        connecting atoms between domains are minimized.
        Returns the host coordinates as Nx3 np.array
        """
        for repeats in range(3):
            for i, (zn_idx, N_idx) in enumerate(zip(host_binding_sites, 
                                                    guest_binding_sites)):
                #get position and orientation of host binding site
                host_position = deepcopy(host_coord.coordinates[zn_idx])
                guest_binding_position = guest_coord.coordinates[N_idx]
                connected_atoms = np.where(
                    guest_coord.connectivity[N_idx] == 1)[0]
                connected_atoms_com = np.mean(
                    guest_coord.coordinates[connected_atoms], axis = 0)
                guest_binding_vector = \
                    guest_binding_position - connected_atoms_com
                guest_binding_vector /= np.linalg.norm(guest_binding_vector)
                
                # rotate porphyrin around host_site-guest_site vector to 
                # minimize connecting atoms distances
                errors = np.zeros(36)
                for angle in range(36):
                    #need to move to origin temporarily for rotation
                    host_coord.coordinates[domain_list[i]] -= host_position
                    host_coord.coordinates[domain_list[i]] = \
                        rotate_array(host_coord.coordinates[domain_list[i]],
                                     guest_binding_vector, np.pi*10/180)
                    host_coord.coordinates[domain_list[i]] += host_position
                    errors[angle] = np.mean(
                        [np.linalg.norm(host_coord.coordinates[x[0]]-
                                        host_coord.coordinates[x[1]])**2
                         for x in domain_connections_list[i]])

                angle_opt = np.argmin(errors) * 10/180 * np.pi
                host_coord.coordinates[domain_list[i]] -= host_position
                host_coord.coordinates[domain_list[i]] = \
                        rotate_array(host_coord.coordinates[domain_list[i]],
                                     guest_binding_vector, angle_opt)
                host_coord.coordinates[domain_list[i]] += host_position
        return host_coord
    
  #Vernier knot
    def _get_sites_for_vernier_knot(self):
        """
        Determines state of host binding sites for Vernier process. 
        Returns three lists of int
        """
        
        atoms_before_host = sum([len(self.structures[i].coordinates.elements) 
                                 for i in range(self.host_index)])
        bound_sites = np.array(deepcopy(self.used_host_binding_sites))
        zn_sites = np.array(deepcopy(self.host_binding_sites))
        free_sites = np.array([x for x in zn_sites if x not in bound_sites])
        
        zn_sites += atoms_before_host
        free_sites += atoms_before_host
        bound_sites += atoms_before_host
        
        return zn_sites, free_sites, bound_sites