import subprocess
import numpy as np
from copy import deepcopy

from .Coordinates import Coordinates
from .Topology import Topology
from .ParaTools.AtomTypeDeterminator import AtomTypeDeterminator
from .ParaTools.ParameterDeterminator import ParameterDeterminator
from .ParaTools.BCCGenerator import BCCGenerator
from .ParaTools.GAFFpara import GAFFpara
#from .ParaTools.GAFF2para import GAFF2para

class TopologyGenerator:
    """
    TopologyGenerator object to create topologies from coordinates. 
    
    This class is used to generate topologies using the General AMBER force
    field (GAFF) using algorithms for atom type assignmnents implemented in
    the antechamber programme.
    
    Attributes
    ----------
    coordinates : RingGen.Coordinates
    elements : list of str
    topology : RingGen.Topology
    """
    def generate_topology(self, 
                          input_object, 
                          input_type=None, 
                          easy_valence=False):
        """
        Creates topology based on coordinate file, Coordinate object or 
        MOPAC output file.

        Parameters
        ----------
        input_object : str or RingGen.Coordinates
            Name of file that contains coordinates/MOPAC output or 
            RingGen.Coordinates object.
        input_type : str, optional
            Needs to be set to "Coordinates" if RingGen.Coordinates object is 
            given. The default is None.
        easy_valence : bool, optional
            Simplifies valence search if set to True, which speeds up the 
            parameterization. Can be used if there are no formal charges or
            delocalized bonds. The default is False.

        Returns
        -------
        self: Allows method chaining.
        """
        
        #Process_input
        coordinates, charges = self._process_input(input_object, input_type)
        self.coordinates = coordinates
        self.elements = coordinates.elements
            
        #Analyze structure
        atom_type_determinator = \
            AtomTypeDeterminator(self.coordinates, easy_valence)
        self.atom_types = atom_type_determinator.get_atom_types()
        self.parameter_determinator = \
            ParameterDeterminator(self.coordinates.connectivity)
        bcc_determinator = BCCGenerator(atom_type_determinator)
        charge_corrections = bcc_determinator.get_bcc_charges()

        #Generate topology
        self.force_field = GAFFpara()
        self.topology = self._build_topology()
        self.topology = self._set_charges(self.topology, 
                                          charges, 
                                          charge_corrections)
        return self
    
    def get_topology(self):
        """
        Get generated topology.

        Returns
        -------
        RingGen.Topology
            Topology that was generated based on coordinates.

        """
        return deepcopy(self.topology)

    def get_coordinates(self):
        """
        Get passed coordinates

        Returns
        -------
        RingGen.Coordinates
            Coordinates that were passed/read in.

        """
        return deepcopy(self.coordinates)
    
    def _process_input(self, input_object, input_type):
        """
        Function to process input.
        
        Needs different action depending on if a file is given or 
        a Coordinate object.
        """
        if input_type == None:
            input_type = input_object.split(".")[-1]
            
        if input_type == "out":
            coordinates = self._get_coordinates_from_MOPAC_file(input_object)
            coordinates.create_connectivity()
            charges = self._get_charges_from_file(input_object)
            
        elif input_type in ["xyz", "pdb", "mop", "gro"]:
            coordinates = Coordinates().read_file(input_object)
            coordinates.create_connectivity()
            with open("mopac_temp.mop", "w+") as f:
                f.write("AM1 1SCF\n\n\n")
                for e,c in zip(coordinates.elements, 
                               coordinates.coordinates):
                    f.write(f"{e:>3}{c[0]:>10.5f}{c[1]:>10.5f}{c[2]:>10.5f}\n")
            subprocess.run(["mopac","mopac_temp.mop"])
            charges = self._get_charges_from_file("mopac_temp.out")
            
        elif input_type == "Coordinates":
            coordinates = input_object
            coordinates.create_connectivity()
            with open("mopac_temp.mop", "w+") as f:
                f.write("AM1 1SCF\n\n\n")
                for e,c in zip(coordinates.elements, coordinates.coordinates):
                    f.write(f"{e:>3}{c[0]:>10.5f}{c[1]:>10.5f}{c[2]:>10.5f}\n")
            try:
                subprocess.run(["mopac","mopac_temp.mop"])
                charges = self._get_charges_from_file("mopac_temp.out")
            except:
                print("MOPAC missing. Skipped AM1 charges calculations and ",
                      "set initial charges to 0. Will still apply BCC. ",
                      "This will probably result in inaccurate structures.", 
                      sep = "")
                charges = list(np.zeros(len(coordinates.elements)))
            
        return coordinates, charges


    def _get_coordinates_from_MOPAC_file(self, file):
        """
        Read coordinates (elements and postions) from MOPAC output file.
        """
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
        return Coordinates().set(elements, np.array(coordinates))
    
    def _get_charges_from_file(self, file):
        """
        Reads partial charges from MOPAC output file.
        """
        with open(file) as f:
            content = f.readlines()

        charges = []
        for i,line in enumerate(content):
            if "Empirical Formula:" in line:
                N_atoms = int(line.split()[-2])
            if "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS" in line:
                start_index = i+3
        for line in content[start_index:start_index + N_atoms]:
            charges.append(float(line.split()[2]))
        return charges

    def _build_topology(self):
        """
        Constructs topology based on self.force_field, lists of indices in 
        self.parameter_determinator and self.atom_types
        """
        topology = Topology()

        topology.atoms = self._get_atoms()
        topology.bonds = self._get_bonds()
        topology.angles = self._get_angles()
        topology.dihedrals = self._get_dihedrals() + self._get_impropers()
        topology.pairs = self._get_pairs()

        return topology
    
    def _get_atoms(self):
        """
        Constructs the atoms of the topology
        """
        mass_dict = {"H": 1.008,
                     "B": 10.811,
                     "C": 12.011,
                     "N": 14.010,
                     "O": 16.000,
                     "F": 18.998,
                     "Si": 28.086,
                     "P": 30.974,
                     "S": 32.066,
                     "Cl": 35.453,
                     "Br": 79.904,
                     "I": 126.904}
        label_counter_dict = {"H": 0,
                              "B": 0,
                              "C": 0,
                              "N": 0,
                              "O": 0,
                              "F": 0,
                              "Si": 0,
                              "P": 0,
                              "S": 0,
                              "Cl": 0,
                              "Br": 0,
                              "I": 0}

        atoms=[]
        for i, (element, atom_type) in enumerate(zip(self.elements, 
                                                     self.atom_types)):
            label_counter_dict[element]+=1
            if label_counter_dict[element]<10:
                label = f"{element}0{label_counter_dict[element]}"
            else:
                label = f"{element}{label_counter_dict[element]}"
            mass = mass_dict[element]
            #id type res_id res_name atom_name charge_group charge mass
            atoms.append([(i+1), atom_type, 1, "MOL", label, (i+1), 0.0, mass])
        return atoms

    def _get_bonds(self):
        """
        Constructs the bonds of the topology
        """
        bonds=[]
        for bond in self.parameter_determinator.bond_indeces:
            atoms_in_bond = [self.atom_types[x] for x in bond]
            bond_name = f"{atoms_in_bond[0]}-{atoms_in_bond[1]}"
            if bond_name in self.force_field.bonds:
                r_e = round(self.force_field.bonds[bond_name]["r_e"]*0.1, 5)
                k_f = round(self.force_field.bonds[bond_name]["k_f"]*418.4*2,1)
                bonds.append([bond[0]+1,bond[1]+1,1, r_e, k_f])
            else:
                bonds.append([bond[0]+1,bond[1]+1,1,0,0])
                print(f"Did not find parameters for bond {bond_name}({bond}).",
                      "Set parameters to 0.", sep = "")
        return bonds

    def _get_angles(self):
        """
        Constructs the angles of the topology
        """
        angles=[]
        for angle in self.parameter_determinator.angle_indeces:
            atoms_in_angle = [self.atom_types[x] for x in angle]
            angle_name = \
                f"{atoms_in_angle[0]}-{atoms_in_angle[1]}-{atoms_in_angle[2]}"
            if angle_name in self.force_field.angles:
                th_e = round(self.force_field.angles[angle_name]["th_e"], 2)
                k_f = round(
                    self.force_field.angles[angle_name]["k_f"]*4.184*2, 2)
                angles.append([angle[0]+1,angle[1]+1,angle[2]+1,1, th_e, k_f])
            else:
                angles.append([angle[0]+1,angle[1]+1,angle[2]+1,1,0,0])
                print("Did not find parameters for angle",
                      f"{angle_name}({angle}). Set parameters to 0.", sep = "")
        return angles

    def _get_dihedrals(self):
        """
        Constructs the dihedrals of the topology
        """
        dihedrals=[]
        for dihedral in self.parameter_determinator.dihedral_indeces:
            atoms_in_dihedral = [self.atom_types[x] for x in dihedral]
            dihedral_name = (f"{atoms_in_dihedral[0]}"
                             +f"-{atoms_in_dihedral[1]}"
                             +f"-{atoms_in_dihedral[2]}"
                             +f"-{atoms_in_dihedral[3]}")
            dihedral_name_simple = (f"X-{atoms_in_dihedral[1]}"
                                    +f"-{atoms_in_dihedral[2]}-X")
            if dihedral_name in self.force_field.dihedrals:
                dihedral_parameters = \
                    self.force_field.dihedrals[dihedral_name]
            elif dihedral_name_simple in self.force_field.dihedrals: 
                dihedral_parameters = \
                    self.force_field.dihedrals[dihedral_name_simple]
            else:
                dihedrals.append([dihedral[0]+1,
                                  dihedral[1]+1,
                                  dihedral[2]+1,
                                  dihedral[3]+1,1, 
                                  180.0, 
                                  4.18, 
                                  2])
                print("Did not find parameters for dihedral",
                      f"{dihedral_name}({dihedral}). Guessed value.", sep = "")
                continue
            
            #some dihedral parameters are made up of multiple contributions
            for dihedral_parameter_cont in dihedral_parameters:
                mul = abs(dihedral_parameter_cont["mul"])
                k_f = round(dihedral_parameter_cont["k_f"]*4.184/mul**2,2)
                th_e = dihedral_parameter_cont["th_e"]
                dihedrals.append([dihedral[0]+1,
                                  dihedral[1]+1,
                                  dihedral[2]+1,
                                  dihedral[3]+1,1,
                                  th_e, 
                                  k_f, 
                                  mul])
                
        return dihedrals

    def _get_impropers(self):
        """Constructs the improper dihedrals of the topology
        """
        impropers=[]
        for improper in self.parameter_determinator.improper_indeces:
            atom_centre = self.atom_types[improper['centre']] 
            atoms_vertices = [self.atom_types[x] for x in improper['vertices']]
            improper_names = []
                
            improper_names.append(f"{atoms_vertices[0]}-{atoms_vertices[1]}"
                                  +f"-{atom_centre}-{atoms_vertices[2]}")
            improper_names.append(f"{atoms_vertices[1]}-{atoms_vertices[0]}"
                                  +f"-{atom_centre}-{atoms_vertices[2]}")
            improper_names.append(f"{atoms_vertices[0]}-{atoms_vertices[2]}"
                                  +f"-{atom_centre}-{atoms_vertices[1]}")
            improper_names.append(f"{atoms_vertices[2]}-{atoms_vertices[0]}"
                                  +f"-{atom_centre}-{atoms_vertices[1]}")
            improper_names.append(f"{atoms_vertices[1]}-{atoms_vertices[2]}"
                                  +f"-{atom_centre}-{atoms_vertices[0]}")
            improper_names.append(f"{atoms_vertices[2]}-{atoms_vertices[1]}"
                                  +f"-{atom_centre}-{atoms_vertices[0]}")
            
            improper_names.append(f"X-{atoms_vertices[1]}-{atom_centre}"
                                  +f"-{atoms_vertices[2]}")
            improper_names.append(f"X-{atoms_vertices[0]}-{atom_centre}"
                                  +f"-{atoms_vertices[2]}")
            improper_names.append(f"X-{atoms_vertices[2]}-{atom_centre}"
                                  +f"-{atoms_vertices[1]}")
            improper_names.append(f"X-{atoms_vertices[0]}-{atom_centre}"
                                  +f"-{atoms_vertices[1]}")
            improper_names.append(f"X-{atoms_vertices[1]}-{atom_centre}"
                                  +f"-{atoms_vertices[0]}")
            improper_names.append(f"X-{atoms_vertices[2]}-{atom_centre}"
                                  +f"-{atoms_vertices[0]}")
            
            improper_names.append(f"X-X-{atom_centre}-{atoms_vertices[2]}")
            improper_names.append(f"X-X-{atom_centre}-{atoms_vertices[1]}")
            improper_names.append(f"X-X-{atom_centre}-{atoms_vertices[0]}")
            
            atom_0 = improper['vertices'][0] 
            atom_1 = improper['vertices'][1] 
            atom_2 = improper['vertices'][2] 
            atom_c = improper['centre'] 
            
            improper_indices = [[atom_0, atom_1, atom_c, atom_2],
                                [atom_1, atom_0, atom_c, atom_2],
                                [atom_0, atom_2, atom_c, atom_1],
                                [atom_2, atom_0, atom_c, atom_1],
                                [atom_1, atom_2, atom_c, atom_0],
                                [atom_2, atom_1, atom_c, atom_0],
                                [atom_0, atom_1, atom_c, atom_2],
                                [atom_1, atom_0, atom_c, atom_2],
                                [atom_0, atom_2, atom_c, atom_1],
                                [atom_2, atom_0, atom_c, atom_1],
                                [atom_1, atom_2, atom_c, atom_0],
                                [atom_2, atom_1, atom_c, atom_0],
                                [atom_0, atom_1, atom_c, atom_2],
                                [atom_0, atom_2, atom_c, atom_1],
                                [atom_1, atom_2, atom_c, atom_0]]
            
            found_improper = False
            for improper_name, ind in zip(improper_names, improper_indices):
                if improper_name in self.force_field.impropers:
                    th_e = self.force_field.impropers[improper_name]["th_e"]
                    k_f = round(
                      self.force_field.impropers[improper_name]["k_f"]*4.184,2)
                    mul = self.force_field.impropers[improper_name]["mul"]
                    impropers.append([ind[0]+1,ind[1]+1,ind[2]+1,ind[3]+1,4,
                                      th_e, k_f, mul])
                    found_improper = True
                    break
            if not found_improper and any([x in improper_names[0] 
                                           for x in ["cc", "cd", "ce", "cf"]]):
                for improper_name,ind in zip(improper_names, improper_indices):
                    for x in ["cc", "cd", "ce", "cf"]:
                        improper_name = improper_name.replace(x, "c2")
                    if improper_name in self.force_field.impropers:
                        th_e =self.force_field.impropers[improper_name]["th_e"]
                        k_f0 =self.force_field.impropers[improper_name]["k_f"]
                        k_f = round(k_f0*4.184, 2)
                        mul = self.force_field.impropers[improper_name]["mul"]
                        impropers.append([ind[0]+1,ind[1]+1,ind[2]+1,ind[3]+1,
                                          4, th_e, k_f, mul ])
                        found_improper = True
                        break
                
            if not found_improper:
                print(f"Did not find parameters for improper ({improper}: ",
                      f"{improper_names[0]}). \n Guessed value.", sep = "")
                impropers.append([ind[0]+1,ind[1]+1,ind[2]+1,ind[3]+1,4, 
                                  180.0, 4.6, 2])
                
        return impropers


    def _get_pairs(self):
        """Constructs the pairs of the topology
        """
        pairs=[]
        for pair in self.parameter_determinator.pair_indeces:
            pairs.append([pair[0]+1,pair[1]+1,1])
        return pairs
    
    def _set_charges(self, topology, charges, charge_corrections):
        """Sets the partial charges based on AM1+BCC
        (AM1 charges from MOPAC calculation +BCC correction applied)
        """
        topology = deepcopy(topology)
        for i,atom in enumerate(topology.atoms):
            topology.atoms[i][6] = round(charges[i] + charge_corrections[i],5)
        return topology