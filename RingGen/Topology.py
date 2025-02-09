from copy import deepcopy
import numpy as np

class Topology:
    """
    Manages a GROMACS topology of a single molecule

    Attributes
    ----------
    name: str
    atoms: list of list (Nx7)
        atom_id, atom_type, res_id, res_name, atom_name, ch_group, charge, mass
    bonds: list of lists (Nx5)
        atom_id_1, atom_id_2, type, r_eq, k_f
    pairs: list of lists (Nx3)
        atom_id_1, atom_id_2, type
    angles: list of lists (Nx6)
        atom_id_1, atom_id_2, atom_id_3, type, th_eq, k_f
    dihedrals: list of lists (Nx8)
        atom_id_1, atom_id_2, atom_id_3, atom_id_4, type, gamma, k_f, n  
    """
    def __init__(self):
        self.name = "Mol"
        self.atoms = []
        self.bonds = []
        self.pairs = []
        self.angles = []
        self.dihedrals = []

    def read_file(self, file, name=None):
        """
        Function to create topology from .top file with GROMACS structure.
        Picks molecule with the provided name, or the first if no name given.
        
        Note
        -------
        Reading capability is handled by TopFile class.
        
        Parameters
        -------
        file : str
        name : str, optional
        
        Returns
        -------
        self: Allows method chaining.
        """
        temp_top_file = TopFile().read_file(file)
        if name is None:
            name = list(temp_top_file.moleculetypes.keys())[0]
        else:
            if name not in temp_top_file.moleculetypes:
                print("The selected molecule was not found in topology file.")
                return self
        new_topology = temp_top_file.moleculetypes[name]
        # replace current object with this new topology object 
        # and delete temporary TopFile
        self.__dict__.update(new_topology.__dict__)
        del temp_top_file

        return self

    def set_name(self, name):
        """
        Sets name of topology.
        
        Returns self: Allows method chaining.
        """
        self.name = name
        return self

    def reduce(self, residues, is_linker=False):
        """
        Reduces topology to only the selected residues (atoms and parameters).
        
        Parameters
        ----------
        residues : list of int
            List of residue_ids that will be kept in topology.
        is_linker : bool, optional
            Can reduce topology to only include terms that contain atoms in 
            all residues given in "residues". Useful to create linker 
            topologies to create larger topologies from fragments.
        
        Returns
        -------
        self: Allows method chaining.
        """
        new_atoms=[]
        for atom in self.atoms:
            if atom[2] in residues:
                new_atoms.append(atom)

        self.bonds = self._reduce_paras(self.bonds, 2, residues, is_linker)
        self.pairs = self._reduce_paras(self.pairs, 2, residues, is_linker)
        self.angles = self._reduce_paras(self.angles, 3, residues, is_linker)
        self.dihedrals = self._reduce_paras(self.dihedrals, 4, residues, 
                                            is_linker)
        self.atoms = new_atoms

        self._atom_number_correction()
        self._residue_correction(residues, is_linker)

        return self

    def get_total_charge(self):
        """
        Calculates net charge of atoms in topology.
        
        Returns total charge (float).
        """
        return sum([float(x[6]) for x in self.atoms])
    
    def find_atoms(self, 
                   atom_name=None, 
                   res_name=None, 
                   res_id=None, 
                   atom_idx=None):
        """
        Function to find atoms that match criteria
        
        Parameters
        ----------
        atom_name : str, optional
        res_name : str, optional
        res_id : int, optional
        atom_idx : int, optional

        Returns
        -------
        list of int
            List of idx of atoms that matched criteria.

        """
        if (atom_name is None and 
            res_name is None and 
            res_id is None and 
            atom_idx is None):
            return []
        else:
            return [x for x in self.atoms
                    if ((x[0] == atom_idx or atom_idx is None) and
                        (x[2] == res_id or res_id is None) and
                        (x[3] == res_name or res_name is None) and
                        (x[4] == atom_name or atom_name is None))]

    def set_charges(self, charges):
        """
        Function to set partial atomic charges from a list of floats.
        
        Returns self: Allows method chaining.
        """
        if len(charges) != len(self.atoms):
            print(("List of charges is not the right size."+
                   f"Nb. of atoms: {len(self.atoms)}"+
                   f"Length of charges list: {len(charges)}"))
            return self
        for i,charge in enumerate(charges):
            self.atoms[i][6] = str(f"{charge:.5f}")
        return self

  #Local functions
    def _reduce_paras(self, paras, length, residues, linker):
        """
        Filters list of parameters belonging to specified residues.
        
        Only parameters that contain atoms belonging to residues listed in
        "residues" are kept in new list.

        Parameters
        ----------
        paras : list of lists
            Complete list of parameters.
        length : int
            Selects how many atom_ids are given in a paramer definition.
        residues : list of int
            Contains res_ids that need to be kept.
        linker : bool
            If true, only parameters with atoms in all residues listed in
            "residues" are kept.

        Returns
        -------
        new_paras : list of lists
            Only relevant parameters remain in list.

        """
        new_paras=[]
        for para in paras:
            #need to subtract 1 to go from GROMACS to python(count from 1 vs 0) 
            atom_ids = np.array(para[:length], int) - 1
            residues_of_para = set([self.atoms[x][2] for x in atom_ids])
            if linker:
                if (all(res in residues for res in residues_of_para) 
                    and len(residues_of_para)==2):
                    new_paras.append(para)
            else:
                if all(res in residues for res in residues_of_para):
                    new_paras.append(para)
        return new_paras

    def _atom_number_correction(self):
        """
        Adjusts atom_ids to be consecutive (necessary after reducing topology).

        No return value.
        """
        atom_key={}
        #Creates dictionary where each old atom number is a key that links to the new atom number
        for i,atom in enumerate(self.atoms):
            atom_key.update({atom[0]:i+1})
            self.atoms[i][0]=i+1

        for para,para_nb in zip(["bonds","pairs","angles","dihedrals"],[2,2,3,4]):
            for i,line in enumerate(deepcopy(self.__dict__[para])):
                for j in range(para_nb):
                    self.__dict__[para][i][j]=atom_key[line[j]]

    def _residue_correction(self, residues=[], linker=False):
        """
        Adjusts res_ids to be consecutive (necessary after reducing topology).

        Requires information about the residues that were selected.

        No return value.
        """
        if residues == []:
            for atom in self.atoms:
                atom[2]=int(atom[2]-int(self.atoms[0][2])+1)
        elif len(residues) == 1:
            for atom in self.atoms:
                atom[2]=1
        elif len(residues) == 2 and linker:
            for atom in self.atoms:
                if atom[2] == residues[0]:
                    atom[2] = 1
                elif atom[2] == residues[1]:
                    atom[2]=2
        else:
            old_residues = [x[2] for x in self.atoms]
            old_residues = list(set(old_residues))
            residue_map = {x:i+1 for i,x in enumerate(old_residues)}
            for i,atom in enumerate(self.atoms):
                self.atoms[i][2] = residue_map[self.atoms[i][2]]
            
        
class TopFile:
    """
    Manages a GROMACS topology file with topologies of one or more molecules.
    
    Attributes
    ----------
    name: str
        Name of topology system, used in system section when written
    defaults: list of str
        list with "default" definitions required in GROMACS files
    atomtypes: list of str
        list with atomtype definitions
    system: list of str
        list with name of the system
    moleculetypes: dict
        contains the GROMACS topologies {name: RingGen.Topology}
    molecule_counter: dict
        keeps track of number of the molecules in the topology {name: int}
        
    Example
    -------
    Use by:
        TopFile().read_file(file: str)
    or:
        TopFile().create_topology(topology: RingGen.Topology)
    """
    
    def __init__(self):
        self.name = "System"
        self.atomtypes = []
        self.molecule_list = {}
        self.moleculetypes = {}

    def read_file(self, file_name):
        """
        Reads topology from a .top file

        Returns self: Allows method chaining.
        """
        file_sections = self._get_file_sections(file_name)
        self._process_file_sections(file_sections)
        return self

    def create_file(self, topology):
        """
        Creates empty TopFile object with a single topology (RingGen.Topology).

        Returns self: Allows method chaining.
        """
        self.moleculetypes.update({topology.name:topology})
        self.molecule_list.update({topology.name: 1})
        return self

    def write_file(self, file_name="topol.top", full_file = True):
        """
        Writes .top GROMACS topology file
        
        Parameters
        ----------
        file_name : str, optional
            Name of the written topology file. The default is "topol.top".
        full_file : bool, optional
            Determines if "defaults", "atomtypes", "system" and "molecules" 
            sections are written in topology file. The default is True.

        Returns
        -------
        None.

        """
        #Construct file contents first and then write
        file_content = []
        if full_file:
            file_content.extend(
                ["[ defaults ]\n",
                 ";nbfunc   comb-rule   gen-pairs   fudgeLJ   fudgeQQ\n",
                 " 1        2           yes         0.5       0.83333333\n",
                 "\n" ])
            
            if self.atomtypes == []:
                atomtypes = self._get_default_atom_types()
            else:
                atomtypes = self.atomtypes
            file_content.extend(
                ["[ atomtypes ]\n"]
                +[f"{x}\n" for x in atomtypes]
                +["\n"])
                
            for molecule in self.moleculetypes.values():
                file_content.extend(self._get_output_form(molecule))
                    
            file_content.extend(
                ["[ system ]\n",
                 f"{self.name} \n",
                 "\n"])
                
            file_content.extend(
                ["[ molecules ]\n",
                 "; Compound  #mols \n"]
                +[f"{x:<15} {y:>8}\n" for x,y in self.molecule_list.items()])
        else:
            for molecule in self.moleculetypes.values():
                file_content.extend(self._get_output_form(molecule))
            
        with open(file_name, "w+") as f:
            f.writelines(file_content)

    def add_moleculetype(self, topology):
        """
        Adds a molecule topology (RingGen.Topology) to topology file.
        
        Returns self: Allows method chaining.
        """
        self.moleculetypes.update({topology.name:topology})
        self.molecule_list.update({topology.name: 1})
        return self
    
    def set_molecule_number(self, name, number):
        """
        Sets number of molecules (with name) present in the system.

        Parameters
        ----------
        name : str
        number : int

        Returns
        -------
        self: Allows method chaining.
        """
        if name in self.molecule_list.keys():
            self.molecule_list.update({name: number})
        else:
            print("{name} not found in topology file.")
        
        return self


    def set_name(self, name):
        """
        Sets system name.
        
        Returns self: Allows method chaining.
        """
        self.name = name
        return self

    def set_atom_types(self, atom_types):
        """
        Set/overwrite the default atom types.
        
        Returns self: Allows method chaining.
        """
        self.atomtypes = [f"{x}" for x in atom_types]
        return self

    # Local functions
    @staticmethod
    def _get_file_sections(file):
        """
        Reads file and partitions it into the blocks of a gromacs .top file
        
        Parameters
        ----------
        file : list of str

        Returns
        -------
        sections : list of list of str
        """
        blocks = ["defaults", "atomtypes", "moleculetype", "system", "molecules"]
        section = []
        sections = []
        with open(file) as f:
            for line in f:
                if ("[" in line and "]" in line 
                    and any(block in line for block in blocks)):
                    sections.append(section)
                    section=[]
                section.append(line)
            sections.append(section)
        return sections
        
    def _process_file_sections(self, file_sections):
        """
        Assigns sections to properties of the TopFile object.
        
        Parameters
        ----------
        file : list of list of str

        Returns
        -------
        None.
        """
        for section in file_sections:
            if len(section) == 0:
                continue
            if "defaults" in section[0]:
                continue
            elif "atomtypes" in section[0]:
                self.atomtypes = section
            elif "system" in section[0]:
                for line in section:
                    if len(line.split()) > 0:
                        if line.split()[0][0] not in [";", "["]:
                            self.name = line.split()[0]
                            continue
            elif "molecules" in section[0]:
                self.molecule_list = {}
                for line in section:
                    if len(line.split()) > 0:
                        if line.split()[0][0] not in [";", "["]:
                            self.molecule_list.update({line.split()[0]: 
                                                       int(line.split()[1])})
            elif "moleculetype" in section[0]:
                new_topology = self._create_topology_from_txt(section)
                new_topology.set_name(self._get_name_from_txt(section))
                self.moleculetypes.update({new_topology.name:new_topology})
                
                
    def _create_topology_from_txt(self, topology_string):
        """
        Assigns parameters to a newly created topology from a toplogy file.

        Parameters
        ----------
        topology_string : list of str
            Contains the topology information, created by 
            self._process_file_sections().

        Returns
        -------
        new_topology : RingGen.Topology
        """
        new_topology = Topology()
        for para in ["atoms","bonds","pairs","angles","dihedrals"]:
            new_topology.__dict__[para] = \
                self._get_parameters_from_topology_string(para, 
                                                          topology_string)
        return new_topology

    @staticmethod
    def _get_name_from_txt(topology_string):
        """
        Returns second (non-commented) line of a [ moleculetype ].
        """
        for line in topology_string[1:]:
            if not (line.startswith(";")):# or line.startswith("[")):
                return line.split()[0]

    @staticmethod
    def _get_parameters_from_topology_string(para, topology_string):
        """
        Returns a list of lists containing all parameters of a certain 
        parametertype from a raw topology text file.
        """
        parameter_array = []
        Read = False
        for index, line in enumerate(topology_string):
            #only start when the right parameter indicator is found
            if f"[ {para} ]" in line:
                Read=True
            #stop reading when the next set of parameters is reached
            if Read and "[" in line and para not in line:
                break
            if Read:
                #remove parts of lines that are comments
                if ";" in line:
                    line = line[:line.find(";")]
                cols = []
                #stop when next section is found
                for word in line.split():
                    if "[" in word or "#" in word:
                        break
                    cols.append(word)
                if cols != []:
                    parameter_array.append(cols)
                    
        #convert str to int and float
        int_key = {"atoms": [1, 0, 1, 0, 0, 1, 0, 0],
                   "bonds": [1, 1, 1, 0, 0],
                   "pairs": [1, 1, 1],
                   "angles":[1, 1, 1, 1, 0, 0],
                   "dihedrals": [1, 1, 1, 1, 1, 0, 0, 0]}
        float_key = {"atoms": [0, 0, 0, 0, 0, 0, 1, 1],
                     "bonds": [0, 0, 0, 1, 1],
                     "pairs": [0, 0, 0],
                     "angles":[0, 0, 0, 0, 1, 1],
                     "dihedrals": [0, 0, 0, 0, 0, 1, 1, 1]}
        for j, parameter in enumerate(parameter_array):
            for i,value in enumerate(parameter):
                if int_key[para][i]:
                    parameter_array[j][i] = int(value)
                elif float_key[para][i]:
                    parameter_array[j][i] = float(value)
        return parameter_array

            
    @staticmethod
    def _get_default_atom_types():
        return ["c       12.010000  0.0000000  A     0.33996695     0.3598240",
                "c1      12.010000  0.0000000  A     0.33996695     0.8786400",
                "c2      12.010000  0.0000000  A     0.33996695     0.3599960",
                "c3      12.010000  0.0000000  A     0.33996695     0.4577296",
                "ca      12.010000  0.0000000  A     0.33996695     0.3598240",
                "cc      12.010000  0.0000000  A     0.33996695     0.3598240",
                "cd      12.010000  0.0000000  A     0.33996695     0.3598240",
                "ce      12.010000  0.0000000  A     0.33996695     0.3598240",
                "cf      12.010000  0.0000000  A     0.33996695     0.3598240",
                "cg      12.010000  0.0000000  A     0.33996695     0.8786400",
                "ch      12.010000  0.0000000  A     0.33996695     0.8786400",
                "ci      12.010000  0.0000000  A     0.33996695     0.4727920",
                "cp      12.010000  0.0000000  A     0.33996695     0.3598240",
                "cq      12.010000  0.0000000  A     0.33996695     0.3598240",
                "h1       1.008000  0.0000000  A      0.2471353     0.0656888",
                "h4       1.008000  0.0000000  A     0.25105526     0.0627600",
                "h5       1.008000  0.0000000  A     0.24214627     0.0627600",
                "ha       1.008000  0.0000000  A     0.25996425     0.0627600",
                "hc       1.008000  0.0000000  A     0.26495328     0.0656888",
                "hn       1.008000  0.0000000  A     0.10690785     0.0656888",
                "n1      14.010000  0.0000000  A     0.32499985     0.7112800",
                "na      14.010000  0.0000000  A     0.32499985     0.7112800",
                "nb      14.010000  0.0000000  A     0.32499985     0.7112800",
                "nc      14.010000  0.0000000  A     0.32499985     0.7112800",
                "nd      14.010000  0.0000000  A     0.32499985     0.7112800",
                "o       16.000000  0.0000000  A     0.29599219     0.8790600",
                "os      16.000000  0.0000000  A     0.30000123     0.7112800",
                "zn      65.400000  0.0000000  A     0.226          0.0133156",
                "si      28.010000  0.0000000  A     0.31680358     0.0627600",
                "XX       1.000000  0.0000000  A     0.0            0.0000000"]

    def _get_output_form(self, topology):
        """
        Creates writeable form of a Topology object. Returns list of str.
        """
        content=["[ moleculetype ]\n",
                 f"{topology.name}          3\n\n"]
        for para in ["atoms","bonds","pairs","angles","dihedrals"]:
            content.extend(self._get_parameter_output_form(para, topology))
        return content
    
    @staticmethod
    def _get_parameter_output_form(para, topology):
        """
        Creates writeable form of a section of "topology" named by "para".
        """
        content=[f"[ {para} ]\n"]
        for row in topology.__dict__[para]:
            if para == "atoms":
                content.append( f"{row[0]:>5}"
                               +f"{row[1]:>11}"
                               +f"{row[2]:>7}"
                               +f"{row[3]:>7}"
                               +f"{row[4]:>7}"
                               +f"{row[5]:>7}"
                               +f"{row[6]:>14.8f}"
                               +f"{row[7]:>10.3f}\n")
            elif para == "bonds":
                content.append(f"{row[0]:>7}"
                               +f"{row[1]:>7}"
                               +f"{row[2]:>7}"
                               +f"{row[3]:>14.7f}"
                               +f"{row[4]:>14.3f}\n")
            elif para == "pairs":
                content.append(f"{row[0]:>7}"
                               +f"{row[1]:>7}"
                               +f"{row[2]:>7}\n")
            elif para == "angles":
                content.append(f"{row[0]:>7}"
                               +f"{row[1]:>7}"
                               +f"{row[2]:>7}"
                               +f"{row[3]:>7}"
                               +f"{row[4]:>14.7f}"
                               +f"{row[5]:>14.7f}\n")
            elif para == "dihedrals":
                content.append(f"{row[0]:>7}"
                               +f"{row[1]:>7}"
                               +f"{row[2]:>7}"
                               +f"{row[3]:>7}"
                               +f"{row[4]:>7}"
                               +f"{row[5]:>14.7f}"
                               +f"{row[6]:>14.7f}"
                               +f"{row[7]:>4.1f}\n")
        content.append("\n")
        return content
