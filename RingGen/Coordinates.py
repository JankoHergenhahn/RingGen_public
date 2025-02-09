import numpy as np
from .Tools import rotate_array, find_sub_systems

class Coordinates:
    """
    Coordinates object for managing molecular coordinates.

    Attributes
    ----------
    elements: list of str
        List of elements of the atoms.
    coordinates: Nx3 numpy array
        Array of coordinates of the atoms
    connectivity: NxN numpy array
        Connectivity C of molecular graph. C[i,j] = 1 if nodes i and j are 
        connected and 0 otherwise. C[i,i] should be 0.
    """
    def __init__(self):
        self.elements = []
        self.coordinates = np.empty((0,3))
        self.connectivity = np.empty((0,0))

  #methods for input/output
    def read_file(self, file, create_connectivity=True, select_molecule=None):
        """
        Function to read coordinates from readable files.

        Parameters
        ----------
        file : str
            File name. File type is determined from file ending.
        create_connectivity : bool, optional
            Determins if connectivity graph is created. The default is True.
        select_molecule : int, optional
            If connectivity graph is disconnected (i.e. multiple molecules are 
            preset) can specify molecule. The default is None.

        Returns
        -------
        self: Allows method chaining.
        """
        
        #check that valid file type
        file_type = file.split(".")[-1]
        if file_type not in ["xyz","pdb","gro", "mol2"]:
            return print("File type not supported.")
        else:
        #use correct function to read file
            if file_type == "xyz":
                self = self.read_xyz(file)
            elif file_type == "pdb":
                self = self.read_pdb(file)
            elif file_type == "gro":
                self = self.read_gro(file)
            elif file_type == "mol2":
                self = self.read_mol2(file)

        if create_connectivity:
            self.create_connectivity()

        #only select molecule if index specified
        if select_molecule is not None:
            self._molecule_selection(select_molecule)
        return self

    def read_xyz(self, file):
        """
        Reads .xyz files.

        Parameters
        ----------
        file : str
            File name.

        Returns
        -------
        self: Allows method chaining.
        """
        self.elements = []
        self.coordinates = np.empty((0,3))
        with open(file) as f:
            for index,line in enumerate(f):
                if index>1 and line!="":
                    cols = line.split()
                    self.elements.append(cols[0])
                    c_temp = [float(cols[1]), float(cols[2]), float(cols[3])]
                    self.coordinates = np.append(self.coordinates,
                                                 [c_temp], 
                                                 axis=0)
        return self

    def read_pdb(self, file, letter_convention = "mixed_case"):
        """
        Reads .pdb files.

        Parameters
        ----------
        file : str
            File name.

        Returns
        -------
        self: Allows method chaining.
        """
        self.elements = []
        self.coordinates = np.empty((0,3))
        with open(file) as f:
            for index,line in enumerate(f):
                if "ATOM" in line or "HETATM" in line:
                    self.elements.append(line[12:16].strip())
                    c_temp = [float(line[30:38].strip()),
                              float(line[38:46].strip()), 
                              float(line[46:54].strip())]
                    self.coordinates = np.append(self.coordinates,
                                                 [c_temp],
                                                 axis=0)
        self.elements = self._correct_elements(self.elements, letter_convention)
        return self

    def read_gro(self, file):
        """
        Reads .gro files.

        Parameters
        ----------
        file : str
            File name.

        Returns
        -------
        self: Allows method chaining.
        """
        self.elements = []
        self.coordinates = np.empty((0,3))
        with open(file) as f:
            for index,line in enumerate(f):
                if index>1 and len(line.split())>=6:
                    cols = line.split()
                    self.elements.append(cols[1])
                    c_temp = [float(cols[3])*10,
                              float(cols[4])*10,
                              float(cols[5])*10]
                    self.coordinates = np.append(self.coordinates,
                                                 [c_temp],
                                                 axis=0)
        self.elements = self._correct_elements(self.elements)
        return self

    def read_mol2(self, file):
        """
        Reads .mol2 files.

        Parameters
        ----------
        file : str
            File name.

        Returns
        -------
        self: Allows method chaining.
        """
        self.elements = []
        self.coordinates = np.empty((0,3))
        read = False
        with open(file) as f:
            for index,line in enumerate(f):
                if "@<TRIPOS>BOND" in line:
                    read = False
                if read:
                    cols = line.split()
                    self.elements.append(cols[-1])
                    c_temp = [float(cols[-4]),
                              float(cols[-3]),
                              float(cols[-2])]
                    self.coordinates = np.append(self.coordinates,
                                                 [c_temp],
                                                 axis=0)
                if "@<TRIPOS>ATOM" in line:
                    read = True
        return self


    def write_xyz(self, output_name='out.xyz', comment="", precision = 6):
        """
        Write .xyz format coordinate file.

        Parameters
        ----------
        output_name : str, optional
            File name of written file. The default is 'out.xyz'.
        comment : str, optional
            Appears on second line in .xyz file. The default is "".
        precision : int, optional
            Number of decimal points for coordinates. The default is 6.

        No return value.
        """
        spacing = precision + 6
        content = [f"{len(self.elements)}\n", 
                   comment+"\n"]
        for atom,coordinate in zip(self.elements, self.coordinates):
            content.append(f"{atom:>5}"
                           +f"{coordinate[0]:>{spacing}.{precision}f}"
                           +f"{coordinate[1]:>{spacing}.{precision}f}"
                           +f"{coordinate[2]:>{spacing}.{precision}f}\n")
        with open(output_name, "w+") as edit_file:
            edit_file.writelines(content)

    def write_pdb(self, output_name='out.pdb', topology=None):
        """
        Write .pdb format coordinate file.

        Parameters
        ----------
        output_name : str, optional
            File name of written file. The default is 'out.xyz'.
        topology : RingGen.Topology, optional
            Topology used to set atom and residue names.

        No return value.
        """
        #Get properties from topology or create default values
        if topology==None:
            print("No topology given to write PDB file. "
                  "Uses default res ids and names.")
            top_name = "MOL"
            res_ids = [1 for x in self.elements]
            res_names = ["MOL" for x in self.elements]
            
            atom_names=[]
            label_counter_dict = {}
            for element in self.elements:
                if element in label_counter_dict:
                    label_counter_dict[element]+=1
                else:
                    label_counter_dict.update({element:1})
                if label_counter_dict[element]<10:
                    label = f"{element}0{label_counter_dict[element]}"
                else:
                    label = f"{element}{label_counter_dict[element]}"
                atom_names.append(label)
        else:
            top_name = topology.name
            res_ids = [x[2] for x in topology.atoms]
            res_names = [x[3] for x in topology.atoms]
            atom_names = [x[4] for x in topology.atoms]

        #create list of text ouput
        content=[f"COMPND    {top_name}\n"]
        for i,coordinate in enumerate(self.coordinates):
            new_line = (f"HETATM{i+1:>5}"
                       +f"{atom_names[i]:>5}"
                       +f"{res_names[i]:>4}"
                       +f"{res_ids[i]:>6}"
                       +f"{coordinate[0]:>12.3f}"
                       +f"{coordinate[1]:>8.3f}"
                       +f"{coordinate[2]:>8.3f}"
                       +f"{self.elements[i]:>24}\n")
            content.append(new_line)

        with open(output_name, "w+") as edit_file:
            edit_file.writelines(content)

  #methods for manipulation
    def set(self,elements,coordinates):
        """
        Sets elements and coordinates.
        
        Returns self: Allows method chaining.
        """
        self.elements = elements
        self.coordinates = coordinates
        return self

    def set_elements(self,elements):
        """
        Sets elements.
        
        Returns self: Allows method chaining.
        """
        self.elements = elements
        return self

    def set_coordinates(self,coordinates):
        """
        Sets coordinates.
        
        Returns self: Allows method chaining.
        """
        self.coordinates = coordinates
        return self
    
    def add(self, coordinates):
        """
        Adds Coordinates object to this Coordinates object.

        Parameters
        ----------
        coordinates : RingGen.Coordinates
            Object that needs to contain elements and coordinates attributes.

        Returns
        -------
        self: Allows method chaining.
        """
        self.coordinates = np.vstack([self.coordinates, 
                                      coordinates.coordinates])
        self.elements = self.elements + coordinates.elements
        return self
    
    def rotate(self,axis=[0,0,1],angle=np.pi):
        """
        Rotates coordinates around "axis" by "angle"
        
        Parameters
        ----------
        axis : 1x3 iterable, optional
            Axis of rotation. The default is [0,0,1].
        angle : float, optional
            Angle in radians. The default is np.pi.

        Returns
        -------
        self: Allows method chaining.
        """
        self.coordinates = rotate_array(self.coordinates, axis, angle)
        return self

    def translate(self, vector):
        """
        Translates coordinates by 'vector' (1x3 iterable)
        
        Returns self: Allows method chaining.
        """
        self.coordinates += np.array(vector)
        return self

    def remove_coordinates(self, indeces):
        """
        Removes atoms as specified by indices (list of int).
        
        Returns self: Allows method chaining.
        """
        self.coordinates = np.delete(self.coordinates,indeces, axis=0)
        self.elements = list(np.delete(np.array(self.elements), indeces))
        return self
    
    def create_connectivity(self, tolerance=1.05):
        """
        Function to create connectivity matrix C based on the coordinates.
        
        C[i,j] = 1 if nodes i and j are connected and 0 otherwise. 
        C[i,j] is 0.
        Connections are determoned by comparing distances between atoms to
        sum of covalent radii multiplied by a tolerance factor
        

        Parameters
        ----------
        tolerance : float, optional
            Determines cut-off for determining if atoms are connected. 
            The default is 1.05.

        Returns
        -------
        self: Allows method chaining.
        """
        cov_radii_dict = {  "H": 0.37, 
                            "B": 0.90,
                            "C": 0.77, 
                            "N": 0.75, 
                            "O": 0.73,
                            "F": 0.71,
                            "Cl":0.99,
                            "Br":1.14,
                            "I": 1.33,
                            "Al":1.40,
                            "Si":1.18,
                            "P": 1.10,
                            "S": 1.02,
                            "Li":0.76,
                            "Na":1.02,
                            "K" :1.38,
                            "Rb":1.52,
                            "Cs":1.67,
                            "Be":0.45,
                            "Mg":0.72,
                            "Ca":1.00,
                            "Sr":1.18,
                            "Ba":1.35,
                            "Sc":1.64,
                            "Ti":1.47,
                            "V": 1.35,
                            "Cr":1.29,
                            "Mn":1.37,
                            "Fe":1.26,
                            "Co":1.25,
                            "Ni":1.25,
                            "Cu":1.28,
                            "Zn":1.37,
                            "Pd":1.37,
                            "Pt":1.39,
                            "Ag":1.44,
                            "Au":1.44}
        if any([x not in cov_radii_dict.keys() for x in self.elements]):
            print("Unknown atoms found. Will guess covalent radius as 1 A.")
        
        nb_atoms = len(self.elements)
        cov_radii = np.array([cov_radii_dict.get(ele, 1.0) * tolerance 
                              for ele in self.elements])
        #creates a NxN array of covalent radii sums
        cov_radii_sum = cov_radii[:, np.newaxis] + cov_radii

        distances = np.zeros((nb_atoms,nb_atoms))
        for i in range(nb_atoms):
            distances[:,i] = np.linalg.norm(self.coordinates
                                            -self.coordinates[i], axis = 1)
            distances[i,i] = 1000 #set to very large value to get C[i,i] = 0
        self.connectivity = np.array(distances < cov_radii_sum, int)

        return self

    def _correct_elements(self, elements, letter_convention = "mixed_case"):
        """
        Function to change atom labels to standard atom names.
        
        Only necessary when importing from certain file types. Tries to match 
        the found atom labels to the correct elements.

        Parameters
        ----------
        elements : list of str
            List of elements of the atoms.
        letter_convention : str, optional
            letter_convention of the origin file: "mixed_case" or "upper_case". 
            The default is "mixed_case".

        Returns
        -------
        corrected_elements : list of str
            Corrected list of elements of the atoms.

        """
        one_letter_elements = ["H", "B", "C", "N", "O", "F", "P", "S", "I"]
        two_letter_elements = ["Si", "Cl", "Br",
                               "Cr","Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Pd"]
        two_letter_elements_upper = [x.upper() for x in two_letter_elements]

        corrected_elements = []
        for ele in elements:
            if ((ele[:2] in two_letter_elements and 
                 letter_convention == "mixed_case") 
                or
                (ele[:2] in two_letter_elements_upper and 
                 letter_convention == "upper_case")):
                corrected_elements.append(ele[0].upper() + ele[1].lower())
            elif ele[0] in one_letter_elements:
                corrected_elements.append(ele[0])
            else:
                corrected_elements.append("X")
                print("Found unknown atom type.")

        return corrected_elements

    def _molecule_selection(self, molecule_nb):
        """
        Selects on of the molecules in an input file.
        
        If connectivity gives rise to multiple subsystems, can select one of 
        those subsystems.

        Parameters
        ----------
        molecule_nb : int
            Index of the molecule. Molecules are ordered by their lowest index.

        No return value.
        """
        subsystems = find_sub_systems()
        if len(subsystems) == 1:
            print("Only one molecule was found.")
            return
        elif len(subsystems) < molecule_nb+1:
            print("Too few molecules were found to select specified molecule.")
            return
        
        self.coordinates = self.coordinates[subsystems[molecule_nb]]
        self.elements = list(np.array(self.elements)[subsystems[molecule_nb]])
        self.create_connectivity()
