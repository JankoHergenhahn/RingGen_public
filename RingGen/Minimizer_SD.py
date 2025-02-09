import numpy as np
from collections import namedtuple

VdW = namedtuple('VdW','R epsilon comment')
VdWs = {
    'h1':VdW(1.3870,0.015,'Veenstra et al JCC,8,(1992),963'),
    'h2':VdW(1.2870,0.015,'Veenstra et al JCC,8,(1992),963'),
    'h3':VdW(1.1870,0.015,'Veenstra et al JCC,8,(1992),963'),
    'h4':VdW(1.4090,0.015,'Spellmeyer, one electrowithdr. neighbor'),
    'h5':VdW(1.3590,0.015,'Spellmeyer, two electrowithdr. neighbor'),
    'ha':VdW(1.4590,0.015,'Spellmeyer'),
    'hc':VdW(1.4870,0.015,'OPLS'),
    'hn':VdW(0.6000,0.015,'!Ferguson base pair geom.'),
    'ho':VdW(0.0000,0.000,'OPLS Jorgensen, JACS,110,(1988),1657'),
    'hp':VdW(0.6000,0.015,'same to hs (be careful !)'),
    'hs':VdW(0.6000,0.015,'W. Cornell CH3SH --> CH3OH FEP'),
    'hw':VdW(0.0000,0.000,'OPLS Jorgensen, JACS,110,(1988),1657'),
    'hx':VdW(1.1000,0.015,'Veenstra et al JCC,8,(1992),963'),
    'o':VdW(1.6612,0.210,'OPLS'),
    'oh':VdW(1.7210,0.210,'OPLS'),
    'os':VdW(1.6837,0.170,'OPLS ether'),
    'ow':VdW(1.7683,0.152,'TIP3P water model'),
    'c':VdW(1.9080,0.086,'OPLS'),
    'c1':VdW(1.9080,0.210,'cp C DLM 11/2007 well depth from OPLS'),
    'c2':VdW(1.9080,0.086,'sp2 atom in the middle of C=CD-CD=C'),
    'c3':VdW(1.9080,0.109,'OPLS'),
    'ca':VdW(1.9080,0.086,'OPLS'),
    'cc':VdW(1.9080,0.086,'OPLS'),
    'cd':VdW(1.9080,0.086,'OPLS'),
    'ce':VdW(1.9080,0.086,'OPLS'),
    'cf':VdW(1.9080,0.086,'OPLS'),
    'cg':VdW(1.9080,0.210,'DLM 12/2007 as c1'),
    'ch':VdW(1.9080,0.210,'DLM 12/2007 as c1'),
    'cp':VdW(1.9080,0.086,'OPLS'),
    'cq':VdW(1.9080,0.086,'OPLS'),
    'cu':VdW(1.9080,0.086,'OPLS'),
    'cv':VdW(1.9080,0.086,'OPLS'),
    'cx':VdW(1.9080,0.086,'OPLS'),
    'cy':VdW(1.9080,0.086,'OPLS'),
    'cz':VdW(1.9080,0.086,'OPLS'),
    'n':VdW(1.8240,0.170,'OPLS'),
    'n1':VdW(1.8240,0.170,'OPLS'),
    'n2':VdW(1.8240,0.170,'OPLS'),
    'n3':VdW(1.8240,0.170,'OPLS'),
    'n4':VdW(1.8240,0.170,'OPLS'),
    'na':VdW(1.8240,0.170,'OPLS'),
    'nb':VdW(1.8240,0.170,'OPLS'),
    'nc':VdW(1.8240,0.170,'OPLS'),
    'nd':VdW(1.8240,0.170,'OPLS'),
    'ne':VdW(1.8240,0.170,'OPLS'),
    'nf':VdW(1.8240,0.170,'OPLS'),
    'nh':VdW(1.8240,0.170,'OPLS'),
    'no':VdW(1.8240,0.170,'OPLS'),
    's':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    's2':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    's4':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    's6':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    'sx':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    'sy':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    'sh':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    'ss':VdW(2.0000,0.250,'W. Cornell CH3SH and CH3SCH3 FEPs'),
    'p2':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'p3':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'p4':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'p5':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'pb':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'pc':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'pd':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'pe':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'pf':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'px':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'py':VdW(2.1000,0.200,'JCC,7,(1986),230;'),
    'f':VdW(1.75,0.061,'Gough et al. JCC 13,(1992),963.'),
    'cl':VdW(1.948,0.265,'Fox, JPCB,102,8070,(98),flex.mdl CHCl3'),
    'br':VdW(2.02,0.420,'Junmei, 2010'),
    'i':VdW(2.15,0.50,'Junmei, 2010'),
    'zn':VdW(1.269, 0.00318, 'Hergenhahn 2021'),
}

class Minimizer:
    """
    Steepest descent optimizer using Gromacs topologies and GAFF force field.
    
    Attributes
    ----------
    topology: RingGen.Topology
        contains force field information
    coordinates: RingGen.Coordinates
        coordinates need to match order of topology        
    constrained_atoms: list of int
        atoms whose positions are constrained during minimizations
    transparent_atoms: list of int
        atoms for which non-bonded parameters are set to 0
    
    """
    def __init__(self, topology, coordinates, 
                 constrained_atoms = [], transparent_atoms = []):
        self.coordinates = coordinates.coordinates
        self._coordinates_shape = self.coordinates.shape
        self.process_topology(topology, transparent_atoms)
        self.constrained_atoms = constrained_atoms
        self.current_energy = self.get_energy(self.coordinates)

    def minimize(self, 
                 steps = 200, 
                 step_size = 5e-3, 
                 verbose = 0, 
                 step_size_limits = None):
        """
        Steepest descent algorithm with variable step size.

        Step size is increased if energy decreased in previous step and
        halfed if the energy increased. Optimization terminates when energy
        no longer decreases with the smallest allowed step size, or if the 
        maximum number of steps has been reached.

        Parameters
        ----------
        steps : int, optional
            Maximum number of iterations. The default is 200.
        step_size : float, optional
            Largest step size in any one step. The default is 5e-2.
        verbose : int, optional
            Level of verbosity. The default is 0.
        step_size_limits : list of float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if step_size_limits == None:
            step_size_limits = [0.1* step_size, 10 * step_size]
        for i in range(steps):
            self.forces = self.get_forces()
            force_lengths = np.linalg.norm(self.forces, axis = 1)
            step = self.forces/np.max(force_lengths) * step_size
            temp_coordinates = self.coordinates + step
            temp_energy = self.get_energy(temp_coordinates)

            if temp_energy < self.current_energy:
                self.coordinates = temp_coordinates
                self.current_energy = temp_energy
                step_size = min([1.1*step_size, step_size_limits[1]])
            else:
                if step_size == step_size_limits[0]:
                    print(f"Minimization has terminated after {i+1} steps. ",
                          "If output is not satisfactory, ",
                          "try decreasing the step size.", sep = "")
                    break
                step_size = max([0.5*step_size, step_size_limits[0]])


            if verbose == 1 and (i % 20 == 0):
                print(f"{i:>3}"
                      +f"{self.current_energy:>10.2f}"
                      +f"{np.mean(np.abs(self.forces)):>10.4f}"
                      +f"{step_size:>10.5f}")
            if verbose == 2:
                print(f"{i:>3}"
                      +f"{self.current_energy:>10.2f}"
                      +f"{np.mean(np.abs(self.forces)):>10.4f}"
                      +f"{step_size:>10.5f}")
        
    def get_coordinates(self):
        return self.coordinates
    
  #Local functions  
    def process_topology(self, topology, transparent_atoms):
        """
        Obtains all necessary information from topology file before 
        optimization starts to speed up calculations.
        """
        bond_indeces = []
        bond_parameters = []
        for bond in topology.bonds:
            bond_indeces.append(bond[:2])
            bond_parameters.append(bond[3:])
        self.bond_indeces = np.array(bond_indeces, np.int32) - 1
        self.bond_parameters = np.array(bond_parameters, np.float32)

        angle_indeces = []
        angle_parameters = []
        for angle in topology.angles:
            angle_indeces.append(angle[:3])
            angle_parameters.append(angle[4:])
        self.angle_indeces = np.array(angle_indeces, np.int32) - 1
        self.angle_parameters = np.array(angle_parameters, np.float32)

        dihedral_indeces = []
        dihedral_parameters = []
        for dihedral in topology.dihedrals:
            dihedral_indeces.append(dihedral[:4])
            dihedral_parameters.append(dihedral[5:])
        self.dihedral_indeces = np.array(dihedral_indeces, np.int32) - 1
        self.dihedral_parameters = np.array(dihedral_parameters, np.float32)


        ES_parameters = np.zeros(self._coordinates_shape[0])
        for atom_index, atom_parameters in enumerate(topology.atoms):
            ES_parameters[atom_index] = atom_parameters[6]
        self.ES_parameters = np.array(ES_parameters, np.float32)
        self.ES_parameters = (self.ES_parameters 
                              * self.ES_parameters[:, np.newaxis])

        #sigma**6,epsilon
        LJ_parameters = np.zeros((2,
                                  self._coordinates_shape[0],
                                  self._coordinates_shape[0])) 
        for atom_index, atom_parameters in enumerate(topology.atoms):
            exclusion_indeces = self.dihedral_indeces[
                np.where(self.dihedral_indeces == atom_index)[0]].ravel()
            exclusion_indeces = list(set(exclusion_indeces))
            for atom_index_2, atom_parameters_2 in enumerate(topology.atoms):
                sigma_1 = 2 * VdWs[atom_parameters[1]].R / 2**(1/6)
                sigma_2 = 2 * VdWs[atom_parameters_2[1]].R / 2**(1/6)
                LJ_parameters[0, atom_index, atom_index_2] = \
                    (0.5 * (sigma_1 + sigma_2))**6
                if (atom_index_2 in exclusion_indeces or
                    atom_index in transparent_atoms or 
                    atom_index_2 in transparent_atoms):
                    LJ_parameters[1, atom_index, atom_index_2] = 0
                    self.ES_parameters[atom_index, atom_index_2] = 0
                    self.ES_parameters[atom_index_2, atom_index] = 0
                else:
                    LJ_parameters[1, atom_index, atom_index_2] = \
                        4.184 * ( (VdWs[atom_parameters[1]]).epsilon 
                                 *(VdWs[atom_parameters_2[1]]).epsilon)**0.5
                    #self.ES_parameters[atom_index, atom_index_2] *= 0.83333333
                    #self.ES_parameters[atom_index_2, atom_index] *= 0.83333333

        self.LJ_parameters = LJ_parameters
        cols, rows = np.meshgrid(range(LJ_parameters.shape[1]), 
                                  range(LJ_parameters.shape[1]))
        self.mask = (rows < cols)    
    
    def get_energy(self, coordinates):
        energy = 0
        bond_values = np.linalg.norm(coordinates[self.bond_indeces[:,1]] 
                                     - coordinates[self.bond_indeces[:,0]], 
                                     axis=1)
        energy += np.sum(0.5 * self.bond_parameters[:,1] 
                         * (0.1 * bond_values - self.bond_parameters[:,0])**2)
        angle_values = get_angle(coordinates[self.angle_indeces[:,0]],
                                 coordinates[self.angle_indeces[:,1]],
                                 coordinates[self.angle_indeces[:,2]])
        energy += \
            np.sum(0.5 * self.angle_parameters[:,1] * 
                   (angle_values - np.pi/180 * self.angle_parameters[:,0])**2)

        dihedral_values = get_dihedral(coordinates[self.dihedral_indeces[:,0]],
                                       coordinates[self.dihedral_indeces[:,1]],
                                       coordinates[self.dihedral_indeces[:,2]],
                                       coordinates[self.dihedral_indeces[:,3]])
        energy += \
            np.sum(self.dihedral_parameters[:,1] 
                   *(1 + np.cos(self.dihedral_parameters[:,2] * dihedral_values 
                                 - self.dihedral_parameters[:,0]*np.pi/180)))

        distances = cdist(coordinates)
        distances[np.where(distances == 0)] = 100
        LJ_values = (4 * self.LJ_parameters[1] 
                     * (self.LJ_parameters[0]**2 / distances**12 
                        -self.LJ_parameters[0]  /  distances**6))
        energy += np.sum(LJ_values)
        ES_values = 1389 * self.ES_parameters / distances
        energy += np.sum(ES_values)

        return energy

    def get_forces(self):
        forces = np.zeros(self._coordinates_shape)
        forces += self.get_bond_forces()
        forces += self.get_angle_forces()
        forces += self.get_dihedral_forces()
        forces += self.get_non_covalent_forces()
        forces[self.constrained_atoms] *= 0
        return forces

    def get_bond_forces(self):
        forces = np.zeros(self._coordinates_shape)

        bond_values = np.linalg.norm(self.coordinates[self.bond_indeces[:,1]]
                                    -self.coordinates[self.bond_indeces[:,0]], 
                                    axis=1)
        distances = (0.1 * bond_values - self.bond_parameters[:,0])
        force_values = - 0.1 * self.bond_parameters[:,1] * distances
        force_directions = (self.coordinates[self.bond_indeces[:,1]] 
                            - self.coordinates[self.bond_indeces[:,0]])
        force_directions /= \
             np.linalg.norm(force_directions, axis=1)[:, np.newaxis]
        forces_1 = force_values[:, np.newaxis] * force_directions

        np.add.at(forces, self.bond_indeces[:,0], -forces_1)
        np.add.at(forces, self.bond_indeces[:,1], +forces_1)
        return forces

    def get_angle_forces(self):
        forces = np.zeros(self._coordinates_shape)
        angle_values = get_angle(self.coordinates[self.angle_indeces[:,0]],
                                 self.coordinates[self.angle_indeces[:,1]],
                                 self.coordinates[self.angle_indeces[:,2]])
        force_values = \
            (- self.angle_parameters[:,1] 
             * (angle_values - np.pi/180 * self.angle_parameters[:,0]))
        force_directions_1, force_directions_2 = get_angle_force_directions(
            self.coordinates[self.angle_indeces[:,0]],
            self.coordinates[self.angle_indeces[:,1]],
            self.coordinates[self.angle_indeces[:,2]])
        
        force_directions_1 /= \
            np.linalg.norm(force_directions_1, axis = 1)[:, np.newaxis]
        force_directions_2 /= \
            np.linalg.norm(force_directions_2, axis = 1)[:, np.newaxis]
        scaling_1 = np.linalg.norm(self.coordinates[self.angle_indeces[:,0]] 
                                   - self.coordinates[self.angle_indeces[:,1]], 
                                   axis = 1)
        scaling_2 = np.linalg.norm(self.coordinates[self.angle_indeces[:,1]] 
                                   - self.coordinates[self.angle_indeces[:,2]], 
                                   axis = 1)
        forces_1 = force_directions_1 * (force_values/scaling_1)[:, np.newaxis]
        forces_2 = force_directions_2 * (force_values/scaling_2)[:, np.newaxis]
        
        np.add.at(forces, self.angle_indeces[:,0], forces_1)
        np.add.at(forces, self.angle_indeces[:,1], -(forces_1+forces_2))
        np.add.at(forces, self.angle_indeces[:,2], forces_2)
        return forces

    def get_dihedral_forces(self):
        forces = np.zeros(self._coordinates_shape)

        dihedral_values = get_dihedral(
            self.coordinates[self.dihedral_indeces[:,0]],
            self.coordinates[self.dihedral_indeces[:,1]],
            self.coordinates[self.dihedral_indeces[:,2]],
            self.coordinates[self.dihedral_indeces[:,3]])
        force_values = (self.dihedral_parameters[:,1] 
                        * self.dihedral_parameters[:,2] 
                        * np.sin(self.dihedral_parameters[:,2]*dihedral_values 
                                 - self.dihedral_parameters[:,0]*np.pi/180))
        force_directions_1, force_directions_2 = get_dihedral_force_directions(
            self.coordinates[self.dihedral_indeces[:,0]],
            self.coordinates[self.dihedral_indeces[:,1]],
            self.coordinates[self.dihedral_indeces[:,2]],
            self.coordinates[self.dihedral_indeces[:,3]])
        lever_1, lever_2 = get_lever_for_dihedral(
            self.coordinates[self.dihedral_indeces[:,0]],
            self.coordinates[self.dihedral_indeces[:,1]],
            self.coordinates[self.dihedral_indeces[:,2]],
            self.coordinates[self.dihedral_indeces[:,3]])
        scaling_1 = (force_values/
                     (lever_1 * np.linalg.norm(force_directions_1, axis = 1)))
        scaling_2 = (force_values/
                     (lever_2 * np.linalg.norm(force_directions_2, axis = 1)))
        forces_1 = force_directions_1 * scaling_1[:, np.newaxis]
        forces_2 = force_directions_2 * scaling_2[:, np.newaxis]

        np.add.at(forces, self.dihedral_indeces[:,0], forces_1)
        np.add.at(forces, self.dihedral_indeces[:,1], -forces_1)
        np.add.at(forces, self.dihedral_indeces[:,2], -forces_2)
        np.add.at(forces, self.dihedral_indeces[:,3], forces_2)
        
        return forces

    def get_non_covalent_forces(self):
        forces = np.zeros(self._coordinates_shape)

        vectors = self.coordinates[:, np.newaxis] - self.coordinates
        distances = np.linalg.norm(vectors, axis = 2)
        distances[np.diag_indices_from(distances)] = 100
        d = np.array(distances, np.float32)
        _d = 1/d
        _d_2 = _d * _d
        _d_6 = _d_2 * _d_2 * _d_2
        _d_12 = _d_6 * _d_6
        _d_7 = _d_6 * _d
        _d_13 = _d_12 * d
        
        LJ_values = 2 * 4 * self.LJ_parameters[1] \
            * (12 * self.LJ_parameters[0]**2 * _d_13 
               - 6 * self.LJ_parameters[0] * _d_7)
        ES_values = 2778 * self.ES_parameters * _d_2
        forces = np.sum(((LJ_values + ES_values) * _d)[:,:,np.newaxis] 
                        * vectors, axis = 1)
        
        return forces


def get_angle(coord_1, coord_2, coord_3):
    vec_1 = coord_1 - coord_2
    vec_1 /= (np.expand_dims(np.linalg.norm(vec_1, axis = 1), 1) 
              @ np.array([[1,1,1]]))
    vec_2 = coord_3 - coord_2
    vec_2 /= (np.expand_dims(np.linalg.norm(vec_2, axis = 1), 1) 
              @ np.array([[1,1,1]]))
    return np.arccos( np.sum(vec_1 * vec_2, axis=1) )

def get_dihedral(A,B,C,D):
        u1=B-A
        u2=C-B
        u3=D-C
        u4 = np.cross(u2,u3)

        x = np.sum((np.expand_dims(np.linalg.norm(u2, axis = 1), 1) 
                    @ np.array([[1,1,1]]))* u1 * u4, axis=1)
        y = np.sum(np.cross(u1,u2) * u4, axis=1)

        return np.arctan2(x,y)

def get_angle_force_directions(coord_1, coord_2, coord_3):
    u1=coord_1-coord_2
    u2=coord_3-coord_2
    u3=np.cross(u1,u2)

    v1 = np.cross(u1,u3)
    v2 = np.cross(u3,u2)
    v1 /= np.linalg.norm(v1, axis = 1)[:, np.newaxis]
    v2 /= np.linalg.norm(v2, axis = 1)[:, np.newaxis]

    return v1, v2

def get_dihedral_force_directions(A,B,C,D):
    u1=B-A
    u2=C-B
    u3=D-C

    v1 = np.cross(-u1,u2)
    v2 = np.cross(u2,u3)

    v1 /= np.linalg.norm(v1, axis = 1)[:, np.newaxis]
    v2 /= np.linalg.norm(v2, axis = 1)[:, np.newaxis]

    return v1, v2

def get_lever_for_dihedral(A,B,C,D):
    u1=B-A
    u2=C-B
    u3=D-C

    v1 = u2 / np.linalg.norm(u2, axis = 1)[:, np.newaxis]
    a = np.sum(u1 * v1, axis=1)
    lever_1 = np.sqrt(np.linalg.norm(u1, axis = 1)**2 - a**2)

    a = np.sum(u3 * v1, axis=1)
    lever_2 = np.sqrt(np.linalg.norm(u3, axis = 1)**2 - a**2)


    return lever_1, lever_2

def cdist(coordinates):
    nb_atoms=len(coordinates)
    distances = np.zeros((nb_atoms,nb_atoms))
    for i in range(nb_atoms):
        distances[:,i] = np.linalg.norm(coordinates-coordinates[i], axis = 1)
    return distances
        
        
        
        
        
        
        
        