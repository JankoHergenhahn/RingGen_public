"""
Commonly used utility functions used in RingGen
"""
import math
import numpy as np

def get_distance(A,B):
    """ Returns distance between two points A and B
    """
    return np.linalg.norm(np.array(A) - np.array(B)) 

def get_angle(A,B,C):
    """ Returns angle between three points A, B (centre) and C in radians
    """
    u1 = A-B
    u2 = C-B
    return np.arccos(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))

def get_dihedral(A,B,C,D):
    """ Returns dihedral angle between four points A, B, C and D in radians
    """
    u1 = B-A
    u2 = C-B
    u3 = D-C
    x = np.dot(np.linalg.norm(u2)*u1,np.cross(u2,u3))
    y = np.dot(np.cross(u1,u2),np.cross(u2,u3))
    return math.atan2(x,y)

def get_COM(coordinates, selection=None):
    """ Returns centre of mass (COM) of a list of coordinates.
    Can provide (optional) list of indices to only get COM of those coordinates.
    """
    coordinates = np.array(coordinates)
    if selection == None:
        return np.mean(coordinates,axis = 0)
    else:
        return np.mean(coordinates[selection],axis = 0)

def get_normal_vector(coordinates):
    """ Returns (normalized) normal vector to the mean plane of a list 
    of coordinates determined by single value decomposition (SVD).
    """
    points = coordinates.T
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))
    normal_vector = svd[0][:, -1]
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector

def get_rot_mat(axis_0, angle):
    """
    Return rotation matrix for a given axis and angle.

    Parameters
    ----------
    axis_0 : 1x3 iterable
        Vector of rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    rotation_matrix : 3x3 numpy array

    """
    #need unit vector
    axis = axis_0 / np.linalg.norm(axis_0)
    rotation_matrix = np.zeros((3,3))
    rotation_matrix[0,0] = math.cos(angle)+(axis[0]**2)*(1-math.cos(angle))
    rotation_matrix[0,1] = axis[0]*axis[1]*(1-math.cos(angle))-axis[2]*math.sin(angle)
    rotation_matrix[0,2] = axis[0]*axis[2]*(1-math.cos(angle))+axis[1]*math.sin(angle)
    rotation_matrix[1,0] = axis[1]*axis[0]*(1-math.cos(angle))+axis[2]*math.sin(angle)
    rotation_matrix[1,1] = math.cos(angle)+(axis[1]**2)*(1-math.cos(angle))
    rotation_matrix[1,2] = axis[1]*axis[2]*(1-math.cos(angle))-axis[0]*math.sin(angle)
    rotation_matrix[2,0] = axis[2]*axis[0]*(1-math.cos(angle))-axis[1]*math.sin(angle)
    rotation_matrix[2,1] = axis[2]*axis[1]*(1-math.cos(angle))+axis[0]*math.sin(angle)
    rotation_matrix[2,2] = math.cos(angle)+(axis[2]**2)*(1-math.cos(angle))
    return rotation_matrix

def rotate_point(point,axis,angle):
    """
    Rotates a single point [x,y,z] around an axis [x,y,z] by an angle

    Parameters
    ----------
    point : 1x3 iterable
        [x,y,z] coordinate.
    axis : 1x3 iterable
        Vector of rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    rotated point as 1x3 numpy array

    """
    rotation_matrix = get_rot_mat(axis, angle)
    return np.matmul(point,rotation_matrix)

def rotate_array(array, axis, angle):
    """
    Rotates every point of an array around an axis [x,y,z] by an angle

    Parameters
    ----------
    array : Nx3 iterable
        List of [x,y,z] coordinates.
    axis : 1x3 iterable
        Vector of rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    List of rotated points as Nx3 numpy array

    """
    rotation_matrix = get_rot_mat(axis, angle)
    return np.matmul(array, rotation_matrix)

def find_sub_systems(connectivity):
    """
    Given a cpnnectivity matrix of a graph, this function finds all subgraphs 
    that are not connected with each other.
    
    Parameters
    ----------
    connectivity : NxN array
        Connectivity matrix C; C[i,j] = 1 if nodes i and j are connected and 0 
        otherwise. C[i,i] should be 0.

    Returns
    -------
    systems : list
        List of list of the indices belonging to the different subsystems.
        
    Example
    -------
    C = np.array([[0,1,0,0,1], 
                  [1,0,0,0,0], 
                  [0,0,0,1,0], 
                  [0,0,1,0,0], 
                  [1,0,0,0,0]])
    find_sub_systems([[0,1,0,0]]) --> [[0,1,4], [2,3]]
    """
    nb_atoms = len(connectivity)
    systems = []
    assigned_atoms = []
    while len(assigned_atoms) < nb_atoms:
        system = []
        unassigned_atoms = [x for x in range(nb_atoms) 
                            if x not in assigned_atoms]        
        atoms_to_check = [unassigned_atoms[0]]
        while len(atoms_to_check) > 0:
            active_atom = atoms_to_check.pop(0)
            system.append(active_atom)
            assigned_atoms.append(active_atom)
            connected_atoms = np.where(connectivity[active_atom] == 1)[0]
            connected_atoms = [x for x in connected_atoms 
                               if ((x not in system) 
                                   and (x not in atoms_to_check))]
            atoms_to_check.extend(connected_atoms)
        system.sort()
        systems.append(system)
        
    #sort subsystems by the lowest indices of the subsystems
    systems.sort(key= lambda x: min(x))

    return systems