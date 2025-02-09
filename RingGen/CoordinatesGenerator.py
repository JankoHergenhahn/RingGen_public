import numpy as np
from copy import deepcopy

from .Coordinates import Coordinates
from .Tools import rotate_point, get_distance

class CoordinatesGenerator:
    """
    Manages generation of structure coordinates based on coordinate fragments.
    
    Attributes
    ----------
    pattern : list of str
        Coordinates are created as specified by this list of fragment names.
    is_cyclic : bool, optional
        Determines if end-points should join up. Default is False.
    default_coordinate_fragments : dict of RingGen.Coordinates
        Coordinate fragments that are used to build the coordinates.
    """
    def __init__(self, 
                 pattern, 
                 is_cyclic = False, 
                 default_coordinate_fragments = {}):
        self.pattern = pattern
        self.is_cyclic = is_cyclic
        self.default_coordinate_fragments = default_coordinate_fragments
        self._coordinates = None

    def build(self, **kwargs):
        """
        Builds coordinates based on the defined pattern.

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
            
        Returns
        -------
        structure_coordinates : RingGen.Coordinates
            Coordinates of the generated structure
        """
        has_custom_twist = ("custom_twist" in kwargs.keys())
        has_end_to_end_distance = ("end_to_end_distance" in kwargs.keys())
        has_radius = ("radius" in kwargs.keys())
        
        default_settings = {"custom_twist": [],
                            "end_to_end_distance": 0.0,
                            "radius": 0.0}
        kwargs = {**default_settings, **kwargs}
        
        #always creates linear chain initially
        if has_custom_twist:
            structure_coordinates = self._build_chain(kwargs["custom_twist"])
        else:
            structure_coordinates = self._build_chain(self._get_twist())
            
        #modify linear chain
        if self.is_cyclic:
            radius = self._get_end_to_end_distance()/(2*np.pi)
            structure_coordinates = \
                self.project_to_cylinder(structure_coordinates, radius)
        else:
            if has_radius:
                radius = kwargs["radius"]
                structure_coordinates = \
                    self.project_to_cylinder(structure_coordinates, radius)
            elif has_end_to_end_distance:
                radius = self._get_radius_from_end_to_end_distance(
                    kwargs["end_to_end_distance"])
                if radius >= 0:
                    structure_coordinates = \
                        self.project_to_cylinder(structure_coordinates, radius)
        self._coordinates = structure_coordinates
        return structure_coordinates
       
    def get_coordinates(self):
        return self._coordinates
         
    def _build_chain(self, twist):
        """
        Generates coordinates of the generated structure in linear chain.
        """
        distance=0
        structure_coordinates = Coordinates()
        for i,block in enumerate(self.pattern):
            temp_coordinates = deepcopy(self.default_coordinate_fragments[block])
            temp_coordinates.translate([distance, 0.0, 0.0])
            distance += temp_coordinates.distance
            temp_coordinates.rotate([1,0,0], 2*np.pi*(twist[i]))
            structure_coordinates.add(temp_coordinates)
        return structure_coordinates
            
    def _get_twist(self):
        """
        Obtains twist for chain generation. Selects appropriate function.
        """
        if self.is_cyclic:
            return self._get_twist_ring()
        else:
            return self._get_twist_chain()
        
    def _get_twist_chain(self):
        """
        Obtains twist for chain generation if structure is linear. 
        Straighforward based on list of fragments that should be co-planar.
        """
        flat_cons = self._get_flat_connections()
        if all(flat_cons):
            return [0 for x in self.pattern]
        twist = [0]
        for con in flat_cons:
            if con:
                twist.append(twist[-1])
            else:
                twist.append(twist[-1] + 0.25)
        return twist
        
    def _get_twist_ring(self):
        """
        Obtains twist for chain generation if structure is cyclic. 
        Based on list of fragments that should be co-planar, special 
        considerations are necessary to join up the end points.
        """
        flat_cons = self._get_flat_connections()
        
        if all(flat_cons):
            return [0 for x in self.pattern]
        
        if not any(flat_cons):
            nb_fragments = len(self.pattern)
            if nb_fragments % 2 == 0:
                total_twist = 0.25*(nb_fragments-1)
            else:
                total_twist = 0.25*(nb_fragments)
            return 0.125 + np.linspace(0, total_twist, nb_fragments)
            
        shiftBlockList=0
        #Shift blocks for easier manipilation so that array ends with flat connection
        if self.is_cyclic and any(flat_cons):
            while flat_cons[-1] == False:
                flat_cons.append(flat_cons.pop(0))
                shiftBlockList += 1
            
        seperated_units_indeces = [0]
        for i,con in enumerate(flat_cons):
            if con:
                seperated_units_indeces.append(i+1)
        
        twist = []
        last_twist = 0.125
        for i in range(len(seperated_units_indeces)-1):
            nb_fragments = seperated_units_indeces[i+1] - seperated_units_indeces[i]
            if nb_fragments == 1:
                twist.append(last_twist)    
            elif 2 <= nb_fragments <= 5: 
                twist.extend([last_twist + x*0.25 for x in range(nb_fragments)])
            else:
                if nb_fragments % 2 == 0:
                    total_twist = 0.25*nb_fragments
                else:
                    total_twist = 0.25*(nb_fragments+1)
                    
                twist.extend(np.linspace(0, total_twist, nb_fragments))
            last_twist = twist[-1]
        #transform back to correct order
        if shiftBlockList:
            for i in range(shiftBlockList):
                flat_cons.insert(0,flat_cons.pop(-1))
                twist.insert(0,twist.pop(-1))
        return twist                
        
    def _get_flat_connections(self):
        """
        Determines which fragments (neighbours) should be coplanar.
        
        Determined based on steric slash of fragments if they were coplanar.

        Returns
        -------
        flat_cons : list of bool
        """
        left_edge = []
        for block_name in self.pattern:
            block = self.default_coordinate_fragments[block_name]
            if len(block.connecting_atoms) >= 1:
                con_atom = block.connecting_atoms[0]
                con_atom_coord = block.coordinates[con_atom]
                furthest_coord = min([x[0] - con_atom_coord[0]
                                     for i,x in enumerate(block.coordinates)
                                     if 0 < get_distance(x, con_atom_coord) < 4])
                left_edge.append(furthest_coord)
        
            else:
                left_edge.append(+0.51)
        
        right_edge = []
        for block_name in self.pattern:
            block = self.default_coordinate_fragments[block_name]
            if len(block.connecting_atoms) >= 2:
                con_atom = block.connecting_atoms[1]
                con_atom_coord = block.coordinates[con_atom]
                furthest_coord = max([x[0] - con_atom_coord[0]
                                     for i,x in enumerate(block.coordinates)
                                     if 0 < get_distance(x, con_atom_coord) < 4])
                right_edge.append(furthest_coord)
        
            else:
                right_edge.append(-1.51)
        
        flat_cons = [re-le < 0.5 for re, le in zip(right_edge[:-1], left_edge[1:])]
        
        if self.is_cyclic:
            flat_cons.append(right_edge[-1] - left_edge[0] < 0.5)
        return flat_cons
    
    def _get_end_to_end_distance(self):
        distance=0
        for fragment in self.pattern:
            distance+= self.default_coordinate_fragments[fragment].distance 
        return distance
    
    def _get_radius_from_end_to_end_distance(self, distance):
        chain_length = self._get_end_to_end_distance()
        if distance > chain_length:
            print("Too large distance specified")
            return -1
        
        # start from a circle with where end-points are touching and then
        # increase radius until ends of the chain are sufficiently far apart
        radius = chain_length/(2*np.pi)
        a=0
        while a < distance:
            radius += 0.1
            a = 2 * radius * np.sin(np.pi-chain_length/(2*radius))
        return radius
        
    @staticmethod
    def project_to_cylinder(coordinates, radius):
        coordinates.coordinates += np.array([0,radius,0])
        for i,coord in enumerate(coordinates.coordinates):
            rotated_point = rotate_point(np.array([0,coord[1],coord[2]]),
                                         [0,0,1],
                                         coord[0]/radius)
            coordinates.coordinates[i] = rotated_point
        return coordinates



