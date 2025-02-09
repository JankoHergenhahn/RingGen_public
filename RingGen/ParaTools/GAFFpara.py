import importlib

class GAFFpara:
    """
    Container for GAFF force field parameters
    """
    def __init__(self):
        self.bonds = self._read_bond_prameters()
        self.angles = self._read_angles_prameters()
        self.dihedrals = self._read_dihedrals_prameters()
        self.impropers = self._read_impropers_prameters()
        
    def _read_bond_prameters(self):
        bonds =  {}
        with importlib.resources.open_text('RingGen.ParaTools.GAFF','gaff_bonds.dat') as f:
            for line in f:
                atom_1 = line[0:2].strip(" ")
                atom_2 = line[3:5].strip(" ")
                k_f = float(line[6:].split()[0])
                r_e = float(line[6:].split()[1])
                bonds.update({f"{atom_1}-{atom_2}": {"k_f": k_f, "r_e": r_e}})
                bonds.update({f"{atom_2}-{atom_1}": {"k_f": k_f, "r_e": r_e}})
        return bonds
    
    def _read_angles_prameters(self):
        angles =  {}
        with importlib.resources.open_text('RingGen.ParaTools.GAFF','gaff_angles.dat') as f:
            for line in f:
                atom_1 = line[0:2].strip(" ")
                atom_2 = line[3:5].strip(" ")
                atom_3 = line[6:8].strip(" ")
                k_f = float(line[9:].split()[0])
                th_e = float(line[9:].split()[1])
                angles.update({f"{atom_1}-{atom_2}-{atom_3}": 
                              {"k_f": k_f, "th_e": th_e}})
                if atom_1 != atom_3:
                    angles.update({f"{atom_3}-{atom_2}-{atom_1}": 
                                  {"k_f": k_f, "th_e": th_e}})
        return angles
    
    def _read_dihedrals_prameters(self):
        dihedrals =  {}
        with importlib.resources.open_text('RingGen.ParaTools.GAFF','gaff_dihedrals.dat') as f:
            for line in f:
                atom_1 = line[0:2].strip(" ")
                atom_2 = line[3:5].strip(" ")
                atom_3 = line[6:8].strip(" ")
                atom_4 = line[9:11].strip(" ")
                k_f = float(line[12:].split()[1])
                th_e = float(line[12:].split()[2])
                mul = int(float(line[12:].split()[3]))
                if f"{atom_1}-{atom_2}-{atom_3}-{atom_4}" not in dihedrals:
                    dihedrals.update({f"{atom_1}-{atom_2}-{atom_3}-{atom_4}":
                                      [{"k_f": k_f, "th_e": th_e, "mul": mul}]})
                    dihedrals.update({f"{atom_4}-{atom_3}-{atom_2}-{atom_1}":
                                      [{"k_f": k_f, "th_e": th_e, "mul": mul}]})
                #some dihedral angles are described by multiple terms, e.g. esters
                else:
                    previous_parameter = dihedrals[f"{atom_1}-{atom_2}-{atom_3}-{atom_4}"]
                    dihedrals.update({f"{atom_1}-{atom_2}-{atom_3}-{atom_4}":
                                      previous_parameter + [{"k_f": k_f, "th_e": th_e, "mul": mul}]})
                    
                    previous_parameter = dihedrals[f"{atom_4}-{atom_3}-{atom_2}-{atom_1}"]
                    dihedrals.update({f"{atom_4}-{atom_3}-{atom_2}-{atom_1}":
                                      previous_parameter + [{"k_f": k_f, "th_e": th_e, "mul": mul}]})
                    
        return dihedrals
    
    def _read_impropers_prameters(self):
        impropers =  {}
        with importlib.resources.open_text('RingGen.ParaTools.GAFF','gaff_impropers.dat') as f:
            for line in f:
                atom_1 = line[0:2].strip(" ")
                atom_2 = line[3:5].strip(" ")
                atom_3 = line[6:8].strip(" ")
                atom_4 = line[9:11].strip(" ")
                k_f = float(line[12:].split()[0])
                th_e = float(line[12:].split()[1])
                mul = int(float(line[12:].split()[2]))
                impropers.update({f"{atom_1}-{atom_2}-{atom_3}-{atom_4}":
                                  {"k_f": k_f, "th_e": th_e, "mul": mul}})
        return impropers