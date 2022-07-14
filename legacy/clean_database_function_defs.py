"""clean database function definitions"""
import pickle
import openbabel as ob
import sys

from pathlib import Path
from clean_database_config import (
    Group_to_Use,
    LongBondLimit,
    testanglelimit4,
    testanglelimit3A,
    testanglelimit3B,
    testanglelimit2,
    output_basepath,
    PTable
)


def SelectConverged(GDBclass):
    '''
    Returns GDB9molecule objects which has converged
    
    '''
    
    Converged = []

    
    for mol in GDBclass:
        if mol.converged == True:
            Converged += [mol]
        else:
            print(str(mol.geomfilename)+'\n')

    return Converged


def SelectFragments(GDBclass):
    '''
    Returns molecules which did not fragment and no change in the topology.
    
    Topology change is the InChI generated from the first geometry differs to 
    the InChI generated from the last geometry.
    
    '''
    
    NoFragments = []

    print(''+'\n')
    print('B. Files that have failed the fragment test \n')
    
    for mol in GDBclass:
        if mol.fragments == 1:
            NoFragments += [mol]
        else:
            print(str(mol.geomfilename)+'\n')
    
    return NoFragments


def SelectHydrocarbons(GDBclass):
    '''
    Returns molecules which has C and H atoms only
    
    '''
    
    Hydrocarbons = []
    
    for mol in GDBclass:
        if 'N' not in mol.atoms and 'O' not in mol.atoms:
            Hydrocarbons += [mol]
    
    return Hydrocarbons


def FilterState(GDBclass):
    '''
    Returns GDB9molecule objects which has converged to the ground state
    
    '''
    
    GroundState = []
    
    for mol in GDBclass:
        if Group_to_Use in mol.geomfilename.split('/')[-1]:
            GroundState += [mol]
    
    return GroundState


def RemoveLongBonds(GDBclass):
    '''
    Remove molecules with bonds longer than 1.6 Angstroms
    
    '''
    
    NoLongBonds = []
    YesLongBonds = []
    
    print(''+'\n')
    print('C. Files that have failed the long bonds test \n')
    
    for mol in GDBclass:
        molecule = BuildOBMol(mol.atoms, mol.coords)
        
        mol_bonds = []
        
        for OBmolbond in ob.OBMolBondIter(molecule):
            mol_bonds += [OBmolbond.GetLength()]
    
        if all(bond < LongBondLimit for bond in mol_bonds):
            NoLongBonds += [mol]
        else:
            YesLongBonds += [mol]
            print(str(mol.geomfilename)+'\n')
    
    return NoLongBonds, YesLongBonds


def RemoveMolAtoms(GDBclass, AtomToRemove):
    '''
    Remove molecules with specific elements
    
    '''
    
    NoHeavyAtoms = []
    
    print(''+'\n')
    print('D. Files containing Fluorine atoms \n')
    
    for mol in GDBclass:
        elements = mol.atoms
        
        if AtomToRemove not in elements:
            NoHeavyAtoms += [mol]
        else:
            print(str(mol.geomfilename)+'\n')
    
    return NoHeavyAtoms


def GetAtomSymbol(AtomNum):

    if AtomNum > 0 and AtomNum < len(PTable):
        return PTable[AtomNum-1]
    else:
        print("No such element with atomic number " + str(AtomNum))
        return 0


def GetAtomNum(AtomSymbol):

    if AtomSymbol in PTable:
        return PTable.index(AtomSymbol)+1
    else:
        print("No such element with symbol " + str(AtomSymbol))
        return 0


def BuildOBMol(atoms, coords):

    mol = ob.OBMol()
    for anum, acoords in zip(atoms, coords):
        atom = ob.OBAtom()
        atom.thisown = False
        atom.SetAtomicNum(GetAtomNum(anum))
        atom.SetVector(acoords[0], acoords[1], acoords[2])
        mol.AddAtom(atom)

    # Restore the bonds
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    # mol.Kekulize()

    return mol


def RemoveTetravalentNs(GDBclass):
    '''
    Remove molecules containing nitrogen atoms with four bonds
    
    '''
    
    NoTetraNs = []
    
    print(''+'\n')
    print('Remove molecules with tetravalent nitrogens \n')
    
    for mol in GDBclass:
        TetravalentN = False
        molecule = BuildOBMol(mol.atoms, mol.coords)   
        
        for OBmolatom in ob.OBMolAtomIter(molecule):
            valencedata = str(OBmolatom.GetTotalValence())
            atomdata = str(OBmolatom.GetType())
            
            if 'N' in atomdata and valencedata == '4':
                TetravalentN = True
    
        if TetravalentN == False:
            NoTetraNs += [mol]
    
    return NoTetraNs

        
def RemoveNonMinimas(GDBclass):
    '''
    Remove molecules with imaginary frequencies
    
    '''
    
    Minimas = []
    
    print(''+'\n')
    print('Remove molecules with imaginary frequencies \n')
    
    for mol in GDBclass:
        
        if hasattr(mol, 'imaginaryfreqs') and mol.imaginaryfreqs == False:
            Minimas += [mol]
        
    return Minimas


def ExtractFilename(GDBmolecule):
    '''
    Returns the output filename of GDB molecule
    
    '''
    outputfilename = GDBmolecule.geomfilename
    outputfilenamelist = outputfilename.split('/')
    
    return outputfilenamelist[-1]


def CheckListContainsOnlyList(ListToTest, ListToCompare):
    
    comparison = True
    
    for element in ListToTest:
        
        if element in ListToCompare:
            pass
        else:
            comparison = False
    
    return comparison
    

def RemoveCarbanions(GDBclass):
    '''
    Remove molecules if it contains a carbon atom with three neighbours and all
    neighbouring carbon atoms have four bonds
    
    '''
    
    NonCarbanions = []
    FailedFilenames = []
    carbanion_checklist = ['C4', 'H1', 'O2', 'N3']
    
    print(''+'\n')
    print('Remove carbanions/carbocations (carbon atoms with three neighbours and all neighbouring carbon atoms have four bonds). \n')
    
    for mol in GDBclass:
        carbanion = False
        molecule = BuildOBMol(mol.atoms, mol.coords)
        
        for OBmolatom in ob.OBMolAtomIter(molecule):
            valencedata = str(OBmolatom.GetTotalValence())
            atomdata = str(OBmolatom.GetType())


            if 'C' in atomdata and valencedata == '3':
                atom_types = []
                
                for neighbour_atom in ob.OBAtomAtomIter(OBmolatom):
                    neighbour_atomtype = str(neighbour_atom.GetType())
                    neighbour_valence = str(neighbour_atom.GetTotalValence())
                    atom_types += [neighbour_atomtype[0]+neighbour_valence]
                
                #print(atom_types)
                CheckC4H1Only = CheckListContainsOnlyList(atom_types, carbanion_checklist)
                #print('CheckC4H1Only: ',CheckC4H1Only)
                
                if CheckC4H1Only == True:
                    carbanion = True
                
        if carbanion == False:
            NonCarbanions += [mol]
        if carbanion == True:
            FailedFilename = ExtractFilename(mol)
            #print('Molecule failed the carbanion test: '+str(FailedFilename))
            FailedFilenames += [FailedFilename]

    fail_output_path = output_basepath / 'meta_results'
    fail_output_path.mkdir(parents=True, exist_ok=True)

    with open(fail_output_path / '_failedcarbanion.txt', 'w') as failoutputfile:
        for fail_molname in FailedFilenames:
            failoutputfile.write(fail_molname+',1'+'\n')

    return NonCarbanions
                   

def RemoveSmallMolecules(GDBclass, heavyatomlimit):
    
    HeavyMols = []
    FailedFilenames = []
    
    for mol in GDBclass:
        
        HeavyAtomCount = 0
        molecule = BuildOBMol(mol.atoms, mol.coords)
        
        for OBmolatom in ob.OBMolAtomIter(molecule):
            atomdata = str(OBmolatom.GetType())
            
            if 'C' in atomdata or 'N' in atomdata or 'O' in atomdata:
                HeavyAtomCount += 1
            
        if HeavyAtomCount >= heavyatomlimit:
            HeavyMols += [mol]
        else:
            FailedFilename = ExtractFilename(mol)
            FailedFilenames += [FailedFilename]

    fail_output_path = output_basepath / 'meta_results'
    fail_output_path.mkdir(parents=True, exist_ok=True)

    with open(fail_output_path / '_failedsmallmolecules.txt', 'w') as failoutputfile:
        for fail_molname in FailedFilenames:
            failoutputfile.write(fail_molname+',1'+'\n')

    return HeavyMols


def RemoveAcuteRings(GDBclass):
    '''
    Remove 8 membered rings with acute / CC angle less than 109.5
    
    '''
    
    NoAcuteRings = []
    FailedFiles = []
    
    for mol in GDBclass:
        
        OBmol = BuildOBMol(mol.atoms, mol.coords)
        
        OBrings = OBmol.GetSSSR()

        AcuteAngleTest = False

        for OBring in OBrings:
            ringsize = OBring.Size()
            
            if ringsize == 8: # check if 8 membered ring exists in the molecule
                

                
                for OBmolangle in ob.OBMolAngleIter(OBmol):
                    
                    angletype = [GetAtomSymbol(OBmol.GetAtom(OBmolangle[0]+1).GetAtomicNum()), \
                                 GetAtomSymbol(OBmol.GetAtom(OBmolangle[1]+1).GetAtomicNum()), \
                                 GetAtomSymbol(OBmol.GetAtom(OBmolangle[2]+1).GetAtomicNum())] 
                    
                    if str(angletype[0]) == 'C' and str(angletype[1]) == 'C' and str(angletype[2]) == 'C':
                        
                        NonHAngle = OBmol.GetAngle(OBmol.GetAtom(OBmolangle[1]+1),\
                                                   OBmol.GetAtom(OBmolangle[0]+1),\
                                                   OBmol.GetAtom(OBmolangle[2]+1))
                        
                        if NonHAngle < 109.5:
                            AcuteAngleTest = True

        if AcuteAngleTest == False:
            NoAcuteRings += [mol]
        else:
            FailedFiles += [mol]
            
    
    return NoAcuteRings, FailedFiles



def RemoveStrainedRings(GDBclass):
    '''
    '''

    NoStrainedRings = []
    StrainedRings = []

    for mol in GDBclass:
        
        OBmol = BuildOBMol(mol.atoms, mol.coords)
        
        OBrings = OBmol.GetSSSR()
        
        m5_rings = []
        m3_rings = []
        
        for OBring in OBrings:
            ringsize = OBring.Size()
            
            if ringsize == 5:
                
                ring_5m = []
                
                for OBatom in ob.OBMolAtomIter(OBmol):
                    
                    if OBring.IsMember(OBatom) == True:
                        ring_5m += [OBatom.GetIndex()]
                
                m5_rings += [ring_5m]
            
            
            if ringsize == 3:
                
                ring_3m = []
                
                for OBatom in ob.OBMolAtomIter(OBmol):
                    
                    if OBring.IsMember(OBatom) == True:
                        ring_3m += [OBatom.GetIndex()]
                
                m3_rings += [ring_3m]
        
        
        ringindex = 0
        fused35 = False
        fused_atoms = []
        
        while ringindex < len(m5_rings):
        
            for m3_ring in m3_rings:
                
                #print(m5_rings[ringindex])
                #print([m3_ring[0],m3_ring[1]])
                #print([m3_ring[1],m3_ring[2]])
                #print([m3_ring[0],m3_ring[2]])
                
                # combination 1
                if set([m3_ring[0],m3_ring[1]]).issubset(set(m5_rings[ringindex])):
                    fused35 = True
                    fused_atoms += [[m3_ring[0],m3_ring[1]]]
        
                # combination 2
                if set([m3_ring[1],m3_ring[2]]).issubset(set(m5_rings[ringindex])):
                    fused35 = True
                    fused_atoms += [[m3_ring[1],m3_ring[2]]]
                    
                # combination 3
                if set([m3_ring[0],m3_ring[2]]).issubset(set(m5_rings[ringindex])):
                    fused35 = True
                    fused_atoms += [[m3_ring[0],m3_ring[2]]]
            
            ringindex += 1
        
        #print(fused_atoms)
        
        #if len(fused_atoms) > 1:
        #    print('This molecule has multiple fused atoms: ')
        #    print(fused_atoms)
        
        
        strained35 = False
        
        if fused35 == True:
            
            for fused_combo in fused_atoms:

                atom1angles = []
                atom2angles = []
                
                #print(fused_combo)
                atom1type = OBmol.GetAtom(fused_combo[0]+1).GetType()
                atom2type = OBmol.GetAtom(fused_combo[1]+1).GetType()
                
                #print(atom1type)
                #print(atom2type)
                
                if atom1type[0] == 'C' and atom2type[0] =='C':
                    
                    for OBangle in ob.OBMolAngleIter(OBmol):
                        
                        # get the index of the 3 atoms making the angle
                        angletype = [OBmol.GetAtom(OBangle[0]+1).GetIndex(), \
                                     OBmol.GetAtom(OBangle[1]+1).GetIndex(), \
                                     OBmol.GetAtom(OBangle[2]+1).GetIndex()]  # vertex first!!!
                        
                        #print(angletype)
                        
                        if angletype[0] == fused_combo[0]:
                            atom1angles += [OBmol.GetAngle(OBmol.GetAtom(OBangle[1]+1), OBmol.GetAtom(OBangle[0]+1), OBmol.GetAtom(OBangle[2]+1))] # vertex middle!!!
                            
                        if angletype[0] == fused_combo[1]:
                            atom2angles += [OBmol.GetAngle(OBmol.GetAtom(OBangle[1]+1), OBmol.GetAtom(OBangle[0]+1), OBmol.GetAtom(OBangle[2]+1))] # vertex middle!!!
                    
                    #print(atom1angles)
                    #print(atom2angles)
                    
                    # -- IMPORTANT --
                    # This section checks whether the 3,5 fused rings have large strain or not
                    # ---------------
                    
                    # Criterion 1 - if the 3,5 fused atom only has 3 angles, double bond must be present across the fused ring
                    # therefore, likely to be extremely strained
                    
                    if len(atom1angles) == 3 or len(atom2angles) == 3:
                        strained35 = True
                    
                    # Criterion 2 - if the 3,5 fused atom has 6 angles, check the degree of strain by measuring how far the angle
                    # is from ideal 109.5 degrees
                    
                    else:
                        atom1strains = [abs(atom1angle - 109.5) for atom1angle in atom1angles]
                        atom2strains = [abs(atom2angle - 109.5) for atom2angle in atom2angles]
                        
                        atom1strainangles = [atom1strain for atom1strain in atom1strains if atom1strain > 30]
                        atom2strainangles = [atom2strain for atom2strain in atom2strains if atom2strain > 30]
                        
                        if len(atom1strainangles) >= 2 or len(atom2strainangles) >= 2:
                            strained35 = True
                    
                    #if strained35 == True:
                    #    print(strained35)
                    #    print(atom1strains)
                    #    print(atom2strains)
    
                    # ---------------
        
        #print(strained35)
        
        if strained35 == True:
            StrainedRings += [mol]
        else:
            NoStrainedRings += [mol]
            
    
    return NoStrainedRings, StrainedRings



def RemoveStrainedMolecules(GDBclass):
    
    '''
    Removes molecules with large angles
    
    '''
    
    NotStrainedMol = []
    StrainedMol = []
    printfilenames = []
    printangles = []
    printvalence = []
    
    for mol in GDBclass:
        
        OBmol = BuildOBMol(mol.atoms, mol.coords)
        strainedmol = False
        filename = ExtractFilename(mol)
        molangles = []
        molvalence = []
        atomanglesum = dict()

        
        for OBangle in ob.OBMolAngleIter(OBmol):
            
            central_val = OBmol.GetAtom(OBangle[0]+1).GetTotalValence()
            central_type = OBmol.GetAtom(OBangle[0]+1).GetType()
            central_idx = OBmol.GetAtom(OBangle[0]+1).GetIdx()
            atomangle = OBmol.GetAngle(OBmol.GetAtom(OBangle[1]+1), OBmol.GetAtom(OBangle[0]+1), OBmol.GetAtom(OBangle[2]+1))
            
            molangles += [atomangle]
            molvalence += [central_val]
            
            testangle = 0
            
            if central_val == 4:
                testangle = atomangle - 109.5
                
            if central_val == 3 and central_type[0] == 'C':
                
                if central_type[0] + str(central_idx) in atomanglesum.keys():
                    atomanglesum[central_type[0] + str(central_idx)].append(atomangle)
                else:
                    atomanglesum[central_type[0] + str(central_idx)] = [atomangle]
            
            if central_val == 2:
                testangle = 180 - atomangle
                
            if testangle > testanglelimit4 and central_type[0] == 'C' and central_val == 4:
                strainedmol = True

            if testangle > testanglelimit2 and central_type[0] == 'C' and central_val == 2:
                strainedmol = True
            
        #print(atomanglesum)
        
        for key, value in atomanglesum.items():
            testangle = 360.0 - sum(value)
            
            if testangle > testanglelimit3A:
                strainedmol = True
                
            if all((angle - 120.0) < testanglelimit3B for angle in value):
                pass
            else:
                strainedmol = True
        
        
        if strainedmol == False:
            NotStrainedMol += [mol]
        else:
            StrainedMol += [mol]
            printfilenames += [filename]
            printangles += [molangles]
            printvalence += [molvalence]
            
        sys.stdout.write('\r')
        sys.stdout.write(str(round(100.0*float(GDBclass.index(mol))/float(len(GDBclass)), 1))+'%')
        sys.stdout.flush()
    
    return NotStrainedMol, StrainedMol, printfilenames, printangles, printvalence
    

def Select8OrLess(GDBclass):
    
    '''
    Select molecules with 8 or less atoms only
    
    '''
    
    EightOrLess = []
    MoreThanEight = []
    HeavyAtoms = ['C', 'O', 'N']
    
    for mol in GDBclass:
        
        atoms = mol.atoms
        heavy_atoms = []
        
        for atom in atoms:
            
            if atom in HeavyAtoms:
                heavy_atoms += [atom]
        
        heavyatom_no = len(heavy_atoms)
        
        if heavyatom_no > 8:
            MoreThanEight += [mol]
        else:
            EightOrLess += [mol]
    
    return EightOrLess, MoreThanEight


def ZeroG(GDBclass):
    '''
    Select molecules with non-zero free energies
    '''
    
    NonZeroG = []
    ZeroG = []
    
    for mol in GDBclass:
        
        if mol.G298 == 0.0:
            ZeroG += [mol]
        else:
            NonZeroG += [mol]
    
    return NonZeroG, ZeroG