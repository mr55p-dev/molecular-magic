"""Original filename CleanDatabase_G_v4C

Script to filter out molecules based on a variety of rules set out
in `clean_database_config.py`
"""

import pickle
from clean_database_config import (
    Group_to_Use,
    output_basepath,
    LongBondLimit,
    molecule_list_path,
    unwanted_files,
)

from clean_database_function_defs import (
    SelectConverged,
    SelectFragments,
    FilterState,
    RemoveLongBonds,
    RemoveStrainedMolecules,
    RemoveMolAtoms,
    RemoveTetravalentNs,
    RemoveNonMinimas,
    RemoveCarbanions,
)


# -- Import data
with open(molecule_list_path, "rb") as read_file:
    GDB_data = pickle.load(read_file)

print("Read in " + str(len(GDB_data)) + " samples")


# -- Clean data: Use only the converged ground state
# print('Removing molecules with 9 or more atoms')
# GDB_data, Removed_GDB5 = Select8OrLess(GDB_data)
# print('Number of samples '+str(len(GDB_data)))

print("Removing geometries which has not terminated normally")
GDB_data = SelectConverged(GDB_data)
print("Number of samples " + str(len(GDB_data)))


print("Removing geometries which havs fragmented")
GDB_data = SelectFragments(GDB_data)
print("Number of samples " + str(len(GDB_data)))


if Group_to_Use == "all":
    print("Will not filter out any subgroups")
else:
    print("Filtering geometries which have label " + Group_to_Use)
    GDB_data = FilterState(GDB_data)
print("Number of samples " + str(len(GDB_data)))


print("Removing geometries with long bonds")
GDB_data, Removed_GDB = RemoveLongBonds(GDB_data)
print("Number of samples " + str(len(GDB_data)))


# print('Removing molecules with 8 heavy ring atoms with small angles')
# GDB_data, Removed_GDB2 = RemoveAcuteRings(GDB_data)
# print('Number of samples '+str(len(GDB_data)))
#


# print('Removing molecules with strained 3,5 fused rings')
# GDB_data, Removed_GDB3 = RemoveStrainedRings(GDB_data)
# print('Number of samples '+str(len(GDB_data)))
#

# print('Removing molecules with zero free energies')
# GDB_data, Removed_GDB6 = ZeroG(GDB_data)
# print('\n'+'Number of samples '+str(len(GDB_data)))
#

print("Removing molecules with large angles")
(
    GDB_data,
    Removed_GDB4,
    Removed_Filenames,
    Removed_Angles,
    Removed_Valences,
) = RemoveStrainedMolecules(GDB_data)
print("\n" + "Number of samples " + str(len(GDB_data)))


print("Removing geometries with Fluorines")
GDB_data = RemoveMolAtoms(GDB_data, "F")
print("Number of samples " + str(len(GDB_data)))


print("Removing molecules with tetravalent N atoms")
GDB_data = RemoveTetravalentNs(GDB_data)
print("Number of samples " + str(len(GDB_data)))


print("Removing molecules with imaginary frequencies")
GDB_data = RemoveNonMinimas(GDB_data)
print("Number of samples " + str(len(GDB_data)))


print("Removing molecules with carbanions")
GDB_data = RemoveCarbanions(GDB_data)
print("Number of samples " + str(len(GDB_data)))


# print('Removing molecules with heavyatom count less than '+str(HeavyAtomLimit))
# GDB_data = RemoveSmallMolecules(GDB_data, HeavyAtomLimit)
# print('Number of samples '+str(len(GDB_data)))
#

# print('Selecting hydrocarbons only')
# GDB_data = SelectHydrocarbons(GDB_data)
# print('Number of samples '+str(len(GDB_data)))
#


###############################################################################
# Compile Failed Filenames
###############################################################################

print("Writing removed filenames")

# -- Files that doesn't exist in the database

# removedlongbondsoutput = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_RemoveLongBonds'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ',filename)
#     else:
#         removedlongbondsoutput.write(filename+',1'+'\n')

# removedlongbondsoutput.close()

# removedacuteanglesoutput = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_RemoveAcuteAngles'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB2:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ',filename)
#     else:
#         removedacuteanglesoutput.write(filename+',1'+'\n')

# removedacuteanglesoutput.close()


# removed35ringstrainoutput = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_Remove35RingStrain'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB3:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ', filename)
#     else:
#         removed35ringstrainoutput.write(filename+',1'+'\n')


# removedstrainedmolouput = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_RemoveStrainedMol'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB4:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ', filename)
#     else:
#         removedstrainedmolouput.write(filename+',1'+'\n')

# removedstrainedmolouput.close()

# removedspcarbonoutput = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_RemoveSpCarbon'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB4:
#    filename = ExtractFilename(mol)
#
#    if filename in unwanted_files
#        print('Passed the following file: ', filename)
#    else:
#        removedspcarbonoutput.write(filename+',1'+'\n')

# removedmorethaneight = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_RemoveMoreThanEight'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB5:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ', filename)
#     else:
#         removedmorethaneight.write(filename+',1'+'\n')

# removedmorethaneight.close()

with open(
    output_basepath / "no_strained_angles.txt" "w",
) as removedstrainednamesandangles:
    for index in range(len(Removed_Filenames)):
        filename = Removed_Filenames[index]
        angles = Removed_Angles[index]
        valences = Removed_Valences[index]

        removedstrainednamesandangles.write(filename + "\n")

        for angle in angles:
            removedstrainednamesandangles.write(str(round(angle, 1)) + ", ")

        removedstrainednamesandangles.write("\n")

        for valence in valences:
            removedstrainednamesandangles.write(str(valence) + ", ")

        removedstrainednamesandangles.write("\n")

# removedzeroG = open(DirectoryName+'/'+DirectoryName+'_'+PickleFile[:-4]+'_ZeroG'+str(LongBondLimit)+'.txt', 'w')

# for mol in Removed_GDB5:
#     filename = ExtractFilename(mol)

#     if filename in unwanted_files
#         print('Passed the following file: ', filename)
#     else:
#         removedzeroG.write(filename+',1'+'\n')

# removedzeroG.close()


with open(output_basepath / "data" / "cleaned_data.pkl", "wb") as GDBDataFile:
    pickle.dump(GDB_data, GDBDataFile)

print("Done")
