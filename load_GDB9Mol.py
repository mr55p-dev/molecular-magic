from legacy.Database import GDB9molecule
from pathlib import Path
import pickle


if __name__ == "__main__":
    # Find the files
    base = Path("dft_test_files")
    files = list(base.glob("*a.out"))
    files = sorted(files, key=lambda x: x.stem)

    # Load GDB9
    mols = []
    for filepath in files:
        mols.append(GDB9molecule(gausfile=str(filepath), strings=False))

    with open("./GDB9.pkl", 'wb') as file:
        pickle.dump(mols, file)
