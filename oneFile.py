import argparse 

if __name__ == "__main__":
    args = argparse.ArgumentParser(prog='Compacting one file', description='Converting \'h.txt\' into a csv file')
    args.add_argument("folder", help = "folder to be converted into a csv file")
