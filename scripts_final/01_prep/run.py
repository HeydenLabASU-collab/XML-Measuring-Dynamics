from XMLEvoDynam.MLInputTools import *
import argparse

parser = argparse.ArgumentParser(description="Process input file and cutoff value.")

# Add arguments
parser.add_argument('--system', type=str, required=True, help='Path to input file')
parser.add_argument('--replica', type=int, required=True, help='Path to input file')
parser.add_argument('--label', type=int, required=True, help='Threshold cutoff value')

# Parse arguments
args = parser.parse_args()

# Access arguments
print(f"System Name: {args.system}")
print(f"Replica #: {args.replica}")
print(f"Output Label: {args.label}")

systemname = args.system
replica = args.replica
label = args.label

reference = f"/scratch/masauer2/Regeneron_USample_Compare/{args.system}/00_system_prep/prot.gro"
trajectory = f"/scratch/masauer2/Regeneron_USample_Compare/{args.system}/03_100ns/sample-NPT_pbc.trr"

pace = 1
GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)
MLInputTools.construct_XML_input(reference, trajectory, False, pace, GROUP1, GROUP2, label, f"{systemname}_R{replica}", nBatches = 10)
