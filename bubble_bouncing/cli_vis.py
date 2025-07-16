from pathlib import Path
import argparse
from bubble_bouncing import BubbleDataVisualizer
from .config import Config

test_folder = Config().test_folder

parser = argparse.ArgumentParser(f"Bubble visualizer.")
parser.add_argument("--folder", type=str, default=test_folder, help="Main data folder.")
parser.add_argument("--mode", type=str, default="w", help="Visualization mode: s, v or vh.")
parser.add_argument("--playback", type=float, default=0.01, help="Playback speed.")
args = parser.parse_args()
vis = BubbleDataVisualizer(args.folder)
mode = args.mode
playback = args.playback

def view_traj():
    vis.traj_com(mode=args.mode, playback=playback)

def view_morphology():
    vis.morphology(mode=args.mode, playback=playback)

def view_oseen():
    vis.Oseen_circulation(mode=args.mode, playback=playback)