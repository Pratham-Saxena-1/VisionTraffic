import pytest
import os
import sys

def main():
    print("==================================================")
    print(" VisionTraffic Evaluation Script")
    print("==================================================")
    
    print("\n[*] Running pytest suite for classical CV validation (DLT & RANSAC)...")
    exit_code = pytest.main(["-v", "tests/"])
    
    if exit_code == 0:
        print("\n[+] All unit tests passed! Geometric algorithms are correct.")
    else:
        print("\n[-] Unit tests failed. Please check the implementation.")
        
    print("\n[*] To evaluate full pipeline performance on video, run:")
    print("    python main.py --input data/sample/traffic.mp4 --mode full")
    print("\n[*] To generate the academic report mapping theory to outputs, run:")
    print("    python generate_report.py")

if __name__ == '__main__':
    main()
