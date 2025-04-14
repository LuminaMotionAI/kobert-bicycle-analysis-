import os
import sys

# Set Java environment variables
java_home = r"C:\Program Files\Java\jdk-21"
jvm_path = r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll"

# Set environment variables
os.environ['JAVA_HOME'] = java_home
os.environ['JPY_JVM'] = jvm_path  # For JPype to find the JVM

print("Java environment variables set:")
print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
print(f"JPY_JVM: {os.environ.get('JPY_JVM')}")

print("\nRunning Keyword Network Analysis...")

# Check if required packages are installed
required_packages = ["networkx", "seaborn"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing required package: {package}")
        from subprocess import call
        call([sys.executable, "-m", "pip", "install", package])

# Run the keyword network analysis script
exec(open("keyword_network_analysis.py", encoding="utf-8").read()) 