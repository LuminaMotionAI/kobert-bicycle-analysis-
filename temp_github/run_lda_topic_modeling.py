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

print("\nRunning LDA Topic Modeling...")

# Install required packages if needed
try:
    import gensim
    import pyLDAvis
except ImportError:
    print("Installing required packages for LDA topic modeling...")
    from subprocess import call
    packages = ["gensim", "pyldavis"]
    for package in packages:
        call([sys.executable, "-m", "pip", "install", package])
    print("Required packages installed.")

# Run the LDA topic modeling script
exec(open("lda_topic_modeling.py", encoding="utf-8").read()) 