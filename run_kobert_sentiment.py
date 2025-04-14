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

print("\nRunning the sentiment analysis...")

# Determine which script to run
script_choice = input("Which sentiment analysis would you like to run?\n1. KoBERT Sentiment Analysis (kobert_sentiment_analysis.py)\n2. NSMC Sentiment Analysis (kobert_sentiment_nsmc.py)\nEnter 1 or 2: ")

if script_choice == "1":
    print("\nRunning kobert_sentiment_analysis.py...")
    exec(open("kobert_sentiment_analysis.py", encoding="utf-8").read())
elif script_choice == "2":
    print("\nRunning kobert_sentiment_nsmc.py...")
    exec(open("kobert_sentiment_nsmc.py", encoding="utf-8").read())
else:
    print("Invalid choice. Please enter 1 or 2.")
    sys.exit(1) 