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

# Now import KoNLPy
try:
    from konlpy.tag import Okt
    okt = Okt()
    print("\nKoNLPy initialized successfully!")
    
    # Test the tokenizer
    test_text = "안녕하세요. KoNLPy 테스트입니다."
    print(f"\nTest text: {test_text}")
    print(f"Nouns: {okt.nouns(test_text)}")
    print(f"Morphs: {okt.morphs(test_text)}")
    print(f"Pos: {okt.pos(test_text)}")
    
except Exception as e:
    print(f"\nError initializing KoNLPy: {e}")
    
print("\nNow you can run the keyword analysis or sentiment analysis scripts.") 