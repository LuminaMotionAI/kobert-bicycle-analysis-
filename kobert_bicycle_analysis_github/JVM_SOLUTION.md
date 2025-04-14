# KoNLPy JVM Issue - Solution

## Problem

KoNLPy requires a Java Virtual Machine (JVM) to run because it uses JPype to interface with Java libraries. The error was:

```
Error initializing KoNLPy: Java VM cannot be loaded. Required for Mecab, Komoran, Hannanum, and Okt.
```

## Root Cause

Even though Java was properly installed on the system (Java 21.0.6), the environment variables were not set correctly for KoNLPy to find and use the JVM.

## Solution

We created wrapper scripts that automatically set the appropriate environment variables before running the KoNLPy-dependent code:

1. Set the `JAVA_HOME` environment variable to point to the JDK installation directory:
   ```python
   os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jdk-21"
   ```

2. Set the `JPY_JVM` environment variable to explicitly point to the JVM DLL file:
   ```python
   os.environ['JPY_JVM'] = r"C:\Program Files\Java\jdk-21\bin\server\jvm.dll"
   ```

3. Created wrapper scripts for all the main functionalities:
   - `run_konlpy.py`: Tests the KoNLPy and JVM setup
   - `run_konlpy_keywords.py`: Runs the keyword analysis functionality
   - `run_kobert_sentiment.py`: Runs sentiment analysis
   - `run_text_mining.py`: Runs the comprehensive text mining script

4. Updated the README with instructions for running the scripts properly

## Important Notes

1. These environment variables only affect the Python process where they are set, not the entire system. 

2. For a permanent solution, you could set these environment variables at the system level:
   - Windows: Control Panel > System > Advanced System Settings > Environment Variables
   - Add `JAVA_HOME` with the value `C:\Program Files\Java\jdk-21`
   - Add `JPY_JVM` with the value `C:\Program Files\Java\jdk-21\bin\server\jvm.dll`

3. If you update or change your Java installation, you'll need to update these paths in the wrapper scripts.

## Verification

The solution was tested and confirmed to work by successfully running:
```
python run_konlpy.py
python run_konlpy_keywords.py
```

The keyword extraction script correctly processed the review data and generated a word cloud image at `output/konlpy_keywords.png`. 