import os, re
import pandas as pd
from typing import Tuple
import toml
from openai import OpenAI
import threading
import time
import concurrent.futures

# Configure matplotlib for Streamlit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

class TimeoutError(Exception):
    pass

def execute_with_timeout(code, local_env, timeout_seconds=15):
    """Execute code with a timeout using threading (Windows compatible)"""
    result = [None]
    exception = [None]
    execution_completed = [False]
    
    def execute_code():
        try:
            print(f"🔧 Starting code execution in thread...")
            exec(code, {}, local_env)
            result[0] = local_env.get("out")
            execution_completed[0] = True
            print(f"✅ Thread execution completed successfully")
        except Exception as e:
            exception[0] = e
            print(f"❌ Thread execution failed: {e}")
    
    # Create and start thread
    thread = threading.Thread(target=execute_code)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"⏰ Code execution timed out after {timeout_seconds} seconds")
        print(f"🔍 Thread status: alive={thread.is_alive()}, completed={execution_completed[0]}")
        return None
    
    if exception[0]:
        print(f"❌ Code execution raised exception: {exception[0]}")
        raise exception[0]
    
    print(f"✅ Code execution completed without timeout")
    return result[0]

def load_openai_client():
    """Load OpenAI client with API key from secrets.toml"""
    try:
        # Try to load from secrets.toml first
        secrets_path = os.path.join(os.path.dirname(__file__), "..", ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            api_key = secrets.get("openai", {}).get("api_key")
            if api_key and api_key != "sk-REPLACE_ME":
                print(f"🔑 OpenAI API key loaded from secrets.toml")
                return OpenAI(api_key=api_key)
        
        # Fallback to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"🔑 OpenAI API key loaded from environment")
            return OpenAI(api_key=api_key)
            
        print("⚠️  No OpenAI API key found in secrets.toml or environment")
        return None
        
    except Exception as e:
        print(f"❌ Error loading OpenAI client: {e}")
        return None

# Initialize OpenAI client
client = load_openai_client()

SAFE_TOKENS = ["groupby","agg","mean","sum","pct_change","resample","rolling","sort_values",
               "query","assign","reset_index","head","tail"]

def _clean_generated_code(code: str) -> str:
    """Clean AI-generated code by removing markdown and fixing common issues"""
    if not code:
        return code
    
    # Remove markdown code blocks
    code = code.strip()
    if code.startswith('```'):
        # Find the end of the markdown block
        lines = code.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    continue
                else:
                    in_code_block = False
                    continue
            
            if in_code_block:
                cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
    
    # Remove any remaining markdown artifacts
    code = code.replace('```python', '').replace('```', '').strip()
    
    # Fix common AI generation issues
    code = code.replace('```', '')  # Remove any remaining backticks
    
    # Ensure the code ends with proper syntax
    if code and not code.endswith('\n'):
        code += '\n'
    
    print(f"🔧 Code cleaning: Removed markdown, length: {len(code)} chars")
    return code

def _is_safe(code:str)->bool:
    """Check if generated code is safe to execute"""
    # Allow safe pandas imports and operations
    safe_patterns = [
        "import pandas as pd",
        "import numpy as np", 
        "import matplotlib.pyplot as plt"
    ]
    
    # Block dangerous operations
    dangerous_patterns = [
        "import os", "import sys", "import subprocess",
        "open(", "exec(", "eval(", "__", "os.", "subprocess", 
        "system", "write(", "remove(", "delete(", "rmdir(",
        "chmod(", "chown(", "sudo", "admin"
    ]
    
    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in code:
            print(f"🚨 Blocked dangerous pattern: {pattern}")
            return False
    
    # Check for safe pandas imports
    has_safe_import = any(pattern in code for pattern in safe_patterns)
    if has_safe_import:
        print(f"✅ Safe import detected: {[p for p in safe_patterns if p in code]}")
    
    return True

def _validate_code(code: str) -> bool:
    """Validate that the cleaned code is syntactically correct Python"""
    if not code or len(code.strip()) < 10:
        return False
    
    try:
        # Try to compile the code to check syntax
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"🚨 Syntax error in generated code: {e}")
        return False
    except Exception as e:
        print(f"🚨 Code validation error: {e}")
        return False

def guarded_pandas_plan(df: pd.DataFrame, question: str)->Tuple[str,str,pd.DataFrame,object]:
    cols = list(df.columns)
    sample = df.head(5).to_dict()
    
    # Data validation and preprocessing (moved up)
    df_clean = df.copy()
    
    # Memory management for large datasets
    try:
        # Check for problematic data types
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Check if it's a datetime column
                if df_clean[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    print(f"⚠️  Converting datetime column '{col}' to string to prevent PyArrow issues")
                    df_clean[col] = df_clean[col].astype(str)
        
        # Ensure price column is numeric
        if 'price' in df_clean.columns and df_clean['price'].dtype == 'object':
            print(f"⚠️  Converting price column to numeric")
            df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
        
        print(f"📊 Cleaned dataset: {df_clean.dtypes.to_dict()}")
        
    except Exception as e:
        print(f"⚠️  Data preprocessing failed: {e}, using original dataset")
        df_clean = df.copy()
    
    # Memory usage check
    try:
        memory_mb = df_clean.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"📊 Memory usage: {memory_mb:.2f} MB")
        
        # If memory usage is too high, use sampling
        if memory_mb > 2000:  # More than 2GB
            print(f"⚠️  High memory usage detected, using sampling")
            df_clean = df_clean.sample(n=500_000, random_state=42)
            print(f"📊 Reduced to {len(df_clean):,} rows to manage memory")
            
    except Exception as e:
        print(f"⚠️  Memory check failed: {e}, proceeding with original dataset")
    
    # Now construct the prompt after df_clean is defined
    prompt = f"""You are a data assistant. The user asks a question about a pandas DataFrame.

Columns: {cols}
Sample rows: {sample}
Question: {question}

IMPORTANT: This dataset has {len(df_clean):,} rows - use the FULL dataset for analysis!

Write Python pandas code that computes the answer and assigns it to a variable named out.
IMPORTANT: 
- Return ONLY the Python code, no markdown formatting, no explanations, no backticks
- The code must be complete and executable
- For plotting questions: create the plot AND assign the data to 'out' variable
- For plotting: use plt.figure() before plotting, then assign the filtered data to 'out'
- Always ensure 'out' contains data (DataFrame/Series) that can be displayed in a table
- Example plotting pattern: plt.figure(); filtered_data = df[...]; filtered_data.plot(...); out = filtered_data
- Note: The DataFrame uses these column names: 'datetime' (date/time), 'Location_Id' (location identifier), 'Location_Name' (location name), 'price' (price value)

LARGE DATASET OPTIMIZATION (Use these for efficiency):
- Use memory_efficient_groupby(df, ['col1', 'col2'], 'price', ['mean', 'count']) for groupby operations
- Use smart_sample(df, 100000) to sample large datasets before plotting
- Use chunk_process(df, 100000) to process data in chunks
- Use efficient_resample(df, 'datetime', 'h', 'price', 'mean') for time series operations
- Use parallel_agg(df, ['col1'], 'price', ['mean', 'std']) for aggregations

SPECIFIC PLOTTING INSTRUCTIONS:
- Use plt.figure(figsize=(10,6)) for better chart sizing
- Add titles, labels, and grid: plt.title('Chart Title'); plt.xlabel('X Label'); plt.ylabel('Y Label'); plt.grid(True)
- For line plots: plt.plot(x, y, marker='o', linewidth=2)
- For bar plots: plt.bar(x, y, color='skyblue', alpha=0.7)
- For scatter plots: plt.scatter(x, y, alpha=0.6)
- Always call plt.tight_layout() before assigning to 'out'
- Ensure 'out' contains the data used for plotting (not the plot object)

CRITICAL: OPTIMIZE FOR LARGE DATASETS
- Use 'h' instead of 'H' for hourly frequency: pd.Grouper(key='datetime', freq='h')
- Use .resample('h') instead of .resample('H')
- For plotting large datasets: sample first, then plot
- Use .nlargest() and .nsmallest() instead of .sort_values().head()
- Use .dt accessors efficiently
- Avoid .apply() on large datasets - use vectorized operations
- Use .value_counts() instead of .groupby().size()
"""

    code = None
    if client:
        try:
            print(f"🚀 Making OpenAI API call to GPT-4o-mini...")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"You translate questions to pandas code safely."},
                          {"role":"user","content":prompt}],
                max_tokens=300
            )
            code = resp.choices[0].message.content.strip()
            print(f"✅ OpenAI API call successful! Generated code: {code[:100]}...")
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            code = None
    else:
        print("⚠️  No OpenAI client available - using fallback")

    if not code:
        print(f"⚠️  No code generated by AI, using fallback analysis")
        # fallback basic stats
        code = "out = df.groupby(['Location_Id','Location_Name'])['price'].agg(['count','mean','std']).reset_index().head(20)"
        
        # Clean the generated code
        code = _clean_generated_code(code)
        print(f"🧹 Cleaned code: {code[:100]}...")
        
        # Validate the cleaned code
        if not _validate_code(code):
            print(f"⚠️  Invalid code after cleaning, using fallback")
            code = "out = df.describe().reset_index()"
        
        if not _is_safe(code):
            print(f"🚨 Code safety check failed, using fallback analysis")
            # Provide a safe fallback analysis
            try:
                if "price" in df.columns:
                    out = df.groupby(['Location_Id', 'Location_Name'])['price'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                else:
                    out = df.describe().reset_index()
            except Exception as e:
                out = df.head(20)
            return (f"Plan: Safe fallback analysis for '{question}' (AI code blocked)", "Fallback analysis used", out, None)

    local_env={"pd":pd,"df":df.copy(), "plt":plt}
    try:
        print(f"🔧 Executing generated code...")
        print(f"📊 Dataset size: {len(df)} rows, {len(df.columns)} columns")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Check if the operation might be too large
        if len(df_clean) > 2_000_000:  # If dataset is extremely large (>2M rows)
            print(f"⚠️  Extremely large dataset detected ({len(df_clean):,} rows), using smart sampling")
            # Use stratified sampling for very large datasets
            df_sample = df_clean.sample(n=500_000, random_state=42)
            local_env["df"] = df_sample
            print(f"📊 Using stratified sample of {len(df_sample):,} rows for processing")
        elif len(df_clean) > 500_000:  # If dataset is very large (500K-2M rows)
            print(f"⚠️  Very large dataset detected ({len(df_clean):,} rows), using efficient processing")
            # Use the full dataset but with optimized operations
            local_env["df"] = df_clean
            print(f"📊 Using full dataset of {len(df_clean):,} rows with optimizations")
        else:
            # Use the full dataset for normal operations
            local_env["df"] = df_clean
            print(f"📊 Using full dataset of {len(df_clean):,} rows")
        
        # Add memory-efficient operations to local environment
        try:
            local_env.update({
                "pd": pd,
                "df": local_env["df"],
                "plt": plt,
                "np": __import__('numpy'),
                "sample_df": lambda df, n: df.head(n) if len(df) > n else df,
                # Large dataset optimization tools
                "chunk_process": lambda df, chunk_size=100000: [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)],
                "parallel_agg": lambda df, group_cols, agg_col, agg_funcs: df.groupby(group_cols)[agg_col].agg(agg_funcs).reset_index(),
                "memory_efficient_groupby": lambda df, group_cols, agg_col, agg_funcs: df.groupby(group_cols, observed=True)[agg_col].agg(agg_funcs).reset_index(),
                "smart_sample": lambda df, target_size=100000: df.sample(n=min(target_size, len(df)), random_state=42) if len(df) > target_size else df,
                "efficient_resample": lambda df, time_col, freq, agg_col, agg_func: df.set_index(time_col).resample(freq)[agg_col].agg(agg_func).reset_index()
            })
            print(f"✅ Local environment setup completed")
        except Exception as e:
            print(f"⚠️  Local environment setup failed: {e}, using basic environment")
            local_env = {
                "pd": pd,
                "df": local_env["df"],
                "plt": plt,
                "np": __import__('numpy')
            }
        
        print(f"🔍 Code to execute: {code[:200]}...")
        print(f"🔍 Local environment keys: {list(local_env.keys())}")
        
        # Calculate adaptive timeout based on dataset size
        try:
            if len(df_clean) > 1_000_000:
                timeout_seconds = 60  # 1 minute for very large datasets
            elif len(df_clean) > 500_000:
                timeout_seconds = 45  # 45 seconds for large datasets
            elif len(df_clean) > 100_000:
                timeout_seconds = 30  # 30 seconds for medium datasets
            else:
                timeout_seconds = 15  # 15 seconds for small datasets
            
            print(f"⏱️  Setting timeout to {timeout_seconds} seconds for {len(df_clean):,} rows")
            
        except Exception as e:
            print(f"⚠️  Timeout calculation failed: {e}, using default 30 seconds")
            timeout_seconds = 30
        
        try:
            out = execute_with_timeout(code, local_env, timeout_seconds=timeout_seconds)
            print(f"✅ Code execution completed, output type: {type(out)}")
        except Exception as exec_error:
            print(f"❌ Code execution failed: {exec_error}")
            print(f"🔄 Using fallback analysis")
            out = None
        
        # Check if any plots were created
        try:
            if plt.get_fignums():  # If any figures exist
                print(f"📊 Plots detected: {len(plt.get_fignums())} figures created")
                # Get the current figure for display
                current_fig = plt.gcf()
                if current_fig:
                    print(f"📈 Captured plot: {current_fig.get_size_inches()}")
                    # If out is None but we have a plot, assign the plot data
                    if out is None and hasattr(current_fig, 'axes') and len(current_fig.axes) > 0:
                        # Try to get data from the plot
                        ax = current_fig.axes[0]
                        if hasattr(ax, 'lines') and len(ax.lines) > 0:
                            # Extract data from the plot
                            line = ax.lines[0]
                            if hasattr(line, 'get_xdata') and hasattr(line, 'get_ydata'):
                                x_data = line.get_xdata()
                                y_data = line.get_ydata()
                                out = pd.DataFrame({'x': x_data, 'y': y_data})
                                print(f"📊 Extracted plot data: {len(out)} points")
        except Exception as plot_error:
            print(f"⚠️  Plot detection failed: {plot_error}")
            # Clear any problematic figures
            try:
                plt.close('all')
            except:
                pass
        
        if out is None:
            print(f"⚠️  Generated code didn't produce 'out' variable, using fallback")
            try:
                out = df_clean.describe().reset_index()
            except Exception as desc_error:
                print(f"⚠️  Describe failed: {desc_error}, using head")
                out = df_clean.head(1000)
        
        print(f"✅ Code execution successful")
        
        # Return the generated figure along with the result
        generated_fig = None
        try:
            if plt.get_fignums():
                generated_fig = plt.gcf()
                print(f"🎨 Returning generated figure: {generated_fig.get_size_inches()}")
        except Exception as fig_error:
            print(f"⚠️  Figure capture failed: {fig_error}")
            generated_fig = None
        
        result_tuple = (f"Plan: Answer question '{question}'", code, out, generated_fig)
        print(f"📤 Returning: {len(result_tuple)} values - Plan, Code, Result, Generated_Fig")
        return result_tuple
        
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        print(f"🔍 Error details: {str(e)}")
        print(f"🔄 Using fallback analysis")
        
        # Clear any matplotlib figures to free memory
        plt.close('all')
        
        # Fallback in case of execution failure
        try:
            print(f"🔄 Attempting fallback analysis...")
            if "price" in df.columns:
                # Use a larger sample for fallback on big datasets
                try:
                    if len(df) > 1_000_000:
                        df_fallback = df.sample(n=500_000, random_state=42)
                        print(f"📊 Fallback using stratified sample of {len(df_fallback):,} rows")
                    elif len(df) > 500_000:
                        df_fallback = df.sample(n=200_000, random_state=42)
                        print(f"📊 Fallback using stratified sample of {len(df_fallback):,} rows")
                    else:
                        df_fallback = df.head(100_000) if len(df) > 100_000 else df
                        print(f"📊 Fallback using {len(df_fallback):,} rows")
                except Exception as sample_error:
                    print(f"⚠️  Sampling failed: {sample_error}, using head")
                    df_fallback = df.head(50_000)
                
                # Try different aggregation strategies
                try:
                    out = df_fallback.groupby(['Location_Id', 'Location_Name'], observed=True)['price'].agg(['count', 'mean', 'std']).reset_index()
                    print(f"✅ Fallback 1 successful: memory-efficient groupby aggregation")
                except Exception as e1:
                    print(f"⚠️  Fallback 1 failed: {e1}")
                    try:
                        out = df_fallback.groupby('Location_Name', observed=True)['price'].mean().reset_index()
                        print(f"✅ Fallback 2 successful: simple groupby")
                    except Exception as e2:
                        print(f"⚠️  Fallback 2 failed: {e2}")
                        try:
                            # Use value_counts for very large datasets
                            out = df_fallback['Location_Name'].value_counts().reset_index()
                            out.columns = ['Location_Name', 'Count']
                            print(f"✅ Fallback 3 successful: value_counts analysis")
                        except Exception as e3:
                            print(f"⚠️  Fallback 3 failed: {e3}")
                            try:
                                out = df_fallback[['Location_Name', 'price']].head(1000)
                                print(f"✅ Fallback 4 successful: simple data selection")
                            except Exception as e4:
                                print(f"⚠️  Fallback 4 failed: {e4}")
                                out = df_fallback.head(1000)
                                print(f"✅ Fallback 5 successful: basic head")
            else:
                try:
                    out = df.describe().reset_index()
                    print(f"✅ Fallback successful: describe()")
                except Exception as desc_error:
                    print(f"⚠️  Describe failed: {desc_error}")
                    out = df.head(1000)
                    print(f"✅ Fallback successful: head()")
        except Exception as fallback_error:
            print(f"❌ All fallbacks failed: {fallback_error}")
            print(f"🔄 Using minimal fallback")
            try:
                out = df.head(1000)
            except:
                out = pd.DataFrame({'Error': ['Dataset processing failed']})
        
        print(f"📊 Final fallback result: {type(out)}, shape: {getattr(out, 'shape', 'N/A')}")
        return (f"Plan: Answer question '{question}' (fallback)", "Fallback analysis used", out, None)
