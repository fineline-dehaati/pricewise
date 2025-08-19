#!/usr/bin/env python3
"""
Build script for creating standalone executable using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_spec_file():
    """Create PyInstaller spec file"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app', 'app'),
        ('data', 'data'),
        ('.streamlit', '.streamlit'),
    ],
    hiddenimports=[
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'plotly',
        'openai',
        'toml',
        'duckdb',
        'prophet',
        'sklearn',
        'scipy',
        'statsmodels',
        'tenacity',
        'dateutil',
        'openpyxl',
        'xlrd',
        'pyarrow'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PriceWise',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PriceWise',
)
'''
    
    with open('PriceWise.spec', 'w') as f:
        f.write(spec_content)
    
    print("✅ Created PriceWise.spec file")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building executable...")
    
    # Build using spec file
    subprocess.check_call([
        'pyinstaller',
        '--clean',
        'PriceWise.spec'
    ])
    
    print("✅ Executable built successfully!")

def create_run_script():
    """Create a simple run script for the executable"""
    if sys.platform.startswith('win'):
        # Windows batch file
        run_script = '''@echo off
echo Starting PriceWise Application...
cd /d "%~dp0"
cd dist\\PriceWise
PriceWise.exe
pause
'''
        with open('run_PriceWise.bat', 'w') as f:
            f.write(run_script)
        print("✅ Created run_PriceWise.bat")
    else:
        # Unix shell script
        run_script = '''#!/bin/bash
echo "Starting PriceWise Application..."
cd "$(dirname "$0")/dist/PriceWise"
./PriceWise
'''
        with open('run_PriceWise.sh', 'w') as f:
            f.write(run_script)
        os.chmod('run_PriceWise.sh', 0o755)
        print("✅ Created run_PriceWise.sh")

def main():
    """Main build process"""
    print("🚀 Starting PriceWise build process...")
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Create spec file
    create_spec_file()
    
    # Build executable
    build_executable()
    
    # Create run script
    create_run_script()
    
    print("\n🎉 Build completed successfully!")
    print("\n📁 Output files:")
    print("   - dist/PriceWise/ (executable and dependencies)")
    print("   - run_PriceWise.bat (Windows run script)")
    print("   - run_PriceWise.sh (Unix run script)")
    print("\n🚀 To run the application:")
    if sys.platform.startswith('win'):
        print("   Double-click run_PriceWise.bat")
    else:
        print("   ./run_PriceWise.sh")

if __name__ == "__main__":
    main()
