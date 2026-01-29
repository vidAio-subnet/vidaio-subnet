# Makefile for building Gramine SGX manifests
#
# Usage:
#   make           - Build the SGX manifest
#   make clean     - Clean build artifacts
#   make check     - Verify SGX availability
#
# Requirements:
#   - Gramine installed (gramine, gramine-sgx-gen-private-key, gramine-manifest, gramine-sgx-sign)
#   - Intel SGX SDK and PSW installed
#   - Python 3.9+ installed

GRAMINE_SIGN_KEY ?= ~/.config/gramine/enclave-key.pem
PYTHON_PATH ?= $(shell which python3)
PYTHON_VERSION ?= $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_STDLIB ?= /usr/lib/python$(PYTHON_VERSION)
PYTHON_DISTLIB ?= /usr/local/lib/python$(PYTHON_VERSION)/dist-packages
ARCH_LIBDIR ?= /lib/x86_64-linux-gnu

# Debug mode (set to 0 for production)
DEBUG_MODE ?= 1

# Log level (error, warning, info, debug, trace)
LOG_LEVEL ?= error

.PHONY: all clean check distclean

all: miner.manifest.sgx

# Generate private key if not exists
$(GRAMINE_SIGN_KEY):
	@echo "Generating Gramine signing key..."
	gramine-sgx-gen-private-key

# Generate manifest from template
miner.manifest: miner.manifest.template
	@echo "Generating manifest from template..."
	gramine-manifest \
		-Dpython_path=$(PYTHON_PATH) \
		-Dpython_stdlib=$(PYTHON_STDLIB) \
		-Dpython_distlib=$(PYTHON_DISTLIB) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dlog_level=$(LOG_LEVEL) \
		-Ddebug_mode=$(DEBUG_MODE) \
		$< > $@

# Sign the manifest for SGX
miner.manifest.sgx miner.sig: miner.manifest $(GRAMINE_SIGN_KEY)
	@echo "Signing manifest for SGX..."
	gramine-sgx-sign \
		--key $(GRAMINE_SIGN_KEY) \
		--manifest $< \
		--output $@
	@echo ""
	@echo "=== MRENCLAVE ==="
	@gramine-sgx-sigstruct-view miner.sig | grep -E "mr_enclave|mr_signer"
	@echo ""
	@echo "Update the allowed_mrenclaves in attestation_verifier.py with the mr_enclave value above."

# Check SGX availability
check:
	@echo "Checking SGX availability..."
	@which is-sgx-available && is-sgx-available || echo "is-sgx-available not found"
	@echo ""
	@ls -la /dev/sgx* 2>/dev/null || echo "SGX devices not found"
	@echo ""
	@echo "Checking for AESM service..."
	@systemctl status aesmd 2>/dev/null || echo "AESM service not found"

# Clean build artifacts
clean:
	rm -f miner.manifest miner.manifest.sgx miner.sig

# Full clean including signing key
distclean: clean
	rm -f $(GRAMINE_SIGN_KEY)

# Print MRENCLAVE from existing signature
mrenclave:
	@gramine-sgx-sigstruct-view miner.sig 2>/dev/null | grep mr_enclave || echo "No signature found. Run 'make' first."

# Run the enclave (for testing)
run: miner.manifest.sgx
	gramine-sgx miner

# Run in direct mode (without SGX, for debugging)
run-direct: miner.manifest
	gramine-direct miner
