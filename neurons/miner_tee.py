"""
TEE Miner Entry Point

This module provides the entry point for miners running inside Intel SGX
enclaves using Gramine. It handles:

1. Attestation with validators
2. Receiving encrypted task payloads
3. Processing videos securely inside the enclave
4. Returning results without exposing any video URLs

Usage:
    # Inside SGX enclave (via Gramine)
    gramine-sgx miner
    
    # For testing without SGX
    python -m neurons.miner_tee --tee-mode
"""

import os
import sys
import time
import asyncio
import argparse
import secrets
import base64
from typing import Optional
from loguru import logger

# Bittensor imports
import bittensor as bt
from bittensor import axon, wallet, subtensor

# Project imports
from vidaio_subnet_core.protocol import (
    TEEAttestationProtocol,
    TEEVideoUpscalingProtocol,
    TEEVideoCompressionProtocol,
    TEEMinerResponse,
    TEECapabilities,
    Version,
)
from vidaio_subnet_core.tee.tee_crypto import (
    TEECrypto,
    EncryptedPayload,
)
from vidaio_subnet_core.tee.secret_provisioning import (
    SecretProvisioningClient,
)
from services.tee.tee_video_processor import (
    TEEVideoProcessor,
    ProcessingResult,
)


# Current version
MINER_VERSION = Version(major=1, minor=0, patch=0)


class TEEMiner:
    """
    TEE-enabled miner running inside Intel SGX enclave.
    
    This miner:
    - Performs attestation with validators
    - Receives encrypted task payloads
    - Processes videos entirely inside the enclave
    - Never exposes video URLs or content to the outside
    """
    
    def __init__(
        self,
        wallet: bt.wallet,
        subtensor: bt.subtensor,
        netuid: int = 1,
        axon_port: int = 8091,
        tee_mode: bool = True,
    ):
        """
        Initialize TEE miner.
        
        Args:
            wallet: Bittensor wallet
            subtensor: Subtensor connection
            netuid: Network UID
            axon_port: Port for axon server
            tee_mode: Whether running in TEE mode (should always be True in production)
        """
        self.wallet = wallet
        self.subtensor = subtensor
        self.netuid = netuid
        self.axon_port = axon_port
        self.tee_mode = tee_mode
        
        # Initialize TEE components
        self.crypto = TEECrypto()
        self.secret_client = SecretProvisioningClient()
        self.video_processor = TEEVideoProcessor(crypto=self.crypto)
        
        # Session keys from attestation (session_key_id -> session_key)
        self.session_keys: dict = {}
        
        # Initialize axon
        self.axon = bt.axon(
            wallet=self.wallet,
            port=self.axon_port,
        )
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"TEE Miner initialized (TEE mode: {tee_mode})")
        logger.info(f"Running inside enclave: {self.secret_client._is_inside_enclave}")
    
    def _register_handlers(self) -> None:
        """Register synapse handlers with axon."""
        # Attestation handler
        self.axon.attach(
            forward_fn=self.forward_attestation,
            blacklist_fn=self.blacklist_attestation,
            priority_fn=self.priority_attestation,
        )
        
        # TEE upscaling handler
        self.axon.attach(
            forward_fn=self.forward_tee_upscaling,
            blacklist_fn=self.blacklist_tee_request,
            priority_fn=self.priority_tee_request,
        )
        
        # TEE compression handler
        self.axon.attach(
            forward_fn=self.forward_tee_compression,
            blacklist_fn=self.blacklist_tee_request,
            priority_fn=self.priority_tee_request,
        )
        
        logger.info("Registered TEE synapse handlers")
    
    # =========================================================================
    # Attestation Handler
    # =========================================================================
    
    async def forward_attestation(
        self,
        synapse: TEEAttestationProtocol
    ) -> TEEAttestationProtocol:
        """
        Handle attestation request from validator.
        
        Flow:
        1. Receive challenge from validator
        2. Generate SGX Quote embedding the challenge
        3. Return quote and capabilities
        4. Validator will verify and provision session key
        """
        logger.info("Received attestation request")
        
        try:
            # Decode challenge
            challenge = base64.b64decode(synapse.challenge) if synapse.challenge else b""
            
            # Generate SGX quote
            quote = self.secret_client.generate_quote(challenge)
            
            # Get capabilities
            capabilities = TEECapabilities(
                sgx_supported=self.secret_client._is_inside_enclave,
                attestation_type=self.secret_client.get_attestation_type(),
                enclave_version=f"{MINER_VERSION.major}.{MINER_VERSION.minor}.{MINER_VERSION.patch}",
                max_enclave_memory_mb=8192,  # 8GB enclave
            )
            
            # Set response
            synapse.sgx_quote = base64.b64encode(quote).decode('utf-8')
            synapse.capabilities = capabilities
            synapse.version = MINER_VERSION
            
            logger.info(f"Generated attestation response (SGX: {capabilities.sgx_supported})")
            
            # If validator returns a session key (in follow-up), store it
            if synapse.encrypted_session_key and synapse.session_key_id:
                session_key = base64.b64decode(synapse.encrypted_session_key)
                self.session_keys[synapse.session_key_id] = session_key
                self.crypto.store_session_key(synapse.session_key_id, session_key)
                synapse.attestation_success = True
                logger.info(f"Stored session key: {synapse.session_key_id}")
            
        except Exception as e:
            logger.error(f"Attestation failed: {e}")
            synapse.attestation_success = False
            synapse.attestation_error = str(e)
        
        return synapse
    
    def blacklist_attestation(
        self,
        synapse: TEEAttestationProtocol
    ) -> tuple[bool, str]:
        """Check if attestation request should be blacklisted."""
        # Allow all attestation requests for now
        return False, ""
    
    def priority_attestation(
        self,
        synapse: TEEAttestationProtocol
    ) -> float:
        """Return priority for attestation request."""
        return 1.0
    
    # =========================================================================
    # TEE Upscaling Handler
    # =========================================================================
    
    async def forward_tee_upscaling(
        self,
        synapse: TEEVideoUpscalingProtocol
    ) -> TEEVideoUpscalingProtocol:
        """
        Handle TEE upscaling request.
        
        The encrypted payload contains:
        - Video URL (only known inside enclave)
        - Storage credentials (only known inside enclave)
        - Task parameters
        
        We return only success/failure and checksum - no URLs.
        """
        logger.info(f"Received TEE upscaling request (round: {synapse.round_id})")
        
        try:
            # Get session key
            session_key_id = synapse.encrypted_payload.session_key_id
            session_key = self.session_keys.get(session_key_id)
            
            if not session_key:
                session_key = self.crypto.get_session_key(session_key_id)
            
            if not session_key:
                logger.error(f"Session key not found: {session_key_id}")
                synapse.miner_response = TEEMinerResponse(
                    success=False,
                    error_message="Session key not found - attestation required"
                )
                return synapse
            
            # Create EncryptedPayload from synapse
            encrypted_payload = EncryptedPayload(
                encrypted_blob=base64.b64decode(synapse.encrypted_payload.encrypted_blob),
                nonce=base64.b64decode(synapse.encrypted_payload.nonce),
                session_key_id=session_key_id,
            )
            
            # Process inside enclave
            result = await self.video_processor.process_encrypted_task(
                encrypted_payload,
                session_key
            )
            
            # Return response (no URLs!)
            synapse.miner_response = TEEMinerResponse(
                success=result.success,
                processing_time_seconds=result.processing_time_seconds,
                error_message=result.error_message,
                result_checksum=result.result_checksum,
            )
            
            synapse.version = MINER_VERSION
            
            logger.info(f"TEE upscaling completed: success={result.success}")
            
        except Exception as e:
            logger.error(f"TEE upscaling failed: {e}")
            synapse.miner_response = TEEMinerResponse(
                success=False,
                error_message=str(e)
            )
        
        return synapse
    
    # =========================================================================
    # TEE Compression Handler
    # =========================================================================
    
    async def forward_tee_compression(
        self,
        synapse: TEEVideoCompressionProtocol
    ) -> TEEVideoCompressionProtocol:
        """
        Handle TEE compression request.
        
        Same security model as upscaling - all sensitive data stays in enclave.
        """
        logger.info(f"Received TEE compression request (round: {synapse.round_id})")
        
        try:
            # Get session key
            session_key_id = synapse.encrypted_payload.session_key_id
            session_key = self.session_keys.get(session_key_id)
            
            if not session_key:
                session_key = self.crypto.get_session_key(session_key_id)
            
            if not session_key:
                logger.error(f"Session key not found: {session_key_id}")
                synapse.miner_response = TEEMinerResponse(
                    success=False,
                    error_message="Session key not found - attestation required"
                )
                return synapse
            
            # Create EncryptedPayload from synapse
            encrypted_payload = EncryptedPayload(
                encrypted_blob=base64.b64decode(synapse.encrypted_payload.encrypted_blob),
                nonce=base64.b64decode(synapse.encrypted_payload.nonce),
                session_key_id=session_key_id,
            )
            
            # Process inside enclave
            result = await self.video_processor.process_encrypted_task(
                encrypted_payload,
                session_key
            )
            
            # Return response (no URLs!)
            synapse.miner_response = TEEMinerResponse(
                success=result.success,
                processing_time_seconds=result.processing_time_seconds,
                error_message=result.error_message,
                result_checksum=result.result_checksum,
            )
            
            synapse.version = MINER_VERSION
            
            logger.info(f"TEE compression completed: success={result.success}")
            
        except Exception as e:
            logger.error(f"TEE compression failed: {e}")
            synapse.miner_response = TEEMinerResponse(
                success=False,
                error_message=str(e)
            )
        
        return synapse
    
    # =========================================================================
    # Common Blacklist/Priority Functions
    # =========================================================================
    
    def blacklist_tee_request(
        self,
        synapse
    ) -> tuple[bool, str]:
        """Check if TEE request should be blacklisted."""
        # Verify session key exists
        session_key_id = synapse.encrypted_payload.session_key_id
        if not session_key_id:
            return True, "Missing session_key_id - attestation required first"
        
        if session_key_id not in self.session_keys:
            if not self.crypto.get_session_key(session_key_id):
                return True, "Unknown session_key_id - attestation required first"
        
        return False, ""
    
    def priority_tee_request(
        self,
        synapse
    ) -> float:
        """Return priority for TEE request."""
        return 1.0
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self) -> None:
        """Start the TEE miner."""
        logger.info("Starting TEE Miner...")
        
        # Start axon
        self.axon.serve(
            netuid=self.netuid,
            subtensor=self.subtensor,
        )
        self.axon.start()
        
        logger.info(f"TEE Miner started on port {self.axon_port}")
        logger.info(f"Waiting for attestation requests from validators...")
    
    def stop(self) -> None:
        """Stop the TEE miner."""
        logger.info("Stopping TEE Miner...")
        self.axon.stop()
        
        # Clear session keys from memory
        self.session_keys.clear()
        logger.info("TEE Miner stopped")
    
    async def run_loop(self) -> None:
        """Main run loop."""
        try:
            while True:
                await asyncio.sleep(10)
                # Periodic maintenance could go here
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()


def main():
    """Main entry point for TEE miner."""
    parser = argparse.ArgumentParser(description="vidaio TEE Miner")
    
    parser.add_argument(
        "--netuid",
        type=int,
        default=1,
        help="Network UID"
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="miner",
        help="Wallet name"
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey"
    )
    parser.add_argument(
        "--subtensor.network",
        dest="subtensor_network",
        type=str,
        default="finney",
        help="Subtensor network"
    )
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=8091,
        help="Axon port"
    )
    parser.add_argument(
        "--tee-mode",
        action="store_true",
        default=True,
        help="Enable TEE mode (default: True)"
    )
    parser.add_argument(
        "--no-tee",
        action="store_true",
        help="Disable TEE mode (for testing only)"
    )
    
    args = parser.parse_args()
    
    # Check TEE environment
    tee_mode = args.tee_mode and not args.no_tee
    if tee_mode:
        if not os.path.exists("/dev/attestation/quote"):
            logger.warning("Not running inside SGX enclave - attestation will be mocked")
    
    # Initialize wallet and subtensor
    wallet = bt.wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
    )
    
    subtensor = bt.subtensor(network=args.subtensor_network)
    
    # Create and start miner
    miner = TEEMiner(
        wallet=wallet,
        subtensor=subtensor,
        netuid=args.netuid,
        axon_port=args.axon_port,
        tee_mode=tee_mode,
    )
    
    miner.start()
    
    # Run event loop
    try:
        asyncio.run(miner.run_loop())
    except KeyboardInterrupt:
        miner.stop()


if __name__ == "__main__":
    main()
