#!/usr/bin/env python3
"""
SN85 Miner Health Monitor

Provides real-time monitoring and diagnostics for the Vidaio miner.
Checks service health, GPU status, and validator scoring potential.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import requests


class MinerMonitor:
    """Monitor SN85 miner health and performance."""

    def __init__(self):
        self.services = {
            'upscaler': {'port': 29115, 'name': 'Video Upscaler'},
            'compressor': {'port': 29116, 'name': 'Video Compressor'},
        }
        self.checks = []
        self.warnings = []
        self.errors = []

    def log(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {"info": "ℹ️", "ok": "✅", "warn": "⚠️", "error": "❌"}.get(level, "ℹ️")
        print(f"[{timestamp}] {prefix} {message}")

    def check_system_resources(self) -> Dict:
        """Check CPU, RAM, and disk usage."""
        self.log("Checking system resources...")
        results = {}

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        results['cpu'] = {'usage': cpu_percent, 'status': 'ok' if cpu_percent < 80 else 'warn'}
        if cpu_percent > 80:
            self.warnings.append(f"High CPU usage: {cpu_percent:.1f}%")

        # RAM
        mem = psutil.virtual_memory()
        results['ram'] = {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'status': 'ok' if mem.percent < 85 else 'warn'
        }
        if mem.percent > 85:
            self.warnings.append(f"High RAM usage: {mem.percent:.1f}%")

        # Disk
        disk = psutil.disk_usage('/')
        results['disk'] = {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent,
            'status': 'ok' if disk.percent < 90 else 'warn'
        }
        if disk.percent > 90:
            self.warnings.append(f"Low disk space: {disk.free / (1024**3):.1f} GB free")

        self.log(f"CPU: {cpu_percent:.1f}%, RAM: {mem.percent:.1f}%, Disk: {disk.percent:.1f}%", "ok")
        return results

    def check_gpu(self) -> Dict:
        """Check NVIDIA GPU status."""
        self.log("Checking GPU status...")
        results = {'available': False, 'status': 'error'}

        try:
            # Try to import torch
            import torch
            results['torch_cuda'] = torch.cuda.is_available()

            if torch.cuda.is_available():
                results['available'] = True
                results['device'] = torch.cuda.get_device_name(0)
                results['cuda_version'] = torch.version.cuda
                results['memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Get nvidia-smi output
                smi = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )

                if smi.returncode == 0:
                    util, mem_used, mem_free, temp = smi.stdout.strip().split(', ')
                    results.update({
                        'utilization': float(util),
                        'memory_used_mb': float(mem_used),
                        'memory_free_mb': float(mem_free),
                        'temperature': float(temp),
                    })

                    # Check for issues
                    if float(temp) > 85:
                        self.warnings.append(f"GPU temperature high: {temp}°C")
                    if float(util) > 95:
                        self.warnings.append(f"GPU utilization critical: {util}%")

                    results['status'] = 'ok'
                    self.log(f"GPU: {results['device']} ({util}% util, {temp}°C)", "ok")
                else:
                    self.warnings.append("nvidia-smi failed")

                # Check NVENC availability
                encoders = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-encoders'],
                    capture_output=True, text=True
                )
                results['nvenc_available'] = 'av1_nvenc' in encoders.stdout or 'hevc_nvenc' in encoders.stdout

                if results['nvenc_available']:
                    self.log("NVENC available for hardware encoding", "ok")
            else:
                self.errors.append("CUDA available but no GPU detected")

        except ImportError:
            self.errors.append("PyTorch not installed - GPU check failed")
        except Exception as e:
            self.errors.append(f"GPU check error: {e}")

        if not results['available']:
            self.log("No GPU detected - CPU-only mode", "warn")

        return results

    def check_services(self) -> Dict:
        """Check if services are running and responsive."""
        self.log("Checking services...")
        results = {}

        for key, service in self.services.items():
            port = service['port']
            result = {'port': port, 'status': 'error'}

            # Check if port is in use
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result['listening'] = sock.connect_ex(('localhost', port)) == 0
                sock.close()
            except Exception as e:
                result['listening'] = False

            # Try HTTP health check
            if result['listening']:
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    result['http_status'] = response.status_code
                    result['responsive'] = response.status_code == 200
                    if result['responsive']:
                        result['status'] = 'ok'
                except requests.RequestException:
                    result['responsive'] = False

            results[key] = result

            status = result.get('status', 'error')
            if status == 'ok':
                self.log(f"{service['name']}: Running on port {port}", "ok")
            elif result.get('listening'):
                self.log(f"{service['name']}: Port open but not responsive", "warn")
                self.warnings.append(f"{service['name']} not responding on port {port}")
            else:
                self.log(f"{service['name']}: Not running", "error")
                self.errors.append(f"{service['name']} not running on port {port}")

        return results

    def check_redis(self) -> Dict:
        """Check Redis connection."""
        self.log("Checking Redis...")
        results = {'status': 'error'}

        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=2)
            r.ping()
            results['status'] = 'ok'
            results['version'] = r.info().get('redis_version', 'unknown')
            self.log(f"Redis v{results['version']}: Connected", "ok")
        except ImportError:
            self.errors.append("Redis Python client not installed")
        except redis.ConnectionError:
            self.errors.append("Redis server not running")
        except Exception as e:
            self.errors.append(f"Redis error: {e}")

        return results

    def check_ffmpeg(self) -> Dict:
        """Check FFmpeg installation and capabilities."""
        self.log("Checking FFmpeg...")
        results = {'status': 'error'}

        try:
            version = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True, text=True, timeout=5
            )

            if version.returncode == 0:
                results['status'] = 'ok'
                results['version'] = version.stdout.split()[2]

                # Check for required encoders
                encoders = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-encoders'],
                    capture_output=True, text=True
                )

                results['encoders'] = {
                    'libx264': 'libx264' in encoders.stdout,
                    'libx265': 'libx265' in encoders.stdout,
                    'libsvtav1': 'libsvtav1' in encoders.stdout,
                    'av1_nvenc': 'av1_nvenc' in encoders.stdout,
                    'hevc_nvenc': 'hevc_nvenc' in encoders.stdout,
                }

                available = [k for k, v in results['encoders'].items() if v]
                self.log(f"FFmpeg {results['version']} - Encoders: {', '.join(available)}", "ok")
            else:
                self.errors.append("FFmpeg check failed")
        except FileNotFoundError:
            self.errors.append("FFmpeg not installed")
        except Exception as e:
            self.errors.append(f"FFmpeg check error: {e}")

        return results

    def check_video2x(self) -> Dict:
        """Check Video2X installation."""
        self.log("Checking Video2X...")
        results = {'status': 'error'}

        try:
            binary = os.environ.get('VIDEO2X_BINARY', 'video2x')
            version = subprocess.run(
                [binary, '--version'],
                capture_output=True, text=True, timeout=5
            )

            if version.returncode == 0:
                results['status'] = 'ok'
                results['version'] = version.stdout.strip()
                results['binary'] = binary
                self.log(f"Video2X: {results['version']}", "ok")
            else:
                results['installed'] = False
                self.errors.append("Video2X not properly installed")
        except FileNotFoundError:
            results['installed'] = False
            self.errors.append("Video2X not found - install from GitHub releases")
        except Exception as e:
            self.errors.append(f"Video2X check error: {e}")

        return results

    def check_wallet(self) -> Dict:
        """Check wallet configuration."""
        self.log("Checking wallet...")
        results = {'status': 'error'}

        wallet_name = os.environ.get('BT_WALLET_NAME')
        hotkey = os.environ.get('BT_WALLET_HOTKEY')

        if not wallet_name or not hotkey:
            # Check .env file
            env_path = Path('.env')
            if env_path.exists():
                env_content = env_path.read_text()
                if 'BT_WALLET_NAME=' in env_content:
                    wallet_name = 'configured in .env'
                if 'BT_WALLET_HOTKEY=' in env_content:
                    hotkey = 'configured in .env'

        results['wallet_configured'] = bool(wallet_name and hotkey)

        if results['wallet_configured']:
            results['status'] = 'ok'
            self.log("Wallet: Configured", "ok")
        else:
            self.errors.append("Wallet not configured - set BT_WALLET_NAME and BT_WALLET_HOTKEY")

        return results

    def check_scoring_simulation(self) -> Dict:
        """Run quick scoring simulation for competitive baseline."""
        self.log("Running scoring simulation...")
        results = {}

        try:
            # Import scoring function
            sys.path.insert(0, 'services/scoring')
            from scoring_function import calculate_score

            # Test scenarios
            scenarios = [
                (15, 89, 89, "Target: 15x @ threshold"),
                (20, 91, 89, "Competitive: 20x @ +2 VMAF"),
                (10, 92, 89, "Quality: 10x @ +3 VMAF"),
                (15, 93, 93, "High tier: 15x @ threshold"),
            ]

            results['scenarios'] = []
            for ratio, vmaf, threshold, desc in scenarios:
                score = calculate_score(ratio, vmaf, threshold)
                results['scenarios'].append({
                    'description': desc,
                    'compression_ratio': ratio,
                    'vmaf': vmaf,
                    'threshold': threshold,
                    'score': score
                })

            # Find sweet spot
            scores = [s['score'] for s in results['scenarios']]
            best = max(results['scenarios'], key=lambda x: x['score'])

            self.log(f"Best strategy: {best['description']} = {best['score']:.3f}", "ok")

            # Estimate competitive score
            if best['score'] >= 0.75:
                results['competitive_status'] = 'excellent'
            elif best['score'] >= 0.70:
                results['competitive_status'] = 'good'
            elif best['score'] >= 0.60:
                results['competitive_status'] = 'fair'
            else:
                results['competitive_status'] = 'needs_improvement'

        except Exception as e:
            results['error'] = str(e)
            self.log("Could not run scoring simulation", "warn")

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a formatted health report."""
        lines = [
            "\n" + "=" * 60,
            "SN85 MINER HEALTH REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "SYSTEM RESOURCES",
            "-" * 40,
        ]

        sys_info = results.get('system', {})
        cpu = sys_info.get('cpu', {})
        ram = sys_info.get('ram', {})
        disk = sys_info.get('disk', {})

        lines.extend([
            f"  CPU Usage: {cpu.get('usage', 'N/A'):.1f}%",
            f"  RAM: {ram.get('used_gb', 0):.1f} / {ram.get('total_gb', 0):.1f} GB ({ram.get('percent', 0):.1f}%)",
            f"  Disk: {disk.get('free_gb', 0):.1f} GB free ({100 - disk.get('percent', 0):.1f}%)",
            "",
            "GPU STATUS",
            "-" * 40,
        ])

        gpu = results.get('gpu', {})
        if gpu.get('available'):
            lines.extend([
                f"  Device: {gpu.get('device', 'N/A')}",
                f"  NVENC: {'Available' if gpu.get('nvenc_available') else 'Not Available'}",
                f"  Memory: {gpu.get('memory_used_mb', 0):.0f} / {gpu.get('memory_total_gb', 0) * 1024:.0f} MB",
                f"  Temperature: {gpu.get('temperature', 0):.0f}°C",
                f"  Utilization: {gpu.get('utilization', 0):.0f}%",
            ])
        else:
            lines.append("  Status: Not Available (CPU-only mode)")

        lines.extend([
            "",
            "SERVICES",
            "-" * 40,
        ])

        for key, service in results.get('services', {}).items():
            status = service.get('status', 'error')
            icon = "✅" if status == 'ok' else "❌"
            lines.append(f"  {icon} {self.services[key]['name']}: {status.upper()}")

        lines.extend([
            "",
            "DEPENDENCIES",
            "-" * 40,
        ])

        redis_ok = results.get('redis', {}).get('status') == 'ok'
        ffmpeg_ok = results.get('ffmpeg', {}).get('status') == 'ok'
        video2x_ok = results.get('video2x', {}).get('status') == 'ok'
        wallet_ok = results.get('wallet', {}).get('status') == 'ok'

        lines.extend([
            f"  {'✅' if redis_ok else '❌'} Redis",
            f"  {'✅' if ffmpeg_ok else '❌'} FFmpeg",
            f"  {'✅' if video2x_ok else '❌'} Video2X",
            f"  {'✅' if wallet_ok else '❌'} Wallet",
        ])

        if self.warnings:
            lines.extend([
                "",
                "WARNINGS",
                "-" * 40,
            ])
            for warning in self.warnings:
                lines.append(f"  ⚠️ {warning}")

        if self.errors:
            lines.extend([
                "",
                "ERRORS - ACTION REQUIRED",
                "-" * 40,
            ])
            for error in self.errors:
                lines.append(f"  ❌ {error}")

        scoring = results.get('scoring', {})
        if scoring.get('scenarios'):
            lines.extend([
                "",
                "COMPETITIVE BASELINE",
                "-" * 40,
                f"  Target Strategy: {scoring.get('competitive_status', 'unknown').upper()}",
                "",
            ])
            for scenario in scoring['scenarios']:
                lines.append(
                    f"  {scenario['description']}: "
                    f"Score = {scenario['score']:.3f}"
                )

        lines.extend([
            "",
            "=" * 60,
            f"Overall Status: {'HEALTHY' if not self.errors else 'ISSUES DETECTED'}",
            "=" * 60,
        ])

        return '\n'.join(lines)

    async def run_checks(self) -> Dict:
        """Run all health checks."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'system': self.check_system_resources(),
            'gpu': self.check_gpu(),
            'services': self.check_services(),
            'redis': self.check_redis(),
            'ffmpeg': self.check_ffmpeg(),
            'video2x': self.check_video2x(),
            'wallet': self.check_wallet(),
            'scoring': self.check_scoring_simulation(),
        }

        return results

    def save_results(self, results: Dict, output_path: Optional[str] = None):
        """Save results to JSON file."""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/health_check_{timestamp}.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='SN85 Miner Health Monitor')
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--watch', type=int, metavar='SECONDS', help='Watch mode - run every N seconds')
    args = parser.parse_args()

    monitor = MinerMonitor()

    async def run_once():
        results = await monitor.run_checks()

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(monitor.generate_report(results))

        if args.save:
            monitor.save_results(results)

        return len(monitor.errors) == 0

    if args.watch:
        print(f"Monitoring every {args.watch} seconds (Ctrl+C to stop)...\n")
        try:
            while True:
                asyncio.run(run_once())
                print(f"\n{'='*60}")
                print(f"Next check in {args.watch}s...")
                print('='*60 + "\n")
                time.sleep(args.watch)
                monitor.warnings = []
                monitor.errors = []
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        success = asyncio.run(run_once())
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
