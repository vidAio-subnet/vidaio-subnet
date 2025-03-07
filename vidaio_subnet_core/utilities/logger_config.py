import sys
import os
from pathlib import Path
from loguru import logger

class LoggerConfig:
    def __init__(self):
        self.is_pm2 = "PM2_HOME" in os.environ
        
        self.project_root = Path(__file__).parent.parent.parent
        
        self.log_path = self.project_root / "logs"
        
        self.config = {
            'development': {
                'console_level': 'DEBUG',
                'file_level': 'DEBUG',
                'colorize': True
            },
            'production': {
                'console_level': 'INFO',
                'file_level': 'INFO',
                'colorize': not self.is_pm2
            }
        }
        
        self.env = os.getenv('APP_ENV', 'development')

    def ensure_log_directory(self):
        """Ensure log directory exists and is writable"""
        try:
            self.log_path.mkdir(parents=True, exist_ok=True)
            
            test_file = self.log_path / '.write_test'
            test_file.touch()
            test_file.unlink()  
            
            logger.info(f"Log directory initialized: {self.log_path}")
        except Exception as e:
            sys.stderr.write(f"Failed to create or access log directory {self.log_path}: {e}\n")
            raise

    def setup_logger(self):
        """Configure logger with all handlers and settings"""
        logger.remove()
        
        self.ensure_log_directory()
        
        env_config = self.config[self.env]
        
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        try:
            logger.add(
                sys.stdout,
                format=log_format,
                colorize=env_config['colorize'],
                level=env_config['console_level'],
                enqueue=True
            )

            logger.add(
                self.log_path / "app_{time}.log",
                rotation="500 MB",
                retention="30 days",
                compression="zip",
                format=log_format,
                level=env_config['file_level'],
                enqueue=True,
                backtrace=True,
                diagnose=True
            )

            logger.add(
                self.log_path / "error_{time}.log",
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                format=log_format,
                level="ERROR",
                enqueue=True,
                backtrace=True,
                diagnose=True
            )

            logger.level("STARTUP", no=15, color="<cyan>")

            logger.configure(
                extra={
                    "environment": self.env,
                    "app_name": os.getenv('APP_NAME', 'default_app')
                }
            )

            logger.info(f"Logger initialized successfully in {self.env} environment")
            
        except Exception as e:
            sys.stderr.write(f"Failed to setup logger: {e}\n")
            raise

    @staticmethod
    def get_logger():
        """Get the configured logger instance"""
        return logger

    @staticmethod
    def add_correlation_id(correlation_id):
        """Add correlation ID to logger context"""
        return logger.bind(correlation_id=correlation_id)

    @staticmethod
    def create_module_logger(module_name):
        """Create a logger instance with module context"""
        return logger.bind(module=module_name)

def initialize_logger():
    """Initialize and configure the logger"""
    log_config = LoggerConfig()
    log_config.setup_logger()
    return logger

logger = initialize_logger()
