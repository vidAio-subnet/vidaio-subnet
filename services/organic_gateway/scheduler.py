from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends

from config import logger
from services import get_redis_connection, TaskService
from config import get_settings

# Data cleanup scheduler
scheduler = BackgroundScheduler()

def setup_data_cleanup_job(settings=Depends(get_settings)):
    """Setup scheduled job for data cleanup"""
    redis_conn = get_redis_connection(settings)
    task_service = TaskService(redis_conn)
    
    def cleanup_job():
        try:
            task_service.cleanup_old_data(settings.DATA_RETENTION_DAYS)
        except Exception as e:
            logger.error(f"Error during scheduled data cleanup: {str(e)}")
    
    # Schedule job to run at the specified interval
    scheduler.add_job(
        cleanup_job, 
        'interval', 
        hours=settings.CLEANUP_INTERVAL_HOURS,
        id='data_cleanup_job',
        replace_existing=True
    )
    
    # Run once at startup
    cleanup_job()
    
    if not scheduler.running:
        scheduler.start()
