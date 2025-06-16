import os
from sqlalchemy import create_engine, Column, String, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker

if __name__ == "__main__": # for testing
    from youtube_requests import RESOLUTIONS, YouTubeHandler, get_matching_format
else:
    from .youtube_requests import RESOLUTIONS, YouTubeHandler, get_matching_format


Base = declarative_base()

class Video(Base):
    __tablename__ = "Video"
    id = Column(String, primary_key=True)
    format_1080_id = Column(String)
    format_1440_id = Column(String)
    format_2160_id = Column(String)
    format_4320_id = Column(String)
    duration = Column(Float)

def get_1080p_videos(session) -> list[Video]:
    return session.query(Video).filter(Video.format_1080_id.isnot(None)).all()

def get_1440p_videos(session) -> list[Video]:
    return session.query(Video).filter(Video.format_1440_id.isnot(None)).all()

def get_2160p_videos(session) -> list[Video]:
    return session.query(Video).filter(Video.format_2160_id.isnot(None)).all()

def get_4320p_videos(session) -> list[Video]:
    return session.query(Video).filter(Video.format_4320_id.isnot(None)).all()


def get_matching_formats(videos:list["dict"], resolution) -> dict:
    matches = {}
    for v in videos:
        video_id = v.get("id")
        formats = v.get("formats", [])
        try:
            matches[video_id] = get_matching_format(formats, resolution)
        except ValueError:
            continue
    return matches


def populate_database(
    session,
    search_term: str,
    count = 10,
    wipe_old = False,
    min_length: int = 0,
    max_length: int = float('inf')
) -> list[str]:
    """
    Populates database with videos based on search term and optional duration filter.
    Returns a list of string IDs of videos added to the DB.
    - min_length and max_length are in seconds.
    """
    if wipe_old:
        for v in session.query(Video).all():
            session.delete(v)
        session.commit()

    handler = YouTubeHandler()
    results = handler.search_videos_raw(search_term, max_results=count)

    # Filter by video length
    filtered_results = []
    durations = {} 
    for video in results:
        duration = video.get("duration")
        durations[video.get("id")] = duration
        if duration is None:
            continue
        if min_length <= duration <= max_length:
            filtered_results.append(video)

    if not filtered_results:
        print("No videos matched duration criteria.")
        return []

    # Resolutions / DB Column names
    resolutions = {
        RESOLUTIONS.HD_1080: "format_1080_id",
        RESOLUTIONS.HD_1440: "format_1440_id",
        RESOLUTIONS.HD_2160: "format_2160_id",
        RESOLUTIONS.HD_4320: "format_4320_id"
    }

    # Find matches by resolution
    matches = {
        res: get_matching_formats(filtered_results, res)
        for res in resolutions
    }

    # Collect unique video IDs
    all_ids = set()
    for match in matches.values():
        all_ids.update(match.keys())

    # Remove videos that already exist
    if not wipe_old:
        existing_ids = set(i[0] for i in session.query(Video.id).filter(Video.id.in_(all_ids)).all())
        all_ids.difference_update(existing_ids)

    # Populate the db
    additions = []
    for vid in all_ids:
        v = {"id": vid}
        for res, col_key in resolutions.items():
            if (fmt := matches.get(res, {}).get(vid, {}).get("format_id")):
                v[col_key] = fmt
        v["duration"] = durations.get(vid)
        additions.append(Video(**v))

    session.bulk_save_objects(additions)
    session.commit()

    return list(all_ids)


if __name__ == "__main__":
    engine = create_engine("sqlite:///videos.sqlite")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    Base.metadata.create_all(bind=engine)
    session.commit()

    for term in [ # Short videos used for testing
        "Nature 4K 1 Minute",
        "Sports 4K 1 Minute",
    ]:
        populate_database(session, term, count=5, min_length=30, max_length=360)
    
    downloader = YouTubeHandler()

    for v in get_2160p_videos(session):
        downloader.download_video_by_format(v.id, v.format_2160_id, output_path=f"{v.id}.mp4")
