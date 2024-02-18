import os
import time
import logging
import requests
import json

from pathlib import Path
from tqdm import tqdm

from dataset.validation_benchmark.events.events import BaseEvent
from video_config import VideoConfig


class VideoDownloader(object):
    """ Downloader for one video that corresponding to one event.
    Args:
        config: VideoConfig object.
    
    Note:
        One video corresponds to one event.
        One event corresponds to one video_config.
        One video_config corresponds to one video_path/video_id.
        One video corresponds to one video_meta_path or video_meta_file.
    """
    def __init__(self, config: VideoConfig = None):
        self.config = config

    @property
    def video_path(self):
        assert self.config is not None, "config must be set, please use self.inject_config(config) to update config."
        return self.config.video_path

    @property
    def video_meta_path(self):
        assert self.config, "config must be set, please use self.inject_config(config) to update config."
        return self.config.video_meta_path
    
    @property
    def video_meta_file(self):
        assert self.config, "config must be set, please use self.inject_config(config) to update config."
        return self.config.video_meta_file

    def inject_config(self, config):
        self.config = config

    def is_video_exist(self, site_id, date, event):
        """ Check if video already exists.
        Args:
            site_id: site id.
            date: date.
            event: event object.
        Returns:
            True if video already exists, False otherwise.
        """
        video_path = os.path.join(self.video_path, f"{site_id}_{date}", event.video_name)
        return os.path.exists(video_path)

    def download_video(self, site_id, date, event):
        """ Download video for event.
        Args:
            event: event object.
        Returns:
            None.
        """
        assert isinstance(event, BaseEvent), "event must be instacne of subclass of BaseEvent"

        self.config.video_id = f"{site_id}_{date}"
        self.config.video_name = event.video_name

        self.download()

    def download(self,):
        video_output_path = Path(self.config.video_path)
        if not video_output_path.exists():
            video_output_path.mkdir(parents=True)

        video_meta_output_path = Path(self.config.video_meta_path)
        if not video_meta_output_path.exists():
            video_meta_output_path.mkdir(parents=True)

        video_path = video_output_path / f"{self.config.video_name}"
        video_meta_path = video_meta_output_path / f"{self.config.video_name}.json"

        if os.path.exists(video_path) and os.path.exists(video_meta_path):
            logging.info(f"Video {self.config.video_id} already exists.")
            return

        logging.info(f"Downloading video {self.config.video_id} ...")
        start_time = time.time()
        
        if not os.path.exists(video_path):
            try:
                response = requests.get(self.config.video_url, stream=True)
                if response.status_code == 200:
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kibibyte
                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading Video")

                    with open(video_path, "wb") as f:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            f.write(data)
                            if time.time() - start_time > 60:
                                logging.info(f"Timeout downloading {self.config.video_id}")
                                break
                    progress_bar.close()

                    if progress_bar.n != total_size_in_bytes:
                        logging.info(f"Error, downloaded video might be incomplete. Downloaded: {progress_bar.n / 1024 / 1024} MB, Expected: {total_size_in_bytes / 1024 / 1024} MB")
                    else:
                        logging.info(f"Video {self.config.video_id} downloaded successfully. Total size: {progress_bar.n / 1024 / 1024} MB")

                else:
                    logging.info(f"Video {self.config.video_id} download failed. Response status code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logging.info(f"Video {self.config.video_id} download failed: {e}")

        # Gen video_meta info
        if not os.path.exists(video_meta_path):
            video_meta = self.config.to_dict()

            with open(video_meta_path, "w", encoding='utf-8') as f:
                json.dump(video_meta, f, indent=4, ensure_ascii=False)
                logging.info(f"Video {self.config.video_id} meta saved.")

