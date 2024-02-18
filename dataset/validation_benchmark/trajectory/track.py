"""
轨迹相关的处理
"""
import os
import os.path as osp
import shutil
import tarfile
import requests

from pathlib import Path

from dataset.common import get_location, download, get_location_pid_output, is_directory_empty


class TrackLoader:
    def __init__(self, hdfs_client = None, hdfs_pid_output: str = "", pid_output_save_path: str = ""):
        self.client = hdfs_client # hdfs client
        self.hdfs_pid_output = hdfs_pid_output
        self.pid_output_save_path = pid_output_save_path
        self.xy = None

        # 定义proto的查找表
        self.proto_table = {}

    def _download_new_xy(self, new_xy_url, local_root):
        local_pb_path = local_root / "new_xy"
        local_pb_path.mkdir(parents=True, exist_ok=True)
        local_file_path = local_root / "new_xy.tar.gz"
        if is_directory_empty(local_pb_path):
            if not local_file_path.exists():
                if new_xy_url.startswith("/bj_dev"):
                    self.client.download(new_xy_url, local_file_path.as_posix())
                else:
                    download(new_xy_url, local_file_path.as_posix())
            else:
                print("File already exists, skip downloading.")
            
            # 解压tar包
            with tarfile.open(local_file_path.as_posix()) as tar:
                tar.extractall(path=local_pb_path.as_posix())
        
        # 删除tar包
        if osp.exists(f"{local_root.as_posix()}/new_xy.tar.gz"):
            os.remove(f"{local_root.as_posix()}/new_xy.tar.gz")
        
        return local_pb_path.as_posix()
    
    def _download_pid_output(self, pid_output_url, local_root):
        local_pb_path = local_root / "pid_output"
        local_file_path = local_root / "pid_output.tar"
        if not local_pb_path.exists() or is_directory_empty(local_pb_path):
            if not local_file_path.exists():
                if pid_output_url.startswith("/bj_dev"):
                    self.client.download(pid_output_url, local_file_path.as_posix())
                else:
                    download(pid_output_url, local_file_path.as_posix())
            else:
                print("File already exists, skip downloading.")

            # 解压tar包
            with tarfile.open(local_file_path.as_posix()) as tar:
                tar.extractall(path=local_pb_path.parent.as_posix())

        # 删除tar包
        if osp.exists(f"{local_root.as_posix()}/pid_output.tar"):
            os.remove(f"{local_root.as_posix()}/pid_output.tar")
        
        return local_pb_path.as_posix()

    def _get_local_path(self, hdfs_pid_output):
        """
        获取new_xy的本地路径
        """
        if hdfs_pid_output.startswith("https://") or hdfs_pid_output.startswith("/bj_dev"):
            local_root = Path(self.pid_output_save_path) / 'track_pb'
            local_root.mkdir(parents=True, exist_ok=True)
            
            # Download from oss
            if hdfs_pid_output.endswith(".tar"):
                local_pb_path = self._download_pid_output(hdfs_pid_output, local_root)
                return local_pb_path
            elif hdfs_pid_output.endswith(".tar.gz"):
                local_pb_path = self._download_new_xy(hdfs_pid_output, local_root)
                return local_pb_path
            else:
                raise ValueError("Invalid hdfs_pid_output format.")
        
        return hdfs_pid_output

    # 定义一个_get_location的函数，用以获取某个pid的轨迹信息
    def get_locs(self, pid, ts=None):
        """ 检查该proto是否已经在proto_table中，如果没有则调用get_location读取
        如果已经在查找表中，则直接返回
        """
        if self.xy is None:
            self.xy = self._get_local_path(self.hdfs_pid_output)

        # pid 预处理，还原_new.pb，因为这是人为加上的pid
        pid_raw = pid
        if pid.endswith("_new"):
            pid_raw = pid.replace("_new", "")
        
        pid_proto_path = osp.join(self.xy, f"{pid_raw}.pb")
        if pid not in self.proto_table.keys():
            if self.hdfs_pid_output.endswith(".tar"):  # pid_output
                pid_locs = get_location_pid_output(pid_proto_path)
                for _pid_raw, loc in pid_locs.items():
                    self.proto_table[pid] = {pid: loc}
            elif self.hdfs_pid_output.endswith(".tar.gz"):  # new_xy
                pid_locs = get_location(pid_proto_path)
                for _pid_raw, loc in pid_locs.items():
                    self.proto_table[pid] = {pid: loc}
            else:
                raise ValueError("Invalid hdfs_pid_output format.")
        
        ret = None
        if ts is None:
            ret = self.proto_table.get(pid, None)
        else:
            ret = self.proto_table[pid][pid]['loc'][ts]
        
        return ret

