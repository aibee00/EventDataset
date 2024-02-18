""" 使用说明
Auther: wphu
Date: 20240130
Description: 通过feishu"小小消息看门狗"应用小程序，获取多维表格的在线数据。

Example: 表格: https://aibee.feishu.cn/base/bascn2k1ah2oGvOYjAuwUL1sXDb?table=tbl6VMwJQw49C3d4&view=vew22Vq15E
在右侧 "插件"->"开发工具" 中可以查看以下信息:
- app_token: bascn2k1ah2oGvOYjAuwUL1sXDb
- table_id: tbl6VMwJQw49C3d4
- view_id: vew22Vq15E
- record_id: rec8g2Wr0G
- field_id: fldsReyDQ1

下面这个参数是需要从"小小消息看门狗"的机器人应用中获取，这个应用是我自己创建的。
- user_access_token: u-djZygsirJ5QWxY4zXaVCKWkh5IsM5hXzV0w0l4Ka2yDu
(如果user_access_token过期了请在下面这个地址重新生成: https://open.feishu.cn/api-explorer/cli_a47ad599af3c900b?apiName=get&project=bitable&resource=app&state=undefined&version=v1)

Run:
`python dataset/validation_benchmark/feishu_data.py --user_access_token u-djZygsirJ5QWxY4zXaVCKWkh5IsM5hXzV0w0l4Ka2yDu --app_token bascn2k1ah2oGvOYjAuwUL1sXDb --table_id tbl6VMwJQw49C3d4 --view_id vewIK4hrbD --record_id rec8g2Wr0G --field_id fld5EidPFr --output_path ./tmp/`
"""

import requests
import argparse
import json
from pathlib import Path

class FeishuConfig:
    def __init__(self, user_access_token, app_token, table_id, view_id, record_id, field_id=None):
        self.user_access_token = user_access_token
        self.app_token = app_token
        self.table_id = table_id
        self.view_id = view_id
        self.record_id = record_id
        self.field_id = field_id

def get_table_data(config):
    headers = {"Authorization": f"Bearer {config.user_access_token}"}
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{config.app_token}/tables/{config.table_id}/records"
    params = {"viewId": config.view_id}
    response = requests.get(url, headers=headers, params=params)

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error: Unable to decode JSON from response. Status Code: {response.status_code}, Response: {response.text}")
        return None

def get_specific_record(config):
    headers = {"Authorization": f"Bearer {config.user_access_token}"}
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{config.app_token}/tables/{config.table_id}/records/{config.record_id}"
    response = requests.get(url, headers=headers)
    return response.json()

def get_all_field_infos(config):
    headers = {"Authorization": f"Bearer {config.user_access_token}"}
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{config.app_token}/tables/{config.table_id}/fields/"
    response = requests.get(url, headers=headers)
    return response.json()

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Fetch data from Feishu table.")
    parser.add_argument("--user_access_token", required=True, help="User access token for Feishu app")
    parser.add_argument("--app_token", required=True, help="Base ID for Feishu app")
    parser.add_argument("--table_id", required=True, help="Table ID")
    parser.add_argument("--view_id", required=True, help="View ID")
    parser.add_argument("--record_id", required=True, help="Record ID, 行")
    parser.add_argument("--field_id", help="Field ID (optional), 列")
    parser.add_argument("--output_path", help="Output path (optional)")

    args = parser.parse_args()
    config = FeishuConfig(args.user_access_token, args.app_token, args.table_id, args.view_id, args.record_id, args.field_id)

    table_data = get_table_data(config)
    specific_record = get_specific_record(config)
    all_field_infos = get_all_field_infos(config)

    print("Table Data:", table_data)
    print("Specific Record:", specific_record)
    print("All Field Infos:", all_field_infos)

    # save
    if args.output_path:
        save_path = Path(args.output_path)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        save_to_json(table_data, save_path / "table_data.json")
        save_to_json(specific_record, save_path / "specific_record.json")
        save_to_json(all_field_infos, save_path / "all_field_infos.json")

if __name__ == "__main__":
    main()
