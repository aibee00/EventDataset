This folder mainly for process of validation benchmark data which labeled or verified by tianjin team.
The expected dataset may like to [CAP Dataset](https://visym.github.io/cap/)
---

# What does it will do?
1. We need extract all video clips according to each event.
2. We need make good record for all videos and integrated into one doc.
3. We need to support extend for new benchmark to increasely label our dataset in the futrue.
4. Just by input site_id and date, if we want to add a new store's data.


# Instruction of process details
1. Get all GT events which include 6 main events:
   - Store Inout
   - Region Inout
   - Car Inout
   - Car Visit
   - Group
   - Reception
2. Get pids and event time period, then we will get all channels which covered the location of current pid during this period.
3. Get all channels videos, cut the valid time period for a clip and save to single folder.
4. Record each event type for each clips for easily recogonize or label later.
5. We need give desciption at each ts.

