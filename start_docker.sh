nvidia-docker run --rm -it \
    -v /ssd/:/ssd \
    -v ${PWD}:/workspace_wphu \
    -v /ssd/wphu/chatglm/THUDM:/THUDM \
    -p 7861:7860 \
    registry.aibee.cn/aibee/eventgpt:grit.v1.0 bash
    # registry.aibee.cn/aibee/visualglm:torch1.9.v1.0 bash
