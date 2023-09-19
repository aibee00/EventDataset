import streamlit as st

# 定义计数器变量
counter = st.session_state.get("counter", 0)

# 点击按钮时增加计数器值
if st.button("Increment"):
    counter += 1

# 显示计数器的值
st.write("Counter:", counter)

# 更新计数器的值到状态中
st.session_state["counter"] = counter

