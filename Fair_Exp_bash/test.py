# 写一个循环python脚本，该python脚本每隔1s就打印当前时间，一直循环直到被手动终止。import time
from datetime import datetime
import time
while True:
    print("当前时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    time.sleep(1)
    