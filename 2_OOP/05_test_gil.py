import numpy as np
import threading
import time
import my_ops

# 这是一个常驻的后台 Python 任务（比如网络下载、数据预处理）
def python_worker():
    for i in range(5):
        print(f"   --> [Python Thread] I am free to run! (Tick {i+1})")
        time.sleep(1)

if __name__ == "__main__":
    # 启动后台 Python 线程
    t = threading.Thread(target=python_worker)
    t.start()

    time.sleep(0.5) # 稍微等一下，让后台线程先跑起来
    
    arr = np.array([1.0, 2.0], dtype=np.float32)
    print("\n[Main] Calling C++ operator...")
    
    # 召唤 C++ 算子。
    # 在这被 C++ 卡住的 3 秒内，看看上面的 python_worker 会不会继续输出！
    my_ops.slow_inplace(arr) 

    print("[Main] C++ operator finished! Array:", arr)
    t.join()