import re
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器或无GUI环境
import matplotlib.pyplot as plt

log_path = "output.txt"  # 请替换为你的实际日志路径
steps = []
losses = []

# 你需要自己设置每个 epoch 的 step 数
total_steps_per_epoch = 44160  # 替换成你的实际每个 epoch 的 step 总数

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 匹配 Epoch:[6/6](33000/44160) loss:0.770 ...
        match = re.search(r'Epoch:\[(\d+)/\d+\]\((\d+)/(\d+)\)\s+loss:([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            step_in_epoch = int(match.group(2))
            loss = float(match.group(4))

            global_step = (epoch - 1) * total_steps_per_epoch + step_in_epoch
            steps.append(global_step)
            losses.append(loss)

# 绘制 loss 曲线
plt.figure(figsize=(12, 6))
plt.plot(steps, losses, label='Training Loss', linewidth=1)
plt.xlabel('Global Step')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
