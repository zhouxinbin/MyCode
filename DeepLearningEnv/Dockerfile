# Dockerfile

# 1. 使用官方的 Miniconda3 镜像作为基础
FROM continuumio/miniconda3:latest

# 2. 设置工作目录
WORKDIR /workspace

# 3. 将 Conda 环境配置文件复制到容器中
COPY environment.yml .

# 4. 使用 environment.yml 创建 Conda 环境
RUN conda env create -f environment.yml

# 5. 将容器的 shell 配置为默认使用新创建的 Conda 环境
SHELL ["conda", "run", "-n", "pytorch_cpu", "/bin/bash", "-c"]

# 6. (可选) 验证环境是否创建成功
RUN echo "Conda environment 'base' for CPU created successfully." && \
    python -c "import torch; print(f'PyTorch version (CPU): {torch.__version__}')"

# 7. 暴露 JupyterLab 的默认端口
EXPOSE 8888

# 8. 容器启动时执行的默认命令
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
