ssh amax连接服务器。服务器有8张卡，查看显存来确定跑哪里。最好4~8个任务一起跑。一个任务跑完另一个接替。将本地的工作区代码上传至服务器，保证本地的代码，git最新的，服务器的一模一样。服务器文件夹和本地对齐。
要设定一个15分钟的定时查看结果跑到哪儿了。一定要限制进程为下面的2
export LD_LIBRARY_PATH=/home/shanjunjie/miniconda3/envs/paper1/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

评估是fp32的评估50帧活体和4场景。qat4是10次mc的10帧活体和4场景
