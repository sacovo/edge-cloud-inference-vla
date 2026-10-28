# Common environment for all ecvla Slurm jobs (source it from sbatch scripts).
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_PREFERENCE=only-managed      # venv must not point at a node-local system python
export UV_FROZEN=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/scratch/sandro.covo/hf
export ECVLA_SCRATCH=/scratch/sandro.covo/ecvla
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ECVLA_RUN="uv run --extra libero"
