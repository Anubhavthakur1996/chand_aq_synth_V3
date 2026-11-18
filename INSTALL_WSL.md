# Installing on WSL2 when the C: drive is nearly full

**Why installs fail:** WSL2 keeps the whole Linux disk in one file (`ext4.vhdx`) on C:.  Inside Linux, `df -h`
shows the file's *maximum* size (e.g. 1 TB), NOT the real free space on C:.  If C: runs out of room during a big
install (torch + CUDA libraries are several GB), the write fails (`Errno 28: No space left on device`) and can leave
truncated files behind, which later show up as `Bus error` or `file too short` when importing numpy/torch.

## 1. Clean up and reinstall
```
deactivate
rm -rf env env_broken          # frees space inside the virtual disk for reuse
pip cache purge
python3 -m venv env && source env/bin/activate
python -m pip install --upgrade pip
mkdir -p /mnt/s/pip_tmp        # only moves downloads/temp files off C:; installed files still use C:
TMPDIR=/mnt/s/pip_tmp pip install --no-cache-dir -r requirements.txt
TMPDIR=/mnt/s/pip_tmp pip install --no-cache-dir --no-deps gretel-synthetics==0.20.0
rm -rf /mnt/s/pip_tmp
```
Keep roughly 8-10 GB free on C: while installing (rough estimate).

## 2. Check every package loads
```
for m in numpy pandas scipy sklearn matplotlib torch gretel_synthetics; do
  python -c "import $m; print('$m OK')" || echo "$m FAILED"
done
python -c "from gretel_synthetics.timeseries_dgan.dgan import DGAN; print('DGAN OK')"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

## 3. Permanent fix: move the WSL disk to another local drive (e.g. S:)
Back up your projects first.  In PowerShell:
```
wsl --shutdown
wsl -l -v                                   # note the distro name
wsl --manage <DistroName> --move S:\WSL\<DistroName>      # recent WSL versions
```
(older WSL: `wsl --export` -> `wsl --unregister` -> `wsl --import`, then set the default user in /etc/wsl.conf)

## 4. Smaller alternative: CPU-only torch
Hundreds of MB instead of several GB.  The GAN here is small, so time a 200-epoch run on CPU before deciding you need the GPU.

## Other tips for an 8 GB RAM machine
If you ever see `Killed`, WSL ran out of memory: lower `RF_TREES` or `N_SYNTH_SEQUENCES` in `config.py`, or raise the WSL memory
limit in `%UserProfile%\.wslconfig` (`[wsl2]` / `memory=6GB`).
