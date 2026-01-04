
from pathlib import Path
import re


def _patch_deepspeed_usage(resemble_dir):
    mock_code = """
# PATCHED V2: DeepSpeed removed for inference-only usage
import types
class MockDeepSpeed:
    def init_distributed(self, *args, **kwargs): pass
    class accelerator:
        def get_accelerator(self): return self
        def communication_backend_name(self): return "nccl"
    def zero_optim_partition(self, *args, **kwargs): return lambda x: x

deepspeed = MockDeepSpeed()
get_accelerator = lambda: MockDeepSpeed().accelerator()
DeepSpeedConfig = lambda x: None
class DeepSpeedEngine: pass
# End Patch
"""
    patched_count = 0
    for filepath in resemble_dir.rglob("*.py"):
        try:
            content = filepath.read_text(encoding="utf-8")
            triggers = ["import deepspeed", "from deepspeed"]
            if not any(t in content for t in triggers):
                continue
            if "# PATCHED V2" in content:
                continue

            print(f" -> Patching {filepath.name} for DeepSpeed...")
            lines = content.splitlines()
            new_lines = []
            injected = False
            for line in lines:
                if "import deepspeed" in line or "from deepspeed" in line:
                    new_lines.append(f"# {line} # PATCHED")
                    if not injected:
                        new_lines.append(mock_code)
                        injected = True
                else:
                    new_lines.append(line)
            filepath.write_text("\n".join(new_lines), encoding="utf-8")
            patched_count += 1
        except Exception as e:
            print(f" -> Failed to patch {filepath}: {e}")

    if patched_count == 0:
        print(" -> DeepSpeed patches already applied or not needed.")


def _patch_torchaudio_loader(resemble_dir):
    monkeypatch_code = """
import soundfile
import torch
import torchaudio
def custom_load(filepath, **kwargs):
    w, sr = soundfile.read(filepath)
    # soundfile returns [time, channels] or [time]
    t = torch.from_numpy(w).float()
    if t.ndim == 1: t = t.unsqueeze(1) # [time, 1]
    # Check if we need to transpose: soundfile is (frames, channels),
    # torchaudio expects (channels, frames)
    # Heuristic: Audio is usually longer than channels.
    if t.ndim == 2 and t.shape[0] > 100 and t.shape[1] < 20:
         t = t.permute(1, 0)
    elif t.ndim == 2 and t.shape[1] > 100 and t.shape[0] < 20:
         pass # Already channels first?
    else:
         # Default mapping for soundfile read is (samples, channels) -> need (channels, samples)
         if t.shape[0] > t.shape[1]:
            t = t.permute(1, 0)

    return t, sr

def custom_save(filepath, src, sample_rate, **kwargs):
    # src is [channels, time] or [1, channels, time]
    if src.ndim == 3: src = src.squeeze(0)
    # torchaudio expects [channels, time], soundfile expects [time, channels]
    if src.ndim == 2 and src.shape[0] < 20 and src.shape[1] > 100:
        src = src.permute(1, 0)

    src = src.detach().cpu().numpy() # [time, channels]
    soundfile.write(filepath, src, sample_rate)

torchaudio.load = custom_load
torchaudio.save = custom_save
"""
    main_py = resemble_dir / "enhancer/__main__.py"
    if main_py.exists():
        content = main_py.read_text(encoding="utf-8")
        if "custom_load" in content:
            print(f" -> {main_py.name} already monkeypatched (Torchaudio).")
        else:
            print(f" -> Monkeypatching {main_py.name} for Torchaudio...")
            if "import torchaudio" in content:
                content = content.replace(
                    "import torchaudio",
                    "import torchaudio\n" + monkeypatch_code
                )
                main_py.write_text(content, encoding="utf-8")
            else:
                print(f" -> Could not find 'import torchaudio' in {main_py.name}")


def patch_resemble_enhance():
    print("[Patch] Checking Resemble-Enhance (DeepSpeed Removal)...")
    venv_site = Path("venv/Lib/site-packages")
    if not venv_site.exists():
        print(" -> venv site-packages not found. Skipping.")
        return

    resemble_dir = venv_site / "resemble_enhance"
    if not resemble_dir.exists():
        print(" -> resemble_enhance package not found.")
        return

    _patch_deepspeed_usage(resemble_dir)
    _patch_torchaudio_loader(resemble_dir)


def patch_resemble_cli_args():
    print("[Patch] Fix Resemble-Enhance CLI Arguments...")
    venv_site = Path("venv/Lib/site-packages")
    resemble_main = venv_site / "resemble_enhance/enhancer/__main__.py"

    if not resemble_main.exists():
        print(f" -> {resemble_main} not found. Skipping.")
        return

    try:
        content = resemble_main.read_text(encoding='utf-8')

        # Check if already patched
        if "chunk_seconds=" in content or "chunk_seconds =" in content:
            if "chunk_seconds=10" in content:
                print(" -> Upgrading patch from 10s to 25s...")
                content = content.replace("chunk_seconds=10", "chunk_seconds=25")
                content = content.replace("chunks_overlap=1", "chunks_overlap=2")
                resemble_main.write_text(content, encoding='utf-8')
                print(" -> Patch upgraded.")
                return
            else:
                print(" -> CLI arguments already patched.")
                return

        # Regex to match the enhance() call with arguments
        pattern = (r"(hwav,\s*sr\s*=\s*enhance\()([\s\S]*?)"
                   r"(lambd=args.lambd,)([\s\S]*?)(\))")

        match = re.search(pattern, content)
        if match:
            full_match = match.group(0)
            if "chunk_seconds" in full_match:
                print(" -> CLI arguments already patched (late check).")
                return

            print(" -> Patching CLI arguments with improved logic...")
            last_paren_idx = full_match.rfind(')')
            call_content = full_match[:last_paren_idx].rstrip()

            if not call_content.endswith(','):
                call_content += ","

            new_args = "\n                chunk_seconds=25,\n                chunks_overlap=2,"
            new_call = f"{call_content}{new_args}\n            )"

            content = content.replace(full_match, new_call)
            resemble_main.write_text(content, encoding='utf-8')
            print(" -> Successfully patched CLI arguments.")
        else:
            print(" -> Could not find 'enhance(...)' call pattern to patch.")

    except Exception as e:
        print(f" -> Failed to patch CLI args: {e}")


def _inject_soundfile_aggressive_patch(sf_py, new_lines):
    # We need to find the def line index in new_lines and insert after it.
    final_lines = []
    injected_now = False
    for line in new_lines:
        final_lines.append(line)
        if not injected_now and "bitrate_mode=None):" in line:
            indent = "    "
            injection = (
                f"{indent}if subtype in ['PCM_16', 'PCM_24'] or "
                "subtype is None: subtype = 'FLOAT'  # Forced 32-bit"
            )
            final_lines.append(injection)
            injected_now = True
            print(" -> Injected aggressive 32-bit override.")

    sf_py.write_text("\n".join(final_lines), encoding="utf-8")


def _scan_and_update_soundfile_lines(lines):
    """Scans soundfile lines and updates the default subtype."""
    new_lines = []
    patched = False
    aggressive_patched = False

    for line in lines:
        if "def write(file, data, samplerate," in line and "subtype=" in line:
            if "subtype='FLOAT'" not in line:
                line = line.replace("subtype=None", "subtype='FLOAT'")
                print(" -> Updated default subtype signature to FLOAT.")
            new_lines.append(line)
            patched = True
        else:
            new_lines.append(line)

        if "Forced 32-bit" in line:
            aggressive_patched = True

    return new_lines, patched, aggressive_patched


def patch_soundfile_32bit_default():
    print("[Patch] Enforcing 32-bit Float Audio Globally...")
    venv_site = Path("venv/Lib/site-packages")
    sf_py = venv_site / "soundfile.py"

    if not sf_py.exists():
        print(" -> soundfile.py not found. Skipping.")
        return

    try:
        lines = sf_py.read_text(encoding="utf-8").splitlines()
        new_lines, patched, aggressive_patched = _scan_and_update_soundfile_lines(lines)

        if patched and not aggressive_patched:
            _inject_soundfile_aggressive_patch(sf_py, new_lines)
        elif patched and aggressive_patched:
            print(" -> Aggressive 32-bit override already present.")
        elif not patched:
            print(" -> Could not find 'def write' line to patch.")

    except Exception as e:
        print(f" -> Failed to patch soundfile: {e}")


def patch_common_separator_force_soundfile():
    print("[Patch] Forcing Audio-Separator to use 'soundfile' (and ignoring pydub)...")
    venv_site = Path("venv/Lib/site-packages")
    sep_py = venv_site / "audio_separator/separator/common_separator.py"

    if not sep_py.exists():
        print(" -> common_separator.py not found. Skipping.")
        return

    try:
        content = sep_py.read_text(encoding="utf-8")
        target_line = 'self.use_soundfile = config.get("use_soundfile")'
        replacement = 'self.use_soundfile = True # Forced by apply_patches.py'

        if replacement in content:
            print(" -> usage of soundfile already forced.")
        elif target_line in content:
            new_content = content.replace(target_line, replacement)
            sep_py.write_text(new_content, encoding="utf-8")
            print(" -> Successfully forced 'use_soundfile = True'.")
        else:
            print(" -> Could not find target line to force soundfile usage.")

    except Exception as e:
        print(f" -> Failed to patch common_separator: {e}")


if __name__ == "__main__":
    patch_resemble_enhance()
    patch_resemble_cli_args()
    patch_soundfile_32bit_default()
    patch_common_separator_force_soundfile()
    print("Optimization patches applied.")
