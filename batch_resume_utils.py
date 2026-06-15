import csv
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_NUM_EPOCHS = 100
CSV_LOG_NAMES = ["train_log.csv", "val_log.csv"]
TXT_LOG_NAMES = ["train_log.txt", "val_log.txt"]


def parse_num_epochs(extra_args, default=DEFAULT_NUM_EPOCHS):
    num_epochs = default
    for idx, arg in enumerate(extra_args):
        if arg == "--num_epochs" and idx + 1 < len(extra_args):
            num_epochs = int(extra_args[idx + 1])
        elif arg.startswith("--num_epochs="):
            num_epochs = int(arg.split("=", 1)[1])
    return num_epochs


def remove_option(extra_args, option_name):
    cleaned = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg == option_name:
            skip_next = True
            continue
        if arg.startswith(option_name + "="):
            continue
        cleaned.append(arg)
    return cleaned


def latest_checkpoint(output_dir):
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None, None

    latest_epoch = None
    latest_path = None
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    for path in checkpoint_dir.glob("checkpoint_epoch_*.pth"):
        match = pattern.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if latest_epoch is None or epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_epoch, latest_path


def last_logged_epoch(output_dir, phase="val"):
    log_path = Path(output_dir) / "logs" / f"{phase}_log.csv"
    if not log_path.exists():
        return None

    last_epoch = None
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            epoch_text = row[0].strip().lstrip("\ufeff")
            if not epoch_text:
                continue
            try:
                last_epoch = int(epoch_text)
            except ValueError:
                continue
    return last_epoch


def experiment_complete(output_dir, target_epochs):
    target_last_epoch = target_epochs - 1
    val_last_epoch = last_logged_epoch(output_dir, phase="val")
    if val_last_epoch is not None and val_last_epoch >= target_last_epoch:
        return True

    checkpoint_epoch, _ = latest_checkpoint(output_dir)
    return checkpoint_epoch is not None and checkpoint_epoch >= target_last_epoch


def backup_logs(output_dir):
    log_dir = Path(output_dir) / "logs"
    if not log_dir.exists():
        return None

    backup_root = Path(output_dir) / ".resume_log_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / time.strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=False)

    copied_any = False
    for name in CSV_LOG_NAMES + TXT_LOG_NAMES:
        src = log_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
            copied_any = True

    return backup_dir if copied_any else None


def read_csv_rows(path):
    if not path.exists():
        return None, {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return None, {}
    header = rows[0]
    epoch_rows = {}
    for row in rows[1:]:
        if not row:
            continue
        try:
            epoch = int(row[0])
        except ValueError:
            continue
        epoch_rows[epoch] = row
    return header, epoch_rows


def merge_csv_log(old_path, new_path, dest_path):
    old_header, old_rows = read_csv_rows(old_path)
    new_header, new_rows = read_csv_rows(new_path)
    header = new_header or old_header
    if header is None:
        return

    merged = {}
    merged.update(old_rows)
    merged.update(new_rows)

    with open(dest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for epoch in sorted(merged):
            writer.writerow(merged[epoch])


def split_txt_log(path):
    if not path.exists():
        return [], {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = []
    epoch_lines = {}
    epoch_pattern = re.compile(r"^\s*(\d+)\s+")
    for line in lines:
        match = epoch_pattern.match(line)
        if match:
            epoch_lines[int(match.group(1))] = line
        else:
            header.append(line)
    return header, epoch_lines


def merge_txt_log(old_path, new_path, dest_path):
    old_header, old_rows = split_txt_log(old_path)
    new_header, new_rows = split_txt_log(new_path)
    header = old_header or new_header

    merged = {}
    merged.update(old_rows)
    merged.update(new_rows)

    with open(dest_path, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for epoch in sorted(merged):
            f.write(merged[epoch] + "\n")


def merge_logs_from_backup(output_dir, backup_dir):
    if backup_dir is None:
        return

    log_dir = Path(output_dir) / "logs"
    if not log_dir.exists():
        return

    for name in CSV_LOG_NAMES:
        merge_csv_log(backup_dir / name, log_dir / name, log_dir / name)
    for name in TXT_LOG_NAMES:
        merge_txt_log(backup_dir / name, log_dir / name, log_dir / name)


def run_experiment_with_resume(
    *,
    python_executable,
    train_script,
    fixed_args,
    output_dir,
    passthrough_args,
    dry_run=False,
    force_rerun=False,
):
    target_epochs = parse_num_epochs(passthrough_args)
    passthrough_args = remove_option(passthrough_args, "--resume")

    output_dir = Path(output_dir)
    latest_epoch, checkpoint_path = latest_checkpoint(output_dir)

    if not force_rerun and experiment_complete(output_dir, target_epochs):
        print(f"[SKIP] Completed: {output_dir} (target epochs: {target_epochs})")
        return

    resume_args = []
    backup_dir = None
    if not force_rerun and checkpoint_path is not None and latest_epoch is not None:
        if latest_epoch < target_epochs - 1:
            resume_args = ["--resume", str(checkpoint_path)]
            print(f"[RESUME] {output_dir}: latest checkpoint epoch {latest_epoch}, target {target_epochs - 1}")
        else:
            print(f"[SKIP] Checkpoint already reaches target: {output_dir}")
            return
    elif output_dir.exists() and not force_rerun:
        partial_epoch = last_logged_epoch(output_dir, phase="val")
        if partial_epoch is not None:
            print(
                f"[WARN] Logs exist for {output_dir}, but no checkpoint was found. "
                "Cannot resume safely, so this experiment is skipped."
            )
            return

    cmd = [
        python_executable,
        str(train_script),
        *fixed_args,
        "--output_dir",
        str(output_dir),
        *resume_args,
        *passthrough_args,
    ]

    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return

    if resume_args:
        backup_dir = backup_logs(output_dir)

    try:
        subprocess.run(cmd, check=True)
    finally:
        merge_logs_from_backup(output_dir, backup_dir)
