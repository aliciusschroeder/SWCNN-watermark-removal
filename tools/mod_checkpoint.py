import torch
import torch.nn as nn
import os
import sys

# Make sure to run this script from parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    from models import HN
else:
    from models import HN

if not sys.flags.interactive:
    print("This script is meant to be run in interactive mode (-i).")
    print("Exiting...")
    sys.exit(1)


LR = 0.001

model = nn.DataParallel(HN(), device_ids=[0]).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

cp_path = "output/models/"
cp_path += input(f'Checkpoint path ({cp_path}...): ')
assert cp_path.endswith('.pth'), "Invalid checkpoint file"
cp = torch.load(cp_path)
print("\n"*5)

def save_mod():
    new_path = cp_path.replace('.pth', '_mod.pth')
    torch.save(cp, new_path)

print(f"Loaded checkpoint from {cp_path}")
model.load_state_dict(cp['model_state_dict'])
optimizer.load_state_dict(cp['optimizer_state_dict'])

T_max = cp['scheduler_state_dict']['T_0']
change = input(f"Change T_max from {T_max} to (optional): ")
if change:
    T_max = int(change)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
scheduler.load_state_dict(cp['scheduler_state_dict'])

def simulate_lr():
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    scheduler.load_state_dict(cp['scheduler_state_dict'])
    for i in range(0, T_max+10):
        scheduler.step()
        print(f"Epoch {epoch + i + 1}: lr = {optimizer.state_dict()['param_groups'][0]['lr']}")

epoch = cp['epoch']
global_step = cp['global_step']

print("Accessible variables: \n" +
      "model & cp['model_state_dict']\n" +
      "optimizer & cp['optimizer_state_dict']\n" +
      "scheduler & cp['scheduler_state_dict']\n" +
      "epoch & cp['epoch']\n" +
      "global_step & cp['global_step']\n\n" +
      "Call save_mod() to save the modified checkpoint"+
      "Call simulate_lr() to simulate current LR behaviour\n\n\n")

print(f"Loaded model state at epoch* {epoch} and global step {global_step}")
print("* Epoch is 0-indexed, resuming training will start at epoch+1\n\n")
print(f"=== Scheduler State ===\n{scheduler.state_dict()}\n\n" +
      f"optimizer.state_dict()['param_groups'][0]['lr']: " +
      f"{optimizer.state_dict()['param_groups'][0]['lr']}\n"
      )
