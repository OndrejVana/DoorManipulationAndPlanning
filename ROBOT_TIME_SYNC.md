# Robot time sync (chrony)

The robot serves time to the laptop; clock skew between them causes TF and costmap timeouts.

```bash
# STEP 0:
ssh root@tiago-114c

# STEP 1:
sudo apt update
sudo apt install chrony -y

# STEP 2:
sudo nano /etc/chrony/chrony.conf

# STEP 3: (add to chrony.conf)
allow 10.68.0.0/24
local stratum 10

# STEP 4:
sudo systemctl restart chrony
sudo systemctl enable chrony

# STEP 5: (Localy)
sudo chronyc makestep
sudo systemctl restart chrony

# Verify
chronyc sources -v
chronyc tracking
```
