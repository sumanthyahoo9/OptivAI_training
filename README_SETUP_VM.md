# Google Cloud Setup Guide for ML/LLM Development

This guide walks through setting up a Google Cloud environment for machine learning development, from installing the Google Cloud SDK to connecting to a VM using VSCode and inspecting system details.

## Prerequisites

- Google Cloud account with available credits
- VSCode installed on your local machine
- Terminal/Command Prompt access

## 1. Installing Google Cloud SDK

### On macOS

**Option 1: Using Homebrew (Recommended)**

```bash
# Install using Homebrew
brew install --cask google-cloud-sdk

# Verify installation
gcloud --version
```

**Option 2: Manual Installation**

```bash
# Download the installer
curl https://sdk.cloud.google.com > install.sh

# Run the installer
bash install.sh

# Follow the prompts to complete installation
# When asked to add SDK to your PATH, select "Y"

# Restart your terminal or run
source ~/.zshrc  # For Zsh
# OR
source ~/.bash_profile  # For Bash
```

### On Windows

1. Download the installer from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. Run the installer and follow the instructions
3. Restart your command prompt after installation

### On Linux

```bash
# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update and install the SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

## 2. Authenticating with Google Cloud

```bash
# Log in to your Google account
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID.

## 3. Creating a VM Instance

### Via Command Line

```bash
gcloud compute instances create ml-instance \
  --machine-type=e2-medium \
  --zone=europe-west2-b \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=10GB
```

### Via Google Cloud Console

1. Navigate to Compute Engine > VM Instances
2. Click "Create Instance"
3. Configure settings:
   - Name: Choose a name for your instance
   - Region/Zone: Select a region close to you (e.g., europe-west2-b)
   - Machine Type: Select as needed (e.g., e2-medium for development)
   - Boot Disk:
     - OS: Ubuntu or Debian
     - Version: Latest LTS release
     - Size: 10GB or more as needed
   - Firewall: Allow HTTP/HTTPS traffic if needed
4. Click "Create"

## 4. Setting Up SSH Config for Your VMs

```bash
# Configure SSH for all instances in your project
gcloud compute config-ssh
```

This command creates entries in your SSH config file for easy connection.

## 5. Connecting to VM with VSCode

1. Install the "Remote - SSH" extension in VSCode
2. Open the Command Palette (F1 or Ctrl+Shift+P)
3. Type "Remote-SSH: Connect to Host" and select it
4. Choose your VM from the list (format: username@hostname)
5. If prompted, select the SSH configuration file to update
6. Wait for VSCode to connect and set up the remote environment

## 6. Checking System Information

Once connected to your VM via VSCode, run these commands in the terminal to get system details:

```bash
# Check memory/RAM
free -h

# Check disk space
df -h

# Check OS version
lsb_release -a

# Check CPU information
lscpu

# Check if CUDA/GPU is available
nvidia-smi

# Check kernel information
uname -a

# List installed Python packages
apt list --installed | grep python

# Check system load and running processes
top  # Press q to exit

# Check system uptime
uptime

# Check resource limits
ulimit -a
```

## 7. Installing Development Tools on VM

```bash
# Update package lists
sudo apt update

# Install Python and development tools
sudo apt install -y python3-pip python3-venv git

# Create and activate a virtual environment
python3 -m venv ml_env
source ml_env/bin/activate

# Install ML libraries as needed
pip install numpy pandas scikit-learn torch tensorflow
```

## 8. Differences Between Debian and Ubuntu

**Debian**:
- More conservative, stable releases
- Less frequent updates
- Excellent for servers and production systems
- Uses APT package manager

**Ubuntu**:
- Based on Debian
- More frequent releases and updates
- More user-friendly
- Better hardware support for newer devices
- Also uses APT package manager

To use Ubuntu instead of Debian when creating a VM, change the `--image-family` parameter:

```bash
gcloud compute instances create ubuntu-instance \
  --machine-type=e2-medium \
  --zone=europe-west2-b \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=10GB
```

## 9. Stopping/Starting Your VM

```bash
# Stop VM (you won't be charged for the running instance, only storage)
gcloud compute instances stop ml-instance --zone=europe-west2-b

# Start VM again when needed
gcloud compute instances start ml-instance --zone=europe-west2-b
```

## 10. Checking Billing and Credits

1. Go to Google Cloud Console
2. Navigate to Billing
3. Select your billing account
4. Go to "Promotions & Credits" to view available credits
5. Set up budget alerts to monitor usage

Remember to stop your VM instances when not in use to avoid unnecessary charges.