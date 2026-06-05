# Oracle Cloud Always Free — Deployment Guide

## Target architecture

- **Instance**: VM.Standard.A1.Flex (Ampere Arm64, Always Free)
- **OS**: Ubuntu 22.04 LTS (Arm64)
- **App**: Docker container running FastAPI + ONNX Runtime
- **Reverse proxy**: Nginx (optional)

---

## Step 1 — Provision the VM

### Option A — Oracle Console (manual)

1. Log in to [cloud.oracle.com](https://cloud.oracle.com)
2. **Compute → Instances → Create Instance**
3. Change shape: **VM.Standard.A1.Flex** — 2 OCPU, 12 GB RAM
4. Image: **Canonical Ubuntu 22.04** (Arm64)
5. Networking: new VCN + public subnet
6. Add your SSH public key
7. Click **Create**

### Option B — Terraform (preferred)

```bash
cd deploy/oracle/terraform
cp terraform.tfvars.example terraform.tfvars   # fill in your values
terraform init
terraform plan
terraform apply
```

Fill in `terraform.tfvars`:
```hcl
region              = "us-ashburn-1"
compartment_id      = "ocid1.compartment.oc1..."
availability_domain = "HZnQ:US-ASHBURN-AD-1"
ubuntu_arm_image_id = "ocid1.image.oc1..."   # find in Console → Images
ssh_public_key      = "ssh-ed25519 AAAA..."
```

---

## Step 2 — Open firewall ports (Security List)

In the OCI Console → VCN → Security Lists, add **Ingress** rules:

| Protocol | Port | Source    | Purpose        |
|----------|------|-----------|----------------|
| TCP      |  22  | 0.0.0.0/0 | SSH            |
| TCP      |  80  | 0.0.0.0/0 | HTTP           |
| TCP      | 443  | 0.0.0.0/0 | HTTPS          |
| TCP      | 8000 | 0.0.0.0/0 | Direct access  |

Also open the **OS firewall**:
```bash
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 80   -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

---

## Step 3 — Bootstrap the VM

```bash
# From your local machine — copy project archive
tar -czf /tmp/pcb-inspector.tar.gz -C /path/to/pcb-defect-yolo26 .
scp /tmp/pcb-inspector.tar.gz ubuntu@<VM_IP>:/tmp/

# SSH in and run bootstrap
ssh ubuntu@<VM_IP>
sudo REPO_URL="" bash /tmp/deploy/oracle/install_vm.sh
```

---

## Step 4 — Transfer trained model weights

```bash
# Copy best ONNX model to the VM
scp runs/train/yolo26n_pcb/weights/best.onnx ubuntu@<VM_IP>:/opt/pcb-inspector/runs/train/yolo26n_pcb/weights/
# or PyTorch fallback
scp runs/train/yolo26n_pcb/weights/best.pt ubuntu@<VM_IP>:/opt/pcb-inspector/runs/train/yolo26n_pcb/weights/
```

---

## Step 5 — Configure environment

```bash
cd /opt/pcb-inspector
cp .env.example .env
nano .env   # set ONNX_MODEL_PATH, CONF_THRESHOLD, etc.
```

---

## Step 6 — Start the service

```bash
cd /opt/pcb-inspector
docker compose up -d
docker compose logs -f    # verify it started
```

### Auto-start on reboot (systemd)

```bash
sudo cp deploy/oracle/systemd.service /etc/systemd/system/pcb-inspector.service
sudo systemctl daemon-reload
sudo systemctl enable pcb-inspector
sudo systemctl start pcb-inspector
```

---

## Step 7 — (Optional) Nginx reverse proxy

```bash
sudo apt-get install -y nginx
sudo cp deploy/oracle/nginx.conf /etc/nginx/sites-available/pcb-inspector
sudo ln -s /etc/nginx/sites-available/pcb-inspector /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## Step 8 — (Optional) HTTPS with Certbot

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.example.com
```

---

## Verify deployment

```bash
curl http://<VM_IP>:8000/health
# Expected: {"status":"ok","model_loaded":true,...}
```

Open `http://<VM_IP>:8000` in your browser to use the web UI.

---

## Resource usage (baseline, no GPU)

| Component | Typical value |
|-----------|---------------|
| Idle RAM  | ~250 MB       |
| Under load RAM | ~600 MB  |
| CPU (inference) | 1-2 cores |
| Inference latency (ONNX, 640px) | 500-1500 ms |

Within the Always Free A1 (2 OCPU / 12 GB) limits.
