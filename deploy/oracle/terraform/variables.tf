variable "region" {
  description = "OCI region, e.g. us-ashburn-1"
  type        = string
}

variable "compartment_id" {
  description = "OCID of the root or target compartment"
  type        = string
}

variable "availability_domain" {
  description = "Availability domain name, e.g. HZnQ:US-ASHBURN-AD-1"
  type        = string
}

variable "ubuntu_arm_image_id" {
  description = "OCID of the Ubuntu 22.04 Arm64 image for your region"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key content (not path)"
  type        = string
  sensitive   = true
}
