terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.0"
    }
  }
}

provider "oci" {
  region = var.region
}

# ── VCN ───────────────────────────────────────────────────────────────────────
resource "oci_core_vcn" "pcb_vcn" {
  compartment_id = var.compartment_id
  display_name   = "pcb-inspector-vcn"
  cidr_block     = "10.0.0.0/16"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.pcb_vcn.id
  display_name   = "pcb-inspector-igw"
  enabled        = true
}

resource "oci_core_route_table" "rt" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.pcb_vcn.id
  display_name   = "pcb-inspector-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_subnet" "pub_subnet" {
  compartment_id    = var.compartment_id
  vcn_id            = oci_core_vcn.pcb_vcn.id
  display_name      = "pcb-inspector-subnet"
  cidr_block        = "10.0.1.0/24"
  route_table_id    = oci_core_route_table.rt.id
  security_list_ids = [oci_core_security_list.sl.id]
}

# ── Security list ─────────────────────────────────────────────────────────────
resource "oci_core_security_list" "sl" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.pcb_vcn.id
  display_name   = "pcb-inspector-sl"

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }

  ingress_security_rules {
    protocol = "6"  # TCP
    source   = "0.0.0.0/0"
    tcp_options { min = 22;   max = 22   }
  }
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options { min = 80;   max = 80   }
  }
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options { min = 443;  max = 443  }
  }
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options { min = 8000; max = 8000 }
  }
}

# ── Compute instance (Always Free A1 Arm) ─────────────────────────────────────
resource "oci_core_instance" "pcb_vm" {
  compartment_id      = var.compartment_id
  availability_domain = var.availability_domain
  shape               = "VM.Standard.A1.Flex"
  display_name        = "pcb-inspector"

  shape_config {
    ocpus         = 2
    memory_in_gbs = 12
  }

  source_details {
    source_type = "image"
    source_id   = var.ubuntu_arm_image_id
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.pub_subnet.id
    assign_public_ip = true
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data = base64encode(file("${path.module}/../install_vm.sh"))
  }
}
