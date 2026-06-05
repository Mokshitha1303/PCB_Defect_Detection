output "instance_public_ip" {
  description = "Public IP of the PCB Inspector VM"
  value       = oci_core_instance.pcb_vm.public_ip
}

output "app_url" {
  description = "Application URL"
  value       = "http://${oci_core_instance.pcb_vm.public_ip}:8000"
}
