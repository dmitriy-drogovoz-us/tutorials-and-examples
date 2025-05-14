output "project_id" {
  value = var.project_id
}
output "gke_cluster_name" {
  value       = local.cluster_name
  description = "GKE cluster name"
}

output "gke_cluster_location" {
  value       = var.cluster_location
  description = "GKE cluster location"
}

output "cloudsql_instance_ip" {
  value = module.cloudsql.private_ip_address
}


output "cloudsql_database_user" {
  value = var.cloudsql_database_user
}


output "cloudsql_database_secret_name" {
  value = var.cloudsql_database_secret_name
}

output "cloudsql_database_name" {
  value = local.database_name
}
