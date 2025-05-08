output "project_id" {
  value       = var.project_id
}
output "gke_cluster_name" {
  value       = local.cluster_name
  description = "GKE cluster name"
}

output "gke_cluster_location" {
  value       = var.cluster_location
  description = "GKE cluster location"
}

output "image_repository_name" {
  value = local.image_repository_name
}
output "image_repository_location" {
  value = var.image_repository_location
}


output "image_repository_full_name" {
  value = "${var.image_repository_location}-docker.pkg.dev/${var.project_id}/${local.image_repository_name}"
}
