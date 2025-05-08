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
