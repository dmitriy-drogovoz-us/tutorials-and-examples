# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

resource "google_artifact_registry_repository" "image_repo" {
  location      = var.image_repository_location
  repository_id = var.image_repository_name
  description   = "Image repository MLFlow tutorial"
  format        = "DOCKER"
}

resource "google_artifact_registry_repository_iam_binding" "binding" {
  project    = var.project_id
  location   = "us"
  repository = var.image_repository_name
  role       = "roles/artifactregistry.reader"
  members = [
    "serviceAccount:${module.infra[0].service_account}"
  ]
  depends_on = [google_artifact_registry_repository.image_repo, module.infra]
}
