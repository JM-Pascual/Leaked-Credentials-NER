provider "aws" {
  region     = "us-east-1"
  access_key = "AKIAIOSFODNN7EXAMPLE"
  secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
}

resource "aws_s3_bucket" "data_lake" {
  bucket = "acme-data-lake-prod"

  tags = {
    Environment = "production"
    Team        = "data-engineering"
  }
}

resource "aws_db_instance" "postgres" {
  identifier        = "prod-db"
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.medium"
  allocated_storage = 100
  db_name           = "appdb"
  username          = "dbadmin"
  password          = "Xq9#mP2vLkR7nYz!"
  skip_final_snapshot = false
}