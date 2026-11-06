#!/bin/bash
set -e

# Create dedicated users
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER agent_user WITH PASSWORD '$AGENT_DB_PASS';
    CREATE USER nango_user WITH PASSWORD '$NANGO_DB_PASS';

    CREATE DATABASE agent_db OWNER agent_user;
    CREATE DATABASE nango_db OWNER nango_user;
EOSQL

# Setup agent_db with pgvector extension and permissions
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "agent_db" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    GRANT ALL PRIVILEGES ON DATABASE agent_db TO agent_user;
    GRANT ALL PRIVILEGES ON SCHEMA public TO agent_user;
EOSQL

# Setup nango_db permissions
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "nango_db" <<-EOSQL
    GRANT ALL PRIVILEGES ON DATABASE nango_db TO nango_user;
    GRANT ALL PRIVILEGES ON SCHEMA public TO nango_user;
EOSQL

# Physical Security Boundary: Revoke all access to nango_db from agent_user (Invariant 3)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "nango_db" <<-EOSQL
    REVOKE CONNECT ON DATABASE nango_db FROM agent_user;
    REVOKE ALL ON SCHEMA public FROM agent_user;
EOSQL
