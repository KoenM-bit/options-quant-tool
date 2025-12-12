{{ config(materialized='ephemeral') }}

-- Reference to existing bronze_fd_options table
SELECT * FROM bronze_fd_options
