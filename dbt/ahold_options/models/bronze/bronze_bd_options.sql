{{ config(materialized='ephemeral') }}

-- Reference to existing bronze_bd_options table
SELECT * FROM bronze_bd_options
