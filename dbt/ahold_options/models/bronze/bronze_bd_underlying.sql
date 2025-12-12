{{ config(materialized='ephemeral') }}

-- Reference to existing bronze_bd_underlying table
SELECT * FROM bronze_bd_underlying
