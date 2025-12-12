-- Cleanup script for old pipeline tables
-- These tables are replaced by the new star schema

-- Old Silver Layer Tables (replaced by dim_underlying, dim_option_contract, fact_option_timeseries)
DROP TABLE IF EXISTS silver_bd_options_enriched CASCADE;
DROP TABLE IF EXISTS silver_options CASCADE;
DROP TABLE IF EXISTS silver_options_chain CASCADE;
DROP TABLE IF EXISTS silver_options_chain_merged CASCADE;
DROP TABLE IF EXISTS silver_underlying_price CASCADE;

-- Show remaining tables
\dt
