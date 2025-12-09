"""
ClickHouse Client - Analytics database interface.

Loads data from MinIO parquet files into ClickHouse for fast analytics queries.
"""

import os
from typing import List, Dict, Any, Optional
import clickhouse_connect
from pathlib import Path


class ClickHouseClient:
    """Client for interacting with ClickHouse analytics database."""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        username: str = None,
        password: str = None
    ):
        """
        Initialize ClickHouse client.
        
        Args:
            host: ClickHouse host (default: from CLICKHOUSE_HOST env var)
            port: ClickHouse HTTP port (default: from CLICKHOUSE_PORT env var)
            database: Database name (default: from CLICKHOUSE_DB env var)
            username: Username (default: from CLICKHOUSE_USER env var)
            password: Password (default: from CLICKHOUSE_PASSWORD env var)
        """
        self.host = host or os.getenv('CLICKHOUSE_HOST', 'clickhouse')
        self.port = port or int(os.getenv('CLICKHOUSE_PORT', '8123'))
        self.database = database or os.getenv('CLICKHOUSE_DB', 'ahold_options')
        self.username = username or os.getenv('CLICKHOUSE_USER', 'default')
        self.password = password or os.getenv('CLICKHOUSE_PASSWORD', 'clickhouse123')
        
        # Create client connection
        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            database=self.database,
            username=self.username,
            password=self.password
        )
    
    def execute(self, query: str) -> Any:
        """
        Execute a query and return results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query results
        """
        return self.client.query(query)
    
    def execute_command(self, command: str) -> None:
        """
        Execute a command (no results expected).
        
        Args:
            command: SQL command to execute
        """
        self.client.command(command)
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists, False otherwise
        """
        query = f"""
            SELECT count() 
            FROM system.tables 
            WHERE database = '{self.database}' 
            AND name = '{table_name}'
        """
        result = self.client.query(query)
        return result.first_row[0] > 0
    
    def create_table_from_s3(
        self,
        table_name: str,
        s3_url: str,
        access_key: str,
        secret_key: str,
        order_by: List[str] = None,
        partition_by: str = None
    ) -> None:
        """
        Create a ClickHouse table and load data from S3/MinIO parquet file.
        
        Args:
            table_name: Name of the table to create
            s3_url: S3 URL to parquet file (e.g., 'http://minio:9000/bucket/file.parquet')
            access_key: S3 access key
            secret_key: S3 secret key
            order_by: List of columns to order by (for performance)
            partition_by: Column to partition by (optional)
        """
        # Drop table if exists
        self.execute_command(f"DROP TABLE IF EXISTS {table_name}")
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            order_clause = f"ORDER BY ({', '.join(order_by)})"
        
        # Build PARTITION BY clause
        partition_clause = ""
        if partition_by:
            partition_clause = f"PARTITION BY {partition_by}"
        
        # Create table with data from S3
        # ClickHouse will infer schema from parquet file
        create_query = f"""
            CREATE TABLE {table_name}
            ENGINE = MergeTree()
            {partition_clause}
            {order_clause}
            AS SELECT * FROM s3(
                '{s3_url}',
                '{access_key}',
                '{secret_key}',
                'Parquet'
            )
        """
        
        self.execute_command(create_query)
    
    def insert_from_s3(
        self,
        table_name: str,
        s3_url: str,
        access_key: str,
        secret_key: str
    ) -> None:
        """
        Insert data into existing ClickHouse table from S3/MinIO parquet file.
        
        Args:
            table_name: Name of the table
            s3_url: S3 URL to parquet file
            access_key: S3 access key
            secret_key: S3 secret key
        """
        insert_query = f"""
            INSERT INTO {table_name}
            SELECT * FROM s3(
                '{s3_url}',
                '{access_key}',
                '{secret_key}',
                'Parquet'
            )
        """
        
        self.execute_command(insert_query)
    
    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows
        """
        result = self.client.query(f"SELECT count() FROM {table_name}")
        return result.first_row[0]
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get table information (schema, size, etc).
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        # Get schema
        schema_query = f"DESCRIBE TABLE {table_name}"
        schema_result = self.client.query(schema_query)
        
        # Get row count
        count = self.get_table_count(table_name)
        
        # Get table size
        size_query = f"""
            SELECT
                formatReadableSize(sum(bytes)) as size,
                sum(rows) as rows
            FROM system.parts
            WHERE database = '{self.database}'
            AND table = '{table_name}'
            AND active
        """
        size_result = self.client.query(size_query)
        
        return {
            'table_name': table_name,
            'columns': [(row[0], row[1]) for row in schema_result.result_rows],
            'row_count': count,
            'size': size_result.first_row[0] if size_result.result_rows else '0 B',
            'rows_in_parts': size_result.first_row[1] if size_result.result_rows else 0
        }
    
    def list_tables(self) -> List[str]:
        """
        List all tables in the database.
        
        Returns:
            List of table names
        """
        query = f"""
            SELECT name 
            FROM system.tables 
            WHERE database = '{self.database}'
            ORDER BY name
        """
        result = self.client.query(query)
        return [row[0] for row in result.result_rows]
    
    def optimize_table(self, table_name: str) -> None:
        """
        Optimize table (merge parts for better query performance).
        
        Args:
            table_name: Name of the table to optimize
        """
        self.execute_command(f"OPTIMIZE TABLE {table_name} FINAL")
    
    def truncate_table(self, table_name: str) -> None:
        """
        Truncate table (remove all data, keep structure).
        
        Args:
            table_name: Name of the table to truncate
        """
        self.execute_command(f"TRUNCATE TABLE {table_name}")
    
    def drop_table(self, table_name: str) -> None:
        """
        Drop table completely.
        
        Args:
            table_name: Name of the table to drop
        """
        self.execute_command(f"DROP TABLE IF EXISTS {table_name}")
    
    def close(self):
        """Close the client connection."""
        self.client.close()


def get_clickhouse_client() -> ClickHouseClient:
    """
    Get a ClickHouse client instance with configuration from environment variables.
    
    Returns:
        ClickHouseClient instance
    """
    return ClickHouseClient()
