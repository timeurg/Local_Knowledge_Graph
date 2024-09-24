import sqlite3
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

# Abstract class that handles general SQLite logic
class SQLiteDB(ABC):
    def __init__(self, db_file: str):
        """
        Initializes the SQLiteDB object by creating a connection to the SQLite database.

        :param db_file: Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_file)
        self._create_table()

    @abstractmethod
    def _create_table(self):
        """
        Abstract method for creating a specific table. Must be implemented in subclass.
        """
        pass
    
    def execute_query(self, query: str, params: Tuple = ()) -> sqlite3.Cursor:
        return self.execute(query, params)
    
    def execute(self, query: str, params: Tuple = ()) -> sqlite3.Cursor:
        """
        Executes a given SQL query with optional parameters and returns a cursor.

        :param query: The SQL query to execute.
        :param params: Optional parameters to bind to the query.
        :return: sqlite3.Cursor object after executing the query.
        """
        cursor = self.conn.execute(query, params)
        self.conn.commit()
        return cursor

    def fetch_one(self, query: str, params: Tuple = ()) -> Optional[Tuple]:
        """
        Executes a query and fetches one result.

        :param query: The SQL query to execute.
        :param params: Optional parameters to bind to the query.
        :return: A tuple containing the result if found, otherwise None.
        """
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()

    def fetch_all(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """
        Executes a query and fetches all results.

        :param query: The SQL query to execute.
        :param params: Optional parameters to bind to the query.
        :return: A list of tuples containing the results.
        """
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()

    def close(self):
        """
        Closes the connection to the database.
        """
        self.conn.close()