from db.sqlite import SQLiteDB
from typing import List, Tuple, Optional

class EmbeddingDB(SQLiteDB):
    def _create_table(self):
        """
        Creates the 'embeddings' table if it doesn't exist already.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            text TEXT,
            embedding BLOB,
            is_question INTEGER
        )
        """
        self.execute_query(create_table_query)

    def insert_embedding(self, text: str, embedding: bytes, is_question: int) -> int:
        """
        Inserts a new embedding into the database.

        :param text: The text associated with the embedding.
        :param embedding: The embedding as a binary object (BLOB).
        :param is_question: 1 if the text is a question, 0 otherwise.
        :return: The id of the newly inserted row.
        """
        insert_query = """
        INSERT INTO embeddings (text, embedding, is_question)
        VALUES (?, ?, ?)
        """
        cursor = self.execute_query(insert_query, (text, embedding, is_question))
        return cursor.lastrowid

    def get_embedding(self, id: int) -> Optional[Tuple[int, str, bytes, int]]:
        """
        Retrieves an embedding by its ID.

        :param id: The ID of the embedding.
        :return: A tuple containing (id, text, embedding, is_question) if found, otherwise None.
        """
        select_query = "SELECT * FROM embeddings WHERE id = ?"
        return self.fetch_one(select_query, (id,))

    def get_all_embeddings(self) -> List[Tuple[int, str, bytes, int]]:
        """
        Retrieves all embeddings from the database.

        :return: A list of tuples, each containing (id, text, embedding, is_question).
        """
        select_query = "SELECT * FROM embeddings"
        return self.fetch_all(select_query)

    def update_embedding(self, id: int, text: str, embedding: bytes, is_question: int) -> bool:
        """
        Updates an existing embedding in the database.

        :param id: The ID of the embedding to update.
        :param text: The updated text.
        :param embedding: The updated embedding as a binary object (BLOB).
        :param is_question: 1 if the text is a question, 0 otherwise.
        :return: True if the update was successful, False otherwise.
        """
        update_query = """
        UPDATE embeddings
        SET text = ?, embedding = ?, is_question = ?
        WHERE id = ?
        """
        cursor = self.execute_query(update_query, (text, embedding, is_question, id))
        return cursor.rowcount > 0

    def delete_embedding(self, id: int) -> bool:
        """
        Deletes an embedding from the database by ID.

        :param id: The ID of the embedding to delete.
        :return: True if the deletion was successful, False otherwise.
        """
        delete_query = "DELETE FROM embeddings WHERE id = ?"
        cursor = self.execute_query(delete_query, (id,))
        return cursor.rowcount > 0
    
    def delete_all(self) -> bool:
        """
        Deletes an embedding from the database by ID.

        :param id: The ID of the embedding to delete.
        :return: True if the deletion was successful, False otherwise.
        """
        delete_query = "DELETE FROM embeddings"
        cursor = self.execute_query(delete_query)
        return cursor.rowcount > 0


def get_db(filename: str) -> EmbeddingDB:
    """
    Returns an instance of the EmbeddingDB class initialized with the provided filename.

    :param filename: Path to the SQLite database file.
    :return: An EmbeddingDB instance.
    """
    return EmbeddingDB(filename)