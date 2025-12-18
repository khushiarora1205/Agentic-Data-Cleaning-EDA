import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DB_USER = "apple"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "demoDb"
NEW_TABLE_NAME = "messy_employees" # Name of the new table
CSV_FILE = "/Users/apple/Documents/AgenticAI_EDA/data/messy_data.csv"        # The file you uploaded

def create_and_load_table():
    try:
        # 1. Read the CSV
        # We read everything as object (string) initially to handle the messiness
        df = pd.read_csv(CSV_FILE)
        
        # Replace NaN with None so Psycopg2 handles them as SQL NULLs
        df = df.replace({np.nan: None})
        
        print(f"📂 Read {len(df)} rows from '{CSV_FILE}'")

        # 2. Connect to Database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cur = conn.cursor()
        print(f"🔌 Connected to '{DB_NAME}'")

        # 3. Create the New Table
        # Note: We use TEXT for Salary and Join_Date to allow '60k' and mixed dates
        create_table_query = f"""
        DROP TABLE IF EXISTS {NEW_TABLE_NAME};
        CREATE TABLE {NEW_TABLE_NAME} (
            id INTEGER,
            name TEXT,
            age REAL, 
            city TEXT,
            salary TEXT,
            join_date TEXT
        );
        """
        cur.execute(create_table_query)
        print(f"🔨 Created table '{NEW_TABLE_NAME}'")

        # 4. Insert Data
        insert_query = f"""
        INSERT INTO {NEW_TABLE_NAME} (id, name, age, city, salary, join_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        # Convert DataFrame to list of tuples for insertion
        data_values = [tuple(x) for x in df.to_numpy()]
        
        cur.executemany(insert_query, data_values)
        print(f"✅ Successfully inserted {len(data_values)} rows into '{NEW_TABLE_NAME}'")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_and_load_table()