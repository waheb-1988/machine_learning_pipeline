# https://www.youtube.com/watch?v=x7SwgcpACng&list=PLB5jA40tNf3tRMbTpBA0N7lfDZNLZAa9G
import pyodbc
import pandas as pd
import mysql.connector
#data = pd.read_csv("C:/Abdelouaheb/perso/Ph/machine_learning_pipeline/data/diabetes/diabetes.csv")


# CSV file path
#csv_path = "C:/Abdelouaheb/perso/Ph/machine_learning_pipeline/data/diabetes/diabetes.csv"

# Trusted Connection to Named Instance


# Connect to the server
cnx = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="0000")
    #database="sakila"  # Make sure to specify the database
print('hhhhh')
# # Get a cursor
cur = cnx.cursor()
print(cur)
# # Execute a query
cur.execute("SELECT * FROM sakila.actor LIMIT 5;")

# # Fetch the results
rows = cur.fetchall()
print(rows)
# # Check if rows are returned
# if rows:
#     print("Results fetched:")
#     for row in rows:
#         print(row)
# else:
#     print("No results found.")

# # Close connection
# cur.close()
# cnx.close()
