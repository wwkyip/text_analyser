import pandas as pd
from pandasql import sqldf

constructivekeyword = pd.read_csv('datalab_nlp/constructivekeyword.csv')
generalkeyword = pd.read_csv('datalab_nlp/generalkeyword.csv')
sentimentkeyword = pd.read_csv('datalab_nlp/sentimentkeyword.csv')
themekeyword = pd.read_csv('datalab_nlp/themekeyword.csv')
pysqldf = lambda q: sqldf(q, globals())


def query(tablename, columns, whereclause, obj):
    if type(obj) is list:
        typ="list"
    else:
        typ = "dict"

    SQLCommand = ("SELECT " + columns + " from " + tablename + " where " + whereclause + ";");
    data = pysqldf(SQLCommand)
    
    for i, row in data.iterrows():
        if typ == "list":
            obj.append(str(row[0]).replace('\xa0',''))
        else:
            obj[row[0]] = row[1]
    return(obj)



'''
import psycopg2
import psycopg2.extras

CONNECT_STR = "dbname=keywords_db user=postgres password=password"


def query_old(tablename, columns, whereclause, obj):
    if type(obj) is list:
        typ="list"
    else:
        typ = "dict"

    conn = psycopg2.connect(CONNECT_STR)
    cursor = conn.cursor()
    SQLCommand = ("SELECT " + columns + " from " + tablename + " where " + whereclause + ";");
    #print(SQLCommand)
    cursor.execute(SQLCommand)

    row = cursor.fetchone()
    while row:
        if typ == "list":
            obj.append(str(row[0]).replace('\xa0',''))
        else:
            obj[str(row[0])] = row[1]
        row = cursor.fetchone()

    conn.close()
    return(obj)

def save_tables(table_name):
    conn = psycopg2.connect(CONNECT_STR)
    cursor = conn.cursor()
    SQLCommand = "select * from constructivekeyword"    
    cursor.execute(SQLCommand)
    constructivekeyword = pd.DataFrame(cursor.fetchall(), columns = ['id','phrase','score','category'])
    
    SQLCommand = "select * from generalkeyword"    
    cursor.execute(SQLCommand)
    generalkeyword = pd.DataFrame(cursor.fetchall(), columns = ['id','phrase','partofspeech','category'])
    
    SQLCommand = "select * from sentimentkeyword"    
    cursor.execute(SQLCommand)
    sentimentkeyword = pd.DataFrame(cursor.fetchall(), columns = ['id','phrase','polarityn','polarity',
                                                                     'partofspeech','subjecttype','stemmed',
                                                                     'polaritytype','personality','source'])
    
    SQLCommand = "select * from themekeyword"    
    cursor.execute(SQLCommand)
    themekeyword = pd.DataFrame(cursor.fetchall(), columns = ['keyword','class','category','industry'])
    
    constructivekeyword.to_csv('constructivekeyword.csv',index=False)
    generalkeyword.to_csv('generalkeyword.csv',index=False)
    sentimentkeyword.to_csv('sentimentkeyword.csv',index=False)
    themekeyword.to_csv('themekeyword.csv',index=False)
 
'''
    

