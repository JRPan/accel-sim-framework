import sqlite3

# read only
table = sqlite3.connect('file:' + '/home/tgrogers-raid/a/pan251/scratch-link/pan251/test_pytorch_tencore/net.sqlite' + "?mode=ro", uri=True)

rows = table.execute('SELECT id, value FROM StringIDs').fetchall()

# create table from ids
StringIDs_table = {}
for row in rows:
    StringIDs_table[row[0]] = row[1]

rows = table.execute('SELECT start, end, text FROM NVTX_EVENTs').fetchall()
print(rows)