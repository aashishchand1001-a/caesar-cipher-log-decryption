# Open the input file and the two output files
infile = open("raw_logs.txt", "r")
master = open("decrypted_master.txt", "w")
alerts = open("security_alerts.txt", "w")
 
for line in infile:
    new_line = ""
 
    for ch in line:
        if ch == " " or ch == "\n":
            # keep spaces and newlines the same
            new_line = new_line + ch
        elif ch.isupper():
            num = ord(ch) - 3
            if num < ord("A"):
                num = num + 26      # wrap around (A goes to X)
            new_line = new_line + chr(num)
        elif ch.islower():
            num = ord(ch) - 3
            if num < ord("a"):
                num = num + 26      # wrap around (a goes to x)
            new_line = new_line + chr(num)
        else:
            # numbers and symbols stay the same
            new_line = new_line + ch
 
    new_line = new_line.strip()
    master.write(new_line + "\n")
 
    if "BREACH" in new_line.upper():
        alerts.write(new_line + "\n")
 
infile.close()
master.close()
alerts.close()
