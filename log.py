import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt",mode = "w")
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info("output my log")

#use print to file
f = open("./logg.txt",'a')
print("output my log use print function",file=f)
