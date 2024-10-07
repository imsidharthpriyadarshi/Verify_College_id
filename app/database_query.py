import psycopg
import os
import shutil


#folder="/home/sidharth/Documents/verify_id/app/data/students_id"
folder="/home/sidharth/Documents/verify_id/app/data/college_id"

with psycopg.connect("dbname=temp user=postgres password=postgres") as conn:
    with conn.cursor() as cur:
        for i in os.listdir(folder):
            cur.execute("Select * from students where photo_filename=%s",[i], binary=True)
            result=cur.fetchone()
            if result is not None:
                current=str(result[9]).lower()
                if current is not None:
                    if current=="frontback id card.":
                        src=os.path.join(folder,i)
                        dst="/home/sidharth/Documents/verify_id/app/data/front_back_id/"
                        dst= os.path.join(dst, i)
                        shutil.move(src, dst)
                        print(result[0])
           
            


            