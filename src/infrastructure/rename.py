import os 
  
# Function to rename multiple files 
def rename(srcdir, new_name):
  
    for count, filename in enumerate(os.listdir(srcdir)):
        dst = new_name + '.' + str(count) + ".jpg"
        src = srcdir + filename 
        dst = srcdir + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    srcdir = input("src: ")
    new_name = input("new name: ")
    rename(srcdir, new_name) 