import concurrent.futures

def try_my_operation(item):
    try:
        return sum(range(item))
    except:
        print('error with item')

items = [1,2,3,4] 
executor = concurrent.futures.ProcessPoolExecutor(10)
futures = [executor.submit(try_my_operation, item) for item in items]
concurrent.futures.wait(futures)

for f in concurrent.futures.as_completed(futures):
        print(f.result())