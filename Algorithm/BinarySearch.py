def binary_search(arr, val):
    s_arr = sorted(arr)
    low = 0
    high = len(arr) - 1
    while(low<=high):
        mid = (low+high)//2
        if s_arr[mid] > val:
            high = mid - 1
        elif s_arr[mid] < val:
            low = mid + 1
        else:
            print("found")
            return s_arr[mid]
    print("non-exist")
