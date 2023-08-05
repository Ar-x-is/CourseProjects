import numpy as np
import time

def calculate_mean_square_diff(emp_arm_means, p):
    num_arms = len(emp_arm_means)
    diff = np.zeros((num_arms, num_arms))
    for i in range(num_arms):
        diff[i] = (emp_arm_means[i] - p)**2

    return diff


def calculate_lines_needed(diff):
    num = 0
    rows = []
    cols = []
    while(np.count_nonzero(np.delete(np.delete(diff, rows, 0), cols, 1)==0) > 0):
        print(rows, cols)
        if len(diff) == 1:
            num += 1
            return num, rows, cols

        max_idx = [0,0] # [idx, row=0/col=1]
        # find the row/col with the maximum number of zeros
        max = 0
        for i in range(np.shape(diff)[0]):
            if i in rows:
                continue
            covered_zeros = np.sum([item in cols for item in np.where(diff[i,:] == 0)[0]])
            num_zeros = np.count_nonzero(diff[i,:] == 0)
            if np.count_nonzero(diff[i,:] == 0) > max and covered_zeros < num_zeros:
                max_idx = [i, 0]
                max = np.count_nonzero(diff[i,:] == 0)

        for i in range(np.shape(diff)[1]):
            if i in cols:
                continue
            covered_zeros = np.sum([item in rows for item in np.where(diff[:,i] == 0)[0]])
            num_zeros = np.count_nonzero(diff[:,i] == 0)
            if np.count_nonzero(diff[:,i] == 0) > max and covered_zeros < num_zeros:
                max_idx = [i, 1]
                max = np.count_nonzero(diff[:,i] == 0)

        # note down the indices of the covered row/col
        if max_idx[1] == 0:
            rows.append(max_idx[0])
        else:
            cols.append(max_idx[0])

        num += 1

    return num, rows, cols


def find_optimal_zeros(diff, depth=0, cols=np.array([])):
    print(depth, cols)
    # it is guaranteed to find a solution
    # there exist n zeros in the nxn matrix such that no two zeros are in the same row or column
    indices = np.where(diff[depth,:] == 0)[0]
    if np.all([i in cols for i in indices]):
        print("all cols occupied")
        return np.delete(cols, -1)

    num_arms = len(diff)
    for i in indices:
        if i in cols:
            continue

        cols = np.append(cols, i)

        if depth == num_arms-1:
            return cols

        cols = find_optimal_zeros(diff, depth+1, cols)

    return cols

def minimise_loss(emp_arm_means, p):
    num_arms = len(emp_arm_means)

    diff = calculate_mean_square_diff(emp_arm_means, p)
    
    # subtract the minimum value of each row from the row
    for i in range(num_arms):
        diff[i,:] = diff[i,:] - np.min(diff[i,:])

    # subtract the minimum value of each column from the column
    for i in range(num_arms):
        diff[:,i] = diff[:,i] - np.min(diff[:,i])

    lines_needed = 0
    while True:
        lines_needed, covered_rows, covered_cols = calculate_lines_needed(diff)
        if lines_needed == num_arms:
            print(lines_needed, covered_rows, covered_cols)
            break
        uncovered_rows = [i for i in range(num_arms) if i not in covered_rows]
        m = np.min(np.delete(np.delete(diff, covered_rows, 0), covered_cols, 1))
        diff[uncovered_rows,:] = diff[uncovered_rows,:] - m
        m = np.min(diff)
        diff[:,covered_cols] = diff[:,covered_cols] - m
        time.sleep(1)

    print(diff)
    print(find_optimal_zeros(diff))

    return 0

p = np.array([0.16, 0.22, 0.37, 0.50, 0.88])
emp_arm_means = np.array([0.11, 0.43, 0.63, 0.6, 0.44])

minimise_loss(emp_arm_means, p)