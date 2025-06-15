import csv

def read_data(filename):
    with open(filename, 'r') as file:       # Open the file for reading and assign it to the variable file.
        csv_reader = csv.reader(file)       # This reads each row in the csv file as a list of values
        headers = next(csv_reader)  # Skip header
        data = [row for row in csv_reader]
    return data

def find_s_algorithm(data):
    # Initialize hypothesis with most specific hypothesis (i.e., first positive example)
    for row in data:
        if row[-1].lower() == 'yes':
            hypothesis = row[:-1]           #List Slicing: start from index 0 up to but not including index -1 (which is the last element)
            break
    else:
        return None  # No positive example found

    # Iterate through the dataset and update hypothesis
    for row in data:
        if row[-1].lower() == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] != row[i]:
                    hypothesis[i] = '?'  # Generalize

    return hypothesis

def main():
    filename = 'training_data.csv'  # Update with your filename
    data = read_data(filename)
    print(data)
    final_hypothesis = find_s_algorithm(data)

    if final_hypothesis:
        print("Most specific hypothesis found by FIND-S:")
        print(final_hypothesis)
    else:
        print("No positive examples found in the dataset.")

if __name__ == "__main__":
    main()
