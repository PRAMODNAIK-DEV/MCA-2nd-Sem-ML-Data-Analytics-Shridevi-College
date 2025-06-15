import csv

def read_data(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        data = [row for row in csv_reader]
    return data

def more_general(h1, h2):
    return all(x == '?' or x == y for x, y in zip(h1, h2))

def min_generalizations(h, x):
    new_h = list(h)
    for i in range(len(h)):
        if h[i] != x[i]:
            new_h[i] = '?' if h[i] != '∅' else x[i]
    return new_h

def min_specializations(h, domains, x):
    specializations = []
    for i in range(len(h)):
        if h[i] == '?':
            for val in domains[i]:
                if x[i] != val:
                    new_h = h.copy()
                    new_h[i] = val
                    specializations.append(new_h)
        elif h[i] != '∅':
            new_h = h.copy()
            new_h[i] = '∅'
            specializations.append(new_h)
    return specializations

def candidate_elimination(data):
    n_features = len(data[0]) - 1
    domains = [set() for _ in range(n_features)]

    for row in data:
        for i in range(n_features):
            domains[i].add(row[i])

    S = ['∅'] * n_features
    G = [['?'] * n_features]

    for row in data:
        instance, label = row[:-1], row[-1].lower()

        if label == 'yes':
            G = [g for g in G if more_general(g, instance)]
            S = min_generalizations(S, instance)
            G = [g for g in G if not more_general(S, g)]

        elif label == 'no':
            if more_general(S, instance):
                S = ['∅'] * n_features

            new_G = []
            for g in G:
                if more_general(g, instance):
                    new_G += min_specializations(g, domains, instance)
                else:
                    new_G.append(g)

            G = [g for g in new_G if more_general(g, S)]

    return S, G

def main():
    filename = 'training_data.csv'
    data = read_data(filename)

    print("\nTraining Data:")
    for row in data:
        print(row)

    S, G = candidate_elimination(data)

    print("\nFinal Specific Hypothesis (S):")
    print(S)

    print("\nFinal General Hypotheses (G):")
    for g in G:
        print(g)

if __name__ == "__main__":
    main()
