import csv

def filter_csv_by_cities(input_file, output_file, target_cities):
    """
    Extract rows from a CSV file where both City1 and City2 columns are in the target_cities list,
    and write them to a new CSV file.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    :param target_cities: List of target city names.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # Create a list to store filtered rows
        filtered_rows = []
        
        for row in reader:
            if row['city1'] in target_cities and row['city2'] in target_cities:
                filtered_rows.append(row)

        # Write the filtered rows to a new CSV file
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

# Example usage
target_cities = ["New York City, NY (Metropolitan Area)", "San Antonio, TX", "Seattle, WA"]
filter_csv_by_cities('data/flight_routes.csv', 'data/reduced.csv', target_cities)
