import csv
import os
import sklearn
import networkx as nx
import pandas as pd

class GraphSetup:
    def __init__(self, data_path):
        df = pd.read_csv(data_path)

        df_filtered = df.drop(df.columns[[0, 3, 4, 20, 21]], axis=1)

        # Year, Quarter, City1, City2, Airport1Id, Airport2Id, Airport1, Airport2, Miles, Passenger, Fare,carrier_lg,large_ms,fare_lg,carrier_low,lf_ms,fare_low
        self.columns = df_filtered.columns

        airport1 = df_filtered.iloc[:, 4].unique()
        airport2 = df_filtered.iloc[:, 5].unique()
        self.nodes = list(set(airport1) | set(airport2))

        airport_attributes = {}

        for _, row in df_filtered.iterrows():
            airport1id = row['airportid_1']
            airport2id = row['airportid_2']
            airport1 = row['airport_1']
            airport2 = row['airport_2']
            city1 = row['city1']
            city2 = row['city2']

            if airport1id not in airport_attributes:
                airport_attributes[airport1id] = {'airport': airport1, 'city': city1}
            
            if airport2id not in airport_attributes:
                airport_attributes[airport2id] = {'airport': airport2, 'city': city2}

        # set the edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(airport_attributes.keys())
        nx.set_node_attributes(self.graph, airport_attributes)
        
        self.train_set, self.test_set = sklearn.model_selection.train_test_split(df_filtered, test_size=0.2)
    
    def train():
        self.edge_labels = {}

        # load in the edges
        for _, row in self.train_set.iterrows():
            airport1id = row['airportid_1']
            airport2id = row['airportid_2']
            miles = row['miles']
            passengers = row['passengers']
            carrier_lg = row['carrier_lg']
            large_ms = row['large_ms']
            fare_lg = row['fare_lg']
            carrier_low = row['carrier_low']
            lf_ms = row['lf_ms']
            fare_low = row['fare_low']
            label = row['tbl1apk']

            self.graph.add_edge(airport1id, 
                                airport2id, 
                                miles=miles, 
                                passengers=passengers,
                                carrier_lg=carrier_lg, 
                                large_ms=large_ms, 
                                fare_lg=fare_lg, 
                                carrier_low=carrier_low, 
                                lf_ms=lf_ms, 
                                fare_low=fare_low,
                                label=label)
            
            self.edge_labels[(airport1id, airport2id, label)] = fare
            
        
    def evaluate(self, test_set):
        pass

g = GraphSetup("data.csv")