import pandas as pd
from bioneuralnet.analysis.dynamic_visualization import DynamicVisualizer

def main():
    adjacency_matrix = pd.read_csv('input/adjacency_matrix.csv', index_col=0)
    
    dynamic_vis = DynamicVisualizer(
        adjacency_matrix=adjacency_matrix,
        layout='spring', 
        notebook=False,  
        bgcolor='#ffffff',
        font_color='black',
        output_dir='visualizations/dynamic',
        output_filename='dynamic_network.html',
        width="100%",
        height="800px"
    )
    
    G = dynamic_vis.generate_graph()
    dynamic_vis.visualize(G)

if __name__ == "__main__":
    main()
