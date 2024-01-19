import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import hmean
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.metrics.pairwise import cosine_similarity


class Plotter():
    """
    A utility class for visualizing vector distances and relationships.

    Methods:
    - points(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        Visualizes vectors in a 2D space.

    - euclid_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        Visualizes the Euclidean distances between vectors and annotates the plot
        with dashed lines representing the distances.

    - dotprod_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        Visualizes the dot product distances and projections between vectors.
        Annotates the plot with dashed lines representing the projection of one
        vector onto the other.

    - cosine_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        Visualizes the cosine similarity between vectors. Annotates the plot with
        arcs representing the angles between vectors and displays cosine similarity
        values.

    Note: The vectors are assumed to be numpy arrays with shape (2,).

    Parameters:
    - vectors (list[np.ndarray]): List of 2D arrays representing vectors.
    - colormap (str): Colormap name for visualization. Default is 'Paired'.

    Example Usage:
    ```
    vectors = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    plotter = Plotter()
    plotter.euclid_dist(vectors)
    ```

    """
    @staticmethod
    def points(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        """
        Visualizes vectors in a 2D space.

        Parameters:
        - vectors (list[np.ndarray]): List of 2D arrays representing vectors.
        - colormap (str): Colormap name for visualization. Default is 'Paired'.

        Returns:
        None
        """
        # Check if the list is empty
        if not vectors:
            raise ValueError("The list of vectors is empty")

        # Check if all arrays are two-dimensional
        if any((not isinstance(vector, np.ndarray)) or (vector.shape != (2,)) for vector in vectors):
            raise TypeError("One vector in the list is not a numpy array with shape (2,)")

        num_vectors: int = len(vectors)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set up a colormap
        cmap = plt.get_cmap(colormap)

        # Plot the vectors
        for i, vector in enumerate(vectors):
            color = cmap(i)
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color, label=f'Vector {i}: {vectors[i]}')

        # Set axis limits
        max_coordinate = max(np.abs(vectors).max() for vectors in vectors)
        min_coordinate = max(1, min(np.abs(vectors).min() for vectors in vectors))
        ax.set_xlim([min_coordinate - 1, max_coordinate + 1])
        ax.set_ylim([min_coordinate - 1, max_coordinate + 1])

        # Set title
        ax.set_title('Vectors')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.gca().set_aspect("equal")
        plt.show()

    @staticmethod
    def euclid_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        """
        Visualizes the Euclidean distances between vectors and annotates the plot
        with dashed lines representing the distances.

        Parameters:
        - vectors (list[np.ndarray]): List of 2D arrays representing vectors.
        - colormap (str): Colormap name for visualization. Default is 'Paired'.

        Returns:
        None
        """
        # Check if the list is empty
        if not vectors:
            raise ValueError("The list of vectors is empty")

        # Check if all arrays are two-dimensional
        if any((not isinstance(vector, np.ndarray)) or (vector.shape != (2,)) for vector in vectors):
            raise TypeError("One vector in the list is not a numpy array with shape (2,)")
        
        # Check that the list contains at least 2 vectors
        num_vectors = len(vectors)
        if num_vectors < 2:
            raise ValueError("The list must contain at least 2 vectors")

        # Calculate distances
        distances = []
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                distance_ij = np.linalg.norm(vectors[i] - vectors[j])
                distances.append((i, j, distance_ij))

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set up a colormap
        cmap = plt.get_cmap(colormap)

        # Plot the vectors
        for i, vector in enumerate(vectors):
            color = cmap(i)
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color, label=f'Vector {i}: {vectors[i]}')

        # Plot lines representing distances
        for i, j, distance_ij in distances:
            color = cmap(i+3)
            midpoint = (vectors[i] + vectors[j]) / 2
            ax.annotate(f'{distance_ij:.3f}', xy=midpoint, xytext=(10, -15), textcoords='offset points')

            ax.plot([vectors[i][0], vectors[j][0]], [vectors[i][1], vectors[j][1]], 'k--')

        # Set axis limits
        max_coordinate = max(np.abs(vectors).max() for vectors in vectors)
        min_coordinate = max(1, min(np.abs(vectors).min() for vectors in vectors))
        ax.set_xlim([min_coordinate - 1, max_coordinate + 1])
        ax.set_ylim([min_coordinate - 1, max_coordinate + 1])

        # Set title
        ax.set_title('Euclidean Distance')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.gca().set_aspect("equal")
        plt.show()
    
    @staticmethod
    def dotprod_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        """
        Visualizes the dot product distances and projections between vectors.
        Annotates the plot with dashed lines representing the projection of one
        vector onto the other.

        Parameters:
        - vectors (list[np.ndarray]): List of 2D arrays representing vectors.
        - colormap (str): Colormap name for visualization. Default is 'Paired'.

        Returns:
        None
        """
        # Check if the list is empty
        if not vectors:
            raise ValueError("The list of vectors is empty")

        # Check if all arrays are two-dimensional
        if any((not isinstance(vector, np.ndarray)) or (vector.shape != (2,)) for vector in vectors):
            raise TypeError("One vector in the list is not a numpy array with shape (2,)")
        
        # Check that the list contains at least 2 vectors
        num_vectors = len(vectors)
        if num_vectors < 2:
            raise ValueError("The list must contain at least 2 vectors")

        # Calculate distances
        distances = []
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                distance_ij = np.linalg.norm(vectors[i] - vectors[j])
                distances.append((i, j, distance_ij))

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set up a colormap
        cmap = plt.get_cmap(colormap)

        # Plot the vectors
        for i, vector in enumerate(vectors):
            color = cmap(i)
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color, label=f'Vector {i}: {vectors[i]}')
        
        # Calculate dot product distance and projection for each pair of vectors
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                color = cmap(i+j+3)
                dot_product = np.dot(vectors[i], vectors[j])
                norm_i = np.linalg.norm(vectors[i])
                norm_j = np.linalg.norm(vectors[j])

                # Calculate projection of the shorter vector onto the longer one
                if norm_i >= norm_j:
                    projection = (dot_product / norm_i**2) * vectors[i]
                    long_vec = vectors[i]
                    short_vec = vectors[j]
                else:
                    projection = (dot_product / norm_j**2) * vectors[j]
                    long_vec = vectors[i]
                    short_vec = vectors[i]

                # Plot the projections
                ax.quiver(0, 0, projection[0], projection[1], angles='xy', scale_units='xy', scale=1, color=color)

                # Plot the projection line as a dashed line perpendicular to the longer vector
                projection_line = Line2D([projection[0], short_vec[0]], [projection[1], short_vec[1]],
                                        linestyle='--', color=color, linewidth=1)
                ax.add_line(projection_line)

                midpoint = (projection+long_vec) / 2
                ax.annotate(f'Dot product similarity: {dot_product:.3f}', xy=midpoint, xytext=(5, -5), color='black', fontsize=8, textcoords='offset points')

        # Set axis limits
        max_coordinate = max(np.abs(vectors).max() for vectors in vectors)
        min_coordinate = max(1, min(np.abs(vectors).min() for vectors in vectors))
        ax.set_xlim([min_coordinate - 1, max_coordinate + 1])
        ax.set_ylim([min_coordinate - 1, max_coordinate + 1])

        # Set title
        ax.set_title('Dot Product Distance and Projection')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.gca().set_aspect("equal")
        plt.show()
    
    @staticmethod
    def cosine_dist(vectors: list[np.ndarray], colormap: str = 'Paired') -> None:
        """
        Visualizes the cosine similarity between vectors. Annotates the plot with
        arcs representing the angles between vectors and displays cosine similarity
        values.

        Parameters:
        - vectors (list[np.ndarray]): List of 2D arrays representing vectors.
        - colormap (str): Colormap name for visualization. Default is 'Paired'.

        Returns:
        None
        """
        # Check if the list is empty
        if not vectors:
            raise ValueError("The list of vectors is empty")

        # Check if all arrays are two-dimensional
        if any((not isinstance(vector, np.ndarray)) or (vector.shape != (2,)) for vector in vectors):
            raise TypeError("One vector in the list is not a numpy array with shape (2,)")
        
        # Check that the list contains at least 2 vectors
        num_vectors = len(vectors)
        if num_vectors < 2:
            raise ValueError("The list must contain at least 2 vectors")

        # Calculate cosine similarities
        similarities = np.zeros((num_vectors, num_vectors))
        for i in range(num_vectors):
            for j in range(num_vectors):
                similarities[i, j] = cosine_similarity([vectors[i]], [vectors[j]])[0, 0]

        # Calculate absolute angles
        theta = []
        for i in range(num_vectors):
            theta.append(math.degrees(math.acos(vectors[i][0] / norm(vectors[i]))))

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set up a colormap
        cmap = plt.get_cmap(colormap)

        # Plot the vectors
        for i, vector in enumerate(vectors):
            color = cmap(i)
            ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color, label=f'Vector {i}: {vectors[i]}')

        # Display cosine similarities and angles
        for i in range(num_vectors):
            color = cmap(i+3)
            for j in range(i+1, num_vectors):
                if i != j:
                    origin = [0, 0]
                    arc_radius = hmean([norm(vectors[i]), norm(vectors[j])]) / 2
                    mid_point = [arc_radius*math.cos(math.radians((theta[i] + theta[j])/2))*1.4,
                                 arc_radius*math.sin(math.radians((theta[i] + theta[j])/2))*1.4]
                
                    arc = mpatches.Arc(origin, arc_radius*2, arc_radius*2, theta1=min(theta[i], theta[j]), theta2=max(theta[i], theta[j]), color=color, alpha=0.5)
                    ax.add_patch(arc)

                    ax.text(mid_point[0], mid_point[1], f'Cosine similarity: {similarities[i, j]:.3f}\nAngle: {abs(theta[i] - theta[j]):.3f}°',
                            ha='center', va='center', fontsize=8, color='black')
                    # Plot lines representing distances

        # Set axis limits
        max_coordinate = max(np.abs(vectors).max() for vectors in vectors)
        min_coordinate = max(1, min(np.abs(vectors).min() for vectors in vectors))
        ax.set_xlim([min_coordinate - 1, max_coordinate + 1])
        ax.set_ylim([min_coordinate - 1, max_coordinate + 1])

        # Set title
        ax.set_title('Cosine Similarity')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.gca().set_aspect("equal")
        plt.show()

def visualize_embeddings(words: list[str], embeds: pd.DataFrame, query_array: np.array = None) -> None:
    words_df = embeds.loc[words].reset_index(drop=True)
    print(words_df)

    if query_array is not None:
        words.append("<QUERY>")
        
        query_df = pd.DataFrame(query_array, columns=['UMAP1', 'UMAP2'])
        words_df = pd.concat([words_df, query_df], ignore_index=True)
        print(words_df)


    # Set up Seaborn with a color palette
    sns.set(style="whitegrid", palette="tab10")

    # Create a scatter plot with Seaborn
    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x='UMAP1', y='UMAP2', data=words_df, palette="tab10", alpha=0.7, legend='full')
    ax.set(xlabel=None)
    ax.set(ylabel=None)

    for n in range(len(words)):
        plt.text(words_df.UMAP1[n], words_df.UMAP2[n], words[n], horizontalalignment='left', size='medium', color='black', weight='semibold')
    
    # Add labels and legend
    plt.title('Word Embeddings - 2D-Representation')
    plt.show()

if __name__=="__main__":
    pass