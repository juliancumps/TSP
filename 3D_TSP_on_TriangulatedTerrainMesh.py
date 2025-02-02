import numpy as np
import random
from scipy.spatial import Delaunay
import pygeodesic.geodesic as geodesic
from noise import pnoise2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TerrainTSPSolver:
    def __init__(self, board_size=20, num_cities=15, num_terrain_points=400):
        self.board_size = board_size
        self.num_cities = num_cities
        self.num_terrain_points = num_terrain_points
        random.seed(random.randint(0, 1000))  # to make the terrain different each time
        self.geodesic_algorithm = None
        self.terrain_tri = None
        self.terrain_points = None
        self.Z = None
        
    def generate_terrain(self, scale = 15.0, octaves = 6, persistence = .5, lacunarity = 2.0):
        """Generate terrain using Perlin noise"""
        width = height = self.board_size
        
        # Create a grid of points
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate terrain heights using Perlin noise
        self.Z = np.zeros_like(X)
        for i in range(width):
            for j in range(height):
                self.Z[i,j] = pnoise2(X[i,j]/scale, Y[i,j]/scale, 
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=42)
        
        # Normalize terrain heights
        self.Z = (self.Z - self.Z.min()) / (self.Z.max() - self.Z.min())
        
        # Create terrain points dictionary
        terrain_points = {}
        k = 0
        step = max(1, width * height // self.num_terrain_points)
        for i in range(0, width, step):
            for j in range(0, height, step):
                if k < self.num_terrain_points:
                    terrain_points[k] = (float(X[i,j]), float(Y[i,j]), float(self.Z[i,j]))
                    k += 1
        
        # Add corner points to ensure proper triangulation
        corners = [
            (0.0, 0.0, 0.0),
            (0.0, float(width-1), 0.0),
            (float(height-1), 0.0, 0.0),
            (float(height-1), float(width-1), 0.0)
        ]
        for i, corner in enumerate(corners):
            terrain_points[self.num_terrain_points + i] = corner
        
        self.terrain_points = terrain_points
        points = np.array(list(terrain_points.values()))
        self.terrain_tri = Delaunay(points[:, :2])
        
        return terrain_points

    def get_height_at_point(self, x, y):
        """Interpolate height at any (x,y) point on the terrain"""
        points = np.array(list(self.terrain_points.values()))
        simplex_index = self.terrain_tri.find_simplex(np.array([x, y]))
        
        if simplex_index == -1:
            return 0  # Point is outside the terrain
            
        # Get the vertices of the triangle containing the point
        triangle_vertices = points[self.terrain_tri.simplices[simplex_index]]
        
        # Calculate barycentric coordinates
        def barycentric_coords(p, a, b, c):
            v0 = b - a
            v1 = c - a
            v2 = p - a
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            return u, v, w

        p = np.array([x, y])
        a = triangle_vertices[0][:2]
        b = triangle_vertices[1][:2]
        c = triangle_vertices[2][:2]
        
        u, v, w = barycentric_coords(p, a, b, c)
        
        # Interpolate z coordinate
        z = (u * triangle_vertices[0][2] + 
             v * triangle_vertices[1][2] + 
             w * triangle_vertices[2][2])
        
        return z
    
    def generate_cities(self):
        """Generate random city locations and project onto terrain"""
        cities = {}
        for i in range(self.num_cities):
            x = round(random.uniform(3, self.board_size - 2), 2)
            y = round(random.uniform(3, self.board_size - 2), 2)
            z = self.get_height_at_point(x, y)
            cities[i] = (x, y, z)
        return cities

    def initialize_geodesic_algorithm(self, terrain_points, cities):
        """Initialize the geodesic algorithm with terrain and city vertices"""
        # Combine terrain points and cities into one vertex list
        combined_points = {}
        for i, coords in terrain_points.items():
            combined_points[i] = coords
        
        terrain_len = len(terrain_points)
        for i, coords in cities.items():
            combined_points[i + terrain_len] = coords
            
        # Create vertices and faces for triangulation
        vertices = np.array(list(combined_points.values()), dtype=np.float64)
        xy_vertices = vertices[:, :2]
        
        # Create Delaunay triangulation
        tri = Delaunay(xy_vertices)
        faces = np.array([tuple(triangle) for triangle in tri.simplices], dtype=np.int32)
        
        # Initialize the geodesic algorithm
        self.geodesic_algorithm = geodesic.PyGeodesicAlgorithmExact(vertices, faces)
        self.terrain_offset = terrain_len

    def distance(self, city1, city2):
        """Calculate geodesic distance between two cities"""
        if self.geodesic_algorithm is None:
            raise ValueError("Geodesic algorithm not initialized")
            
        distance, path = self.geodesic_algorithm.geodesicDistance(
            city2 + self.terrain_offset,
            city1 + self.terrain_offset
        )
        return distance, path
    
    def total_distance(self, tour):
        """Calculate total tour distance and get complete path"""
        total = 0
        total_path = np.empty((0, 3))
        
        # Calculate distances between consecutive cities
        for i in range(len(tour) - 1):
            d, path = self.distance(tour[i], tour[i + 1])
            total += d
            total_path = np.concatenate((total_path, path), axis=0)
            
        # Add return to start
        d, path = self.distance(tour[-1], tour[0])
        total += d
        total_path = np.concatenate((total_path, path), axis=0)
        
        return total, total_path
    
    def simulated_annealing(self, cities, initial_solution, temperature=1000, cooling_rate=0.99):
        """Optimize tour using simulated annealing"""
        current_solution = initial_solution
        current_cost, path_to = self.total_distance(current_solution)
        best_solution = current_solution
        best_path = path_to
        best_cost = current_cost
        
        while temperature > 0.1:
            # Generate neighbor solution by reversing a random segment
            neighbor = current_solution[:]
            i, j = sorted(random.sample(range(len(neighbor)), 2))
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
            
            # Calculate cost difference
            neighbor_cost, neighbor_path = self.total_distance(neighbor)
            cost_diff = neighbor_cost - current_cost
            
            # Accept better solutions or sometimes accept worse ones based on temperature
            if cost_diff < 0 or random.random() < np.exp(-cost_diff/temperature):
                current_solution = neighbor
                current_cost = neighbor_cost
                if neighbor_cost < best_cost:
                    best_solution = neighbor
                    best_cost = neighbor_cost
                    best_path = neighbor_path
                    
            temperature *= cooling_rate
            
        return best_solution, best_cost, best_path
    
    def plot_solution(self, terrain_points, cities, optimal_tour, optimal_path):
        """Visualize the terrain, cities, and optimal tour"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert terrain points to arrays for triangulation
        points = np.array(list(terrain_points.values()))
        xy_points = points[:, :2]
        
        # Create Delaunay triangulation
        tri = Delaunay(xy_points)
        
        # Plot terrain surface
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                       triangles=tri.simplices,
                       cmap='terrain',
                       alpha=0.6)
        
        # Plot cities
        for _, (x, y, z) in cities.items():
            ax.scatter(x, y, z, color='black', marker='s', s=100)
            
        # Plot optimal tour
        x = [p[0] for p in optimal_path]
        y = [p[1] for p in optimal_path]
        z = [p[2] for p in optimal_path]
        ax.plot(x, y, z, color='blue', linewidth=2, label='Optimal Tour')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D TSP Solution over Terrain')
        plt.show()

#################################################################################

# Usage:
temperature = 1000

cooling_rate = .999

solver = TerrainTSPSolver()
terrain = solver.generate_terrain()
cities = solver.generate_cities()

# Initialize the geodesic algorithm before calculating distances
solver.initialize_geodesic_algorithm(terrain, cities)

initial_tour = list(cities.keys())
random.shuffle(initial_tour)
print('Calculating...')
optimal_tour, distance, path = solver.simulated_annealing(cities, initial_tour, temperature, cooling_rate)
solver.plot_solution(terrain, cities, optimal_tour, path)