import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import collections
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Avoid the red monsters."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated haunted maze. Reach the green exit while avoiding monsters. The maze gets harder each stage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.NUM_STAGES = 3
        self.MAX_MONSTER_TOUCHES = 2

        # Colors
        self.COLOR_BG = (10, 20, 30)
        self.COLOR_WALL_GLOW = (50, 80, 150)
        self.COLOR_WALL = (150, 180, 255)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_EXIT_GLOW = (0, 128, 64)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_MONSTER_GLOW = (150, 0, 0)
        self.COLOR_PARTICLE = (200, 200, 220)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_big = pygame.font.Font(None, 48)

        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.monsters = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.monster_touch_counter = 0
        self.current_stage = 0

        # These will be initialized in _start_new_stage
        self.maze_dim = 0
        self.cell_width = 0
        self.cell_height = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.monster_touch_counter = 0
        self.current_stage = 0
        self.particles = []

        self._start_new_stage()

        return self._get_observation(), self._get_info()

    def _start_new_stage(self):
        """Initializes state for a new maze/stage."""
        # Difficulty scaling
        maze_base_size = 10
        monster_base_speed = 0.5

        self.maze_dim = int(maze_base_size * (1 + self.current_stage * 0.1))
        monster_speed = monster_base_speed + self.current_stage * 0.05

        self.maze = self._generate_maze(self.maze_dim, self.maze_dim)

        # Place player and exit
        self.player_pos = (0, 0)
        self.exit_pos = (self.maze_dim - 1, self.maze_dim - 1)

        # Initialize monsters
        self.monsters = []
        try:
            # This can fail if the maze is too small or disconnected
            random_points = self._get_random_empty_points(4)
            path1 = self._find_path(random_points[0], random_points[1])
            path2 = self._find_path(random_points[2], random_points[3])
            
            if path1:
                self.monsters.append({
                    "pos": random_points[0], "path": path1, "path_idx": 0,
                    "path_dir": 1, "speed": monster_speed, "move_counter": 0.0
                })
            if path2:
                 self.monsters.append({
                    "pos": random_points[2], "path": path2, "path_idx": 0,
                    "path_dir": 1, "speed": monster_speed, "move_counter": 0.0
                })
        except (ValueError, IndexError):
            # Fallback if not enough points are available
            pass


        # Calculate rendering scales
        self.cell_width = (self.WIDTH - 80) / self.maze_dim
        self.cell_height = (self.HEIGHT - 80) / self.maze_dim
        self.grid_offset_x = 40
        self.grid_offset_y = 40

    def _get_random_empty_points(self, n=1):
        """Gets N random valid (non-wall) points in the maze."""
        points = []
        for r, row in enumerate(self.maze):
            for c, cell in enumerate(row):
                # A cell is "empty" if it has at least one opening
                if not (cell['top'] and cell['bottom'] and cell['left'] and cell['right']):
                    if (c, r) != self.player_pos and (c, r) != self.exit_pos:
                        points.append((c, r))
        
        if len(points) < n:
            # Fallback to just using any point if not enough "empty" ones
            all_points = [(c, r) for r in range(self.maze_dim) for c in range(self.maze_dim) if (c,r) != self.player_pos and (c,r) != self.exit_pos]
            if len(all_points) < n:
                return [self.player_pos] * n # Should not happen
            points_indices = self.np_random.choice(len(all_points), size=n, replace=False)
            return [all_points[i] for i in points_indices]

        points_indices = self.np_random.choice(len(points), size=n, replace=False)
        return [points[i] for i in points_indices]

    def _generate_maze(self, width, height):
        """Generates a maze using Randomized DFS. Returns a grid of wall dicts."""
        maze = [[{'top': True, 'bottom': True, 'left': True, 'right': True, 'visited': False} for _ in range(width)] for _ in range(height)]
        stack = [(0, 0)]
        maze[0][0]['visited'] = True

        while stack:
            x, y = stack[-1]
            neighbors = []
            if y > 0 and not maze[y - 1][x]['visited']: neighbors.append(('N', x, y - 1))
            if y < height - 1 and not maze[y + 1][x]['visited']: neighbors.append(('S', x, y + 1))
            if x > 0 and not maze[y][x - 1]['visited']: neighbors.append(('W', x - 1, y))
            if x < width - 1 and not maze[y][x + 1]['visited']: neighbors.append(('E', x + 1, y))

            if neighbors:
                # FIX: np.random.choice on mixed-type lists converts everything to strings.
                # Instead, we select an index randomly and then get the neighbor tuple.
                neighbor_idx = self.np_random.integers(len(neighbors))
                direction, nx, ny = neighbors[neighbor_idx]

                if direction == 'N':
                    maze[y][x]['top'] = False
                    maze[ny][nx]['bottom'] = False
                elif direction == 'S':
                    maze[y][x]['bottom'] = False
                    maze[ny][nx]['top'] = False
                elif direction == 'W':
                    maze[y][x]['left'] = False
                    maze[ny][nx]['right'] = False
                elif direction == 'E':
                    maze[y][x]['right'] = False
                    maze[ny][nx]['left'] = False
                
                maze[ny][nx]['visited'] = True
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _find_path(self, start, end):
        """Finds a path in the maze using BFS. Returns a list of (x, y) tuples."""
        queue = collections.deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == end:
                return path
            
            # Check neighbors
            # Up
            if y > 0 and not self.maze[y][x]['top'] and (x, y-1) not in visited:
                new_path = list(path)
                new_path.append((x, y-1))
                queue.append(new_path)
                visited.add((x, y-1))
            # Down
            if y < self.maze_dim-1 and not self.maze[y][x]['bottom'] and (x, y+1) not in visited:
                new_path = list(path)
                new_path.append((x, y+1))
                queue.append(new_path)
                visited.add((x, y+1))
            # Left
            if x > 0 and not self.maze[y][x]['left'] and (x-1, y) not in visited:
                new_path = list(path)
                new_path.append((x-1, y))
                queue.append(new_path)
                visited.add((x-1, y))
            # Right
            if x < self.maze_dim-1 and not self.maze[y][x]['right'] and (x+1, y) not in visited:
                new_path = list(path)
                new_path.append((x+1, y))
                queue.append(new_path)
                visited.add((x+1, y))
        return [] # Path not found

    def step(self, action):
        movement = action[0]
        reward = -0.1  # Time penalty
        terminated = False
        
        # 1. Update Player Position
        px, py = self.player_pos
        current_cell = self.maze[py][px]
        
        if movement == 1 and py > 0 and not current_cell['top']: self.player_pos = (px, py - 1)
        elif movement == 2 and py < self.maze_dim - 1 and not current_cell['bottom']: self.player_pos = (px, py + 1)
        elif movement == 3 and px > 0 and not current_cell['left']: self.player_pos = (px - 1, py)
        elif movement == 4 and px < self.maze_dim - 1 and not current_cell['right']: self.player_pos = (px + 1, py)

        # 2. Update Monster Positions
        for monster in self.monsters:
            monster['move_counter'] += monster['speed']
            while monster['move_counter'] >= 1.0:
                monster['move_counter'] -= 1.0
                
                next_idx = monster['path_idx'] + monster['path_dir']
                if not (0 <= next_idx < len(monster['path'])):
                    monster['path_dir'] *= -1
                    next_idx = monster['path_idx'] + monster['path_dir']
                
                monster['path_idx'] = next_idx
                monster['pos'] = monster['path'][monster['path_idx']]

        # 3. Update Particles
        # Spawn new particles
        for monster in self.monsters:
            for _ in range(2): # Spawn rate
                self.particles.append(Particle(self._grid_to_pixel(monster['pos']), self.np_random))
        # Update and remove old particles
        self.particles = [p for p in self.particles if p.update()]
        
        # 4. Check for Collisions and Events
        # Player-Monster collision
        for monster in self.monsters:
            if self.player_pos == monster['pos']:
                reward -= 5
                self.monster_touch_counter += 1
                if self.monster_touch_counter >= self.MAX_MONSTER_TOUCHES:
                    terminated = True
                # Reset player to start to avoid repeated hits in one spot
                self.player_pos = (0, 0)
                break

        # Player-Exit collision
        if not terminated and self.player_pos == self.exit_pos:
            reward += 10 * (1 + self.current_stage)
            self.score += 10 * (1 + self.current_stage)
            self.current_stage += 1
            if self.current_stage >= self.NUM_STAGES:
                terminated = True
            else:
                self._start_new_stage()

        # 5. Update Step Counter and Check Termination
        self.steps += 1
        truncated = self.steps >= self.MAX_STEPS

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        gx, gy = grid_pos
        px = self.grid_offset_x + gx * self.cell_width + self.cell_width / 2
        py = self.grid_offset_y + gy * self.cell_height + self.cell_height / 2
        return int(px), int(py)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles (behind everything else)
        for p in self.particles:
            p.draw(self.screen, self.COLOR_PARTICLE)

        # Render maze walls with glow
        for y in range(self.maze_dim):
            for x in range(self.maze_dim):
                px, py = self.grid_offset_x + x * self.cell_width, self.grid_offset_y + y * self.cell_height
                if self.maze[y][x]['top']:
                    pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (px, py), (px + self.cell_width, py), 3)
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.cell_width, py), 1)
                if self.maze[y][x]['bottom']:
                    pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (px, py + self.cell_height), (px + self.cell_width, py + self.cell_height), 3)
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.cell_height), (px + self.cell_width, py + self.cell_height), 1)
                if self.maze[y][x]['left']:
                    pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (px, py), (px, py + self.cell_height), 3)
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.cell_height), 1)
                if self.maze[y][x]['right']:
                    pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (px + self.cell_width, py), (px + self.cell_width, py + self.cell_height), 3)
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.cell_width, py), (px + self.cell_width, py + self.cell_height), 1)

        # Render Exit
        exit_px, exit_py = self._grid_to_pixel(self.exit_pos)
        exit_rad = int(min(self.cell_width, self.cell_height) * 0.4)
        pygame.gfxdraw.filled_circle(self.screen, exit_px, exit_py, exit_rad, self.COLOR_EXIT_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, exit_px, exit_py, int(exit_rad * 0.8), self.COLOR_EXIT)

        # Render Player
        player_px, player_py = self._grid_to_pixel(self.player_pos)
        player_rad = int(min(self.cell_width, self.cell_height) * 0.3)
        pygame.gfxdraw.filled_circle(self.screen, player_px, player_py, player_rad, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_px, player_py, player_rad, self.COLOR_PLAYER)
        
        # Render Monsters
        for monster in self.monsters:
            m_px, m_py = self._grid_to_pixel(monster['pos'])
            flicker = self.np_random.uniform(0.8, 1.2)
            monster_rad = int(min(self.cell_width, self.cell_height) * 0.35 * flicker)
            pygame.gfxdraw.filled_circle(self.screen, m_px, m_py, monster_rad, self.COLOR_MONSTER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, m_px, m_py, int(monster_rad * 0.7), self.COLOR_MONSTER)
            
    def _render_ui(self):
        stage_text = self.font_ui.render(f"Stage: {self.current_stage + 1}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        time_text = self.font_ui.render(f"Time: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        touches_text = self.font_ui.render(f"Health: {self.MAX_MONSTER_TOUCHES - self.monster_touch_counter}", True, self.COLOR_TEXT)
        self.screen.blit(touches_text, (self.WIDTH / 2 - touches_text.get_width() / 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "monster_touches": self.monster_touch_counter,
        }

    def close(self):
        pygame.quit()


class Particle:
    def __init__(self, pos, np_random):
        self.np_random = np_random
        self.x, self.y = pos
        self.vx = self.np_random.uniform(-0.5, 0.5)
        self.vy = self.np_random.uniform(-0.5, 0.5)
        self.lifespan = self.np_random.integers(20, 50)
        self.age = 0
        self.radius = self.np_random.uniform(1, 3)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.age += 1
        return self.age < self.lifespan

    def draw(self, surface, color):
        alpha = max(0, 255 * (1 - self.age / self.lifespan))
        # Create a temporary surface for alpha blending
        radius = int(self.radius)
        if radius <= 0: return
        
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*color, int(alpha)), (radius, radius), radius)
        surface.blit(temp_surf, (int(self.x) - radius, int(self.y) - radius))


# Example usage for testing
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # To run the game manually
    import sys
    pygame.display.set_caption(env.game_description)
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    action = np.array([0, 0, 0]) # Start with no-op
    
    while not done:
        # Manual control
        manual_move = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                manual_move = True
                keys = pygame.key.get_pressed()
                mov = 0
                if keys[pygame.K_UP]: mov = 1
                elif keys[pygame.K_DOWN]: mov = 2
                elif keys[pygame.K_LEFT]: mov = 3
                elif keys[pygame.K_RIGHT]: mov = 4
                action = np.array([mov, 0, 0])

        # Step the environment if a move was made or in auto-advance mode
        if manual_move or env.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
            # Reset action to no-op after a move
            action = np.array([0, 0, 0])

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # A small clock tick prevents 100% CPU usage.
        env.clock.tick(30)

    env.close()
    sys.exit()