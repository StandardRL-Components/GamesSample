import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect red fruits and avoid purple obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a maze to collect fruits while dodging moving obstacles. Get bonus points for risky collections near enemies!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CELL_SIZE = 20
    MAZE_WIDTH = SCREEN_WIDTH // CELL_SIZE
    MAZE_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

    MAX_STEPS = 1000
    WIN_CONDITION_FRUITS = 25
    NUM_OBSTACLES = 3

    # Colors
    COLOR_BG = (20, 30, 40)  # Dark blue-grey
    COLOR_WALL = (40, 60, 80)  # Darker blue
    COLOR_PLAYER = (255, 200, 0)  # Bright Yellow
    COLOR_PLAYER_OUTLINE = (255, 255, 255)
    COLOR_FRUIT = (220, 30, 30)  # Bright Red
    COLOR_OBSTACLE = (160, 30, 220)  # Bright Purple
    COLOR_TEXT = (240, 240, 240)
    COLOR_BONUS_TEXT = (50, 255, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.fruit_pos = None
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False

        self.np_random = None # Will be seeded in reset()

    def _generate_maze(self):
        # Grid of cells, each with [top, right, bottom, left] walls
        maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH, 4), dtype=bool)
        visited = np.zeros((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=bool)
        stack = []

        # Start DFS from a random cell
        start_x, start_y = self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT)
        visited[start_y, start_x] = True
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            # Check neighbors
            # Up
            if cy > 0 and not visited[cy - 1, cx]: neighbors.append((cx, cy - 1, 0, 2))
            # Right
            if cx < self.MAZE_WIDTH - 1 and not visited[cy, cx + 1]: neighbors.append((cx + 1, cy, 1, 3))
            # Down
            if cy < self.MAZE_HEIGHT - 1 and not visited[cy + 1, cx]: neighbors.append((cx, cy + 1, 2, 0))
            # Left
            if cx > 0 and not visited[cy, cx - 1]: neighbors.append((cx - 1, cy, 3, 1))

            if neighbors:
                nx, ny, wall_dir, neighbor_wall_dir = neighbors[self.np_random.integers(len(neighbors))]

                # Knock down walls
                maze[cy, cx, wall_dir] = False
                maze[ny, nx, neighbor_wall_dir] = False

                visited[ny, nx] = True
                stack.append((nx, ny))
            else:
                stack.pop()

        self.maze = maze

    def _find_valid_spawn_location(self, occupied_pos):
        while True:
            pos = (self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT))
            if pos not in occupied_pos:
                return pos

    def _find_obstacle_path(self, min_len=5):
        # Find horizontal corridors
        h_corridors = []
        for r in range(1, self.MAZE_HEIGHT - 1):
            for c in range(1, self.MAZE_WIDTH - min_len):
                if not self.maze[r, c, 3]:  # No left wall
                    path = []
                    for i in range(min_len):
                        path.append((c + i, r))
                        if self.maze[r, c + i, 1]:  # Right wall found
                            path = []
                            break
                    if path:
                        h_corridors.append(path)

        # Find vertical corridors
        v_corridors = []
        for c in range(1, self.MAZE_WIDTH - 1):
            for r in range(1, self.MAZE_HEIGHT - min_len):
                if not self.maze[r, c, 0]:  # No top wall
                    path = []
                    for i in range(min_len):
                        path.append((c, r + i))
                        if self.maze[r + i, c, 2]:  # Bottom wall found
                            path = []
                            break
                    if path:
                        v_corridors.append(path)

        all_corridors = h_corridors + v_corridors
        if not all_corridors:  # Fallback if no long corridors found
            return [(1, 1), (1, 2), (1, 3)]

        # FIX: Use indexing instead of np.random.choice to preserve tuple types
        return all_corridors[self.np_random.integers(len(all_corridors))]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        self.particles = []

        self._generate_maze()

        occupied_positions = set()

        # Place player
        self.player_pos = self._find_valid_spawn_location(occupied_positions)
        occupied_positions.add(self.player_pos)

        # Place obstacles
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            path = self._find_obstacle_path()
            if not path: continue
            start_pos = path[0]
            obstacle = {
                "pos": start_pos,
                "path": path,
                "path_idx": 0,
                "direction": 1
            }
            self.obstacles.append(obstacle)
            occupied_positions.add(start_pos)

        # Place fruit
        self.fruit_pos = self._find_valid_spawn_location(occupied_positions)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False

        # --- 1. Update Player Position ---
        px, py = self.player_pos
        next_px, next_py = px, py

        if movement == 1 and py > 0 and not self.maze[py, px, 0]:  # Up
            next_py -= 1
        elif movement == 2 and py < self.MAZE_HEIGHT - 1 and not self.maze[py, px, 2]:  # Down
            next_py += 1
        elif movement == 3 and px > 0 and not self.maze[py, px, 3]:  # Left
            next_px -= 1
        elif movement == 4 and px < self.MAZE_WIDTH - 1 and not self.maze[py, px, 1]:  # Right
            next_px += 1

        self.player_pos = (next_px, next_py)

        # --- 2. Update Obstacles ---
        for obs in self.obstacles:
            obs['path_idx'] += obs['direction']
            if not (0 <= obs['path_idx'] < len(obs['path'])):
                obs['direction'] *= -1
                obs['path_idx'] += 2 * obs['direction']
                # Clamp index to be safe
                obs['path_idx'] = max(0, min(len(obs['path']) - 1, obs['path_idx']))

            obs['pos'] = obs['path'][obs['path_idx']]

        # --- 3. Update Particles ---
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0], p['pos'][1] - 0.5)
            p['lifetime'] -= 1

        # --- 4. Check for Events & Calculate Reward ---
        event_occurred = False

        # Check for obstacle collision
        for obs in self.obstacles:
            if self.player_pos == obs['pos']:
                reward = -100.0
                terminated = True
                self.game_over = True
                event_occurred = True
                break

        if not event_occurred:
            # Check for fruit collection
            if self.player_pos == self.fruit_pos:
                event_occurred = True
                self.fruits_collected += 1
                base_reward = 1.0

                # Check for risky bonus
                min_dist_to_obs = min(
                    abs(self.player_pos[0] - o['pos'][0]) + abs(self.player_pos[1] - o['pos'][1])
                    for o in self.obstacles
                ) if self.obstacles else float('inf')

                bonus_reward = 0.0
                if min_dist_to_obs <= 1:
                    bonus_reward = 5.0
                    # Add particle effect
                    px_pos = (
                        (self.player_pos[0] + 0.5) * self.CELL_SIZE,
                        (self.player_pos[1] + 0.5) * self.CELL_SIZE
                    )
                    self.particles.append({
                        "text": f"+{int(bonus_reward)}",
                        "pos": px_pos,
                        "lifetime": 30,  # 1 second at 30fps
                        "color": self.COLOR_BONUS_TEXT
                    })

                reward = base_reward + bonus_reward

                # Check for win condition
                if self.fruits_collected >= self.WIN_CONDITION_FRUITS:
                    reward += 100.0
                    terminated = True
                    self.game_over = True
                else:
                    # Respawn fruit
                    occupied = {self.player_pos} | {o['pos'] for o in self.obstacles}
                    self.fruit_pos = self._find_valid_spawn_location(occupied)

        # If no event, apply proximity/safe move reward
        if not event_occurred:
            min_dist_to_obs = min(
                abs(self.player_pos[0] - o['pos'][0]) + abs(self.player_pos[1] - o['pos'][1])
                for o in self.obstacles
            ) if self.obstacles else float('inf')

            if min_dist_to_obs <= 2:
                reward = 0.1  # Risky move bonus
            else:
                reward = -0.2  # Safe move penalty

        self.score += reward
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)

        # Draw Maze Walls
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                px, py = x * self.CELL_SIZE, y * self.CELL_SIZE
                if self.maze[y, x, 0]:  # Top
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px + self.CELL_SIZE, py), 2)
                if self.maze[y, x, 1]:  # Right
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px + self.CELL_SIZE, py),
                                     (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if self.maze[y, x, 2]:  # Bottom
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py + self.CELL_SIZE),
                                     (px + self.CELL_SIZE, py + self.CELL_SIZE), 2)
                if self.maze[y, x, 3]:  # Left
                    pygame.draw.line(self.screen, self.COLOR_WALL, (px, py), (px, py + self.CELL_SIZE), 2)

        # Draw Obstacles
        for obs in self.obstacles:
            ox, oy = obs['pos']
            rect = pygame.Rect(ox * self.CELL_SIZE, oy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect.inflate(-4, -4))

        # Draw Fruit
        fx, fy = self.fruit_pos
        center_x = int(fx * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(fy * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE / 3)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_FRUIT)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_FRUIT)

        # Draw Player
        px, py = self.player_pos
        center_x = int(px * self.CELL_SIZE + self.CELL_SIZE / 2)
        center_y = int(py * self.CELL_SIZE + self.CELL_SIZE / 2)
        radius = int(self.CELL_SIZE / 2 * 0.8)
        # Outline
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_PLAYER_OUTLINE)
        # Fill
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius - 2, self.COLOR_PLAYER)

        # Draw Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30.0))))
            text_surf = self.font_small.render(p['text'], True, p['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=p['pos'])
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Fruits
        fruit_text = f"Fruits: {self.fruits_collected} / {self.WIN_CONDITION_FRUITS}"
        fruit_surf = self.font_large.render(fruit_text, True, self.COLOR_TEXT)
        fruit_rect = fruit_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(fruit_surf, fruit_rect)

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            win_text = "LEVEL COMPLETE!" if self.fruits_collected >= self.WIN_CONDITION_FRUITS else "GAME OVER"
            win_surf = self.font_large.render(win_text, True, self.COLOR_TEXT)
            win_rect = win_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(win_surf, win_rect)

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_collected": self.fruits_collected,
            "player_pos": self.player_pos,
            "fruit_pos": self.fruit_pos,
        }

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    import time

    # Set this to 'human' to visualize, or 'rgb_array' for headless
    render_mode = "human"

    env = GameEnv()

    if render_mode == "human":
        # Unset the dummy driver to allow for display
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.init()
        pygame.display.set_caption("Maze Runner")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    obs, info = env.reset(seed=42)
    done = False

    total_reward = 0
    total_steps = 0

    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop for human play
    running = True
    while running:
        action = [0, 0, 0]  # Default action: no-op

        if render_mode == "human":
            # Event handling for human mode
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # For turn-based, a single key press triggers a step
                if event.type == pygame.KEYDOWN:
                    if event.key in key_map:
                        action[0] = key_map[event.key]
                    # Allow no-op step for waiting
                    elif event.key == pygame.K_SPACE:
                        action[0] = 0
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    else:
                        continue  # Don't step on other key presses

                    # Take a step in the environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    total_steps += 1
                    done = terminated or truncated
        else: # Headless automated testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            done = terminated or truncated


        # Rendering for human mode
        if render_mode == "human":
            # Convert observation back to a surface and draw it
            # Pygame uses (width, height), numpy uses (height, width)
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(30) # Limit frame rate


        # Handle episode end
        if done:
            print(f"Episode finished after {total_steps} steps.")
            print(f"Final Score: {info['score']:.2f}")
            print(f"Fruits Collected: {info['fruits_collected']}")

            if render_mode == "human":
                time.sleep(2)  # Pause before reset
            
            if not running: break

            # Reset for a new game
            obs, info = env.reset(seed=43)
            done = False
            total_reward = 0
            total_steps = 0
            if render_mode != "human": # Run one episode in headless
                running = False


    env.close()