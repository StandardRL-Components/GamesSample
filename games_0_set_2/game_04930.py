import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to navigate the maze. Avoid enemies and collect power-ups to reach the exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated maze, dodging enemies and collecting power-ups to reach the exit within the time limit. Bold plays are rewarded!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_WIDTH
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 60.0
        self.TIME_PER_STEP = self.TIME_LIMIT_SECONDS / self.MAX_STEPS
        self.INITIAL_LIVES = 5
        self.NUM_ENEMIES = 8
        self.NUM_POWERUPS = 10

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_WALL = (20, 30, 80)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_ENEMY = (255, 0, 60)
        self.COLOR_POWERUP = (200, 0, 255)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TEXT = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont('monospace', 18, bold=True)
        self.font_game_over = pygame.font.SysFont('monospace', 48, bold=True)
        
        # Initialize state variables
        self.maze = None
        self.player_pos = None
        self.enemies = None
        self.powerups = None
        self.exit_pos = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.time_remaining = 0.0
        self.game_over = False
        self.game_outcome = ""
        self.np_random = None

        # self.reset() is called by the wrapper, but for standalone use it's good practice
        # However, to avoid issues with test harnesses that might call reset again,
        # we'll let the first call be from the outside.
        # We need to initialize the np_random generator here though.
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.game_outcome = ""

        self._generate_maze()
        self._place_game_elements()
        
        return self._get_observation(), self._get_info()

    def _generate_maze(self):
        self.maze = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8) # 1 = wall
        stack = []
        
        start_x, start_y = (
            self.np_random.integers(0, self.GRID_WIDTH // 2) * 2,
            self.np_random.integers(0, self.GRID_HEIGHT // 2) * 2
        )
        self.maze[start_y, start_x] = 0 # 0 = path
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # np_random.choice doesn't work on lists of tuples as expected
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                self.maze[ny, nx] = 0
                self.maze[y + (ny - y) // 2, x + (nx - x) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_game_elements(self):
        open_cells = np.argwhere(self.maze == 0).tolist()
        self.np_random.shuffle(open_cells)
        
        # Ensure there are enough cells before popping
        if len(open_cells) < 2: # Need at least player and exit
            # This is an edge case for tiny mazes, regenerate
            self._generate_maze()
            self._place_game_elements()
            return

        self.player_pos = tuple(open_cells.pop(0))
        self.exit_pos = tuple(open_cells.pop(-1))

        # Place powerups
        self.powerups = []
        for _ in range(self.NUM_POWERUPS):
            if open_cells:
                self.powerups.append(tuple(open_cells.pop(0)))

        # Place enemies
        self.enemies = []
        potential_paths = self._find_enemy_paths()
        if potential_paths:
            self.np_random.shuffle(potential_paths)
            for _ in range(self.NUM_ENEMIES):
                if potential_paths:
                    path = potential_paths.pop(0)
                    start_pos = path[self.np_random.integers(0, len(path))]
                    self.enemies.append({
                        "pos": start_pos,
                        "path": path,
                        "direction": 1
                    })

    def _find_enemy_paths(self):
        paths = []
        # Horizontal paths
        for r in range(self.GRID_HEIGHT):
            c = 0
            while c < self.GRID_WIDTH:
                if self.maze[r, c] == 0:
                    path = []
                    cc = c
                    while cc < self.GRID_WIDTH and self.maze[r, cc] == 0:
                        path.append((cc, r))
                        cc += 1
                    if len(path) > 2:
                        paths.append(path)
                    c = cc
                else:
                    c += 1
        # Vertical paths
        for c in range(self.GRID_WIDTH):
            r = 0
            while r < self.GRID_HEIGHT:
                if self.maze[r, c] == 0:
                    path = []
                    rr = r
                    while rr < self.GRID_HEIGHT and self.maze[rr, c] == 0:
                        path.append((c, rr))
                        rr += 1
                    if len(path) > 2:
                        paths.append(path)
                    r = rr
                else:
                    r += 1
        return paths

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Cost of existing

        # 1. Calculate pre-move state for reward
        old_player_pos = self.player_pos
        dist_before = self._get_dist_to_nearest_enemy(old_player_pos)

        # 2. Update player position
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT and self.maze[new_y, new_x] == 0:
                self.player_pos = (new_x, new_y)
        
        # 3. Calculate risk/reward based on movement relative to enemies
        dist_after = self._get_dist_to_nearest_enemy(self.player_pos)
        if dist_after < dist_before:
            reward += 0.5  # Risky move bonus
        elif dist_after > dist_before and movement != 0:
            reward -= 0.2  # Safe move penalty

        # 4. Update enemies
        self._move_enemies()

        # 5. Check for collisions and events
        # Player vs Powerup
        if self.player_pos in self.powerups:
            self.powerups.remove(self.player_pos)
            reward += 1.0
            self.score += 100

        # Player vs Enemy
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                self.lives -= 1
                reward -= 5.0
                self.player_pos = old_player_pos # Revert move
                break
        
        # 6. Update game state
        self.steps += 1
        self.time_remaining -= self.TIME_PER_STEP

        # 7. Check for termination
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 50.0
            self.score += 5000
            terminated = True
            self.game_over = True
            self.game_outcome = "YOU WON!"
        elif self.lives <= 0:
            reward -= 25.0 # Lesser penalty than timeout
            terminated = True
            self.game_over = True
            self.game_outcome = "OUT OF LIVES"
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 50.0
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME'S UP!"

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_enemy(self, pos):
        if not self.enemies:
            return float('inf')
        min_dist = float('inf')
        for enemy in self.enemies:
            dist = math.hypot(pos[0] - enemy["pos"][0], pos[1] - enemy["pos"][1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _move_enemies(self):
        for enemy in self.enemies:
            try:
                current_index = enemy["path"].index(enemy["pos"])
                next_index = current_index + enemy["direction"]
                if not (0 <= next_index < len(enemy["path"])):
                    enemy["direction"] *= -1
                    next_index = current_index + enemy["direction"]
                enemy["pos"] = enemy["path"][next_index]
            except (ValueError, IndexError):
                # Enemy is off its path, reset it to the start
                if enemy["path"]:
                    enemy["pos"] = enemy["path"][0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self.maze is not None: # Ensure maze exists before rendering
            self._render_game()
            self._render_ui()
            if self.game_over:
                self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.maze[r, c] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw exit
        ex, ey = self.exit_pos
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw powerups
        flicker = (self.steps % 10) < 5
        for px, py in self.powerups:
            center_x = int((px + 0.5) * self.CELL_SIZE)
            center_y = int((py + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.3) if flicker else int(self.CELL_SIZE * 0.35)
            points = [
                (center_x, center_y - radius),
                (center_x + radius, center_y),
                (center_x, center_y + radius),
                (center_x - radius, center_y),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_POWERUP)

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            rect = pygame.Rect(ex * self.CELL_SIZE + 2, ey * self.CELL_SIZE + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)

        # Draw player
        px, py = self.player_pos
        player_center = (int((px + 0.5) * self.CELL_SIZE), int((py + 0.5) * self.CELL_SIZE))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_center, int(self.CELL_SIZE * 0.4))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Lives
        heart_size = 15
        y = 5  # The y-coordinate for the top of the hearts
        for i in range(self.lives):
            x_pos = self.WIDTH // 2 - (self.INITIAL_LIVES * (heart_size + 5)) // 2 + i * (heart_size + 5)
            # Define points with absolute screen coordinates
            points = [
                (x_pos + heart_size // 2, y + heart_size),
                (x_pos, y + heart_size // 3),
                (x_pos + heart_size // 4, y),
                (x_pos + heart_size // 2, y + heart_size // 4),
                (x_pos + heart_size * 3 // 4, y),
                (x_pos + heart_size, y + heart_size // 3)
            ]
            # Draw the polygon directly with the calculated points
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Time
        time_text = self.font_ui.render(f"TIME: {max(0, int(self.time_remaining))}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 5))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = self.font_game_over.render(self.game_outcome, True, self.COLOR_TEXT)
        text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block will not be executed in the testing environment,
    # but is useful for local testing.
    # It requires removing the "dummy" video driver setting.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            if movement != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")
        
        # Get the observation from the environment for display
        obs_for_display = env._get_observation()
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose from (H, W, C) to (W, H, C) for pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs_for_display, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000)
            print("Resetting environment.")
            obs, info = env.reset()
            terminated = False

        clock.tick(30)

    env.close()