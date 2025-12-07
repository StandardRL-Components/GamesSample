import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Reach the glowing exit before the timer runs out. Avoid the shadows."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A tense survival horror game. Navigate a dark, procedurally generated maze and escape before you are caught by the lurking shadows or the timer expires."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_W, self.MAZE_H = 31, 19  # Must be odd numbers
        self.CELL_SIZE = 20
        self.MAZE_DRAW_OFFSET_X = (self.WIDTH - self.MAZE_W * self.CELL_SIZE) // 2
        self.MAZE_DRAW_OFFSET_Y = (self.HEIGHT - self.MAZE_H * self.CELL_SIZE) // 2
        self.MAX_STEPS = 3000
        self.NUM_ENEMIES = 5
        self.ENEMY_PATROL_LENGTH = 10

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PATH = self.COLOR_BG
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_EXIT = (220, 220, 255)
        self.COLOR_ENEMY = (5, 5, 5)
        self.COLOR_UI_BAR = (200, 200, 220)
        self.COLOR_UI_TEXT = (200, 200, 220)
        
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
        try:
            self.font_ui = pygame.font.SysFont("monospace", 16)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 20)

        # State variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.enemies = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.np_random = None

        self.validate_implementation()
    
    def _generate_maze(self):
        # Maze grid: 1 for wall, 0 for path
        maze = np.ones((self.MAZE_H, self.MAZE_W), dtype=np.uint8)
        
        # Use Randomized DFS to carve paths
        stack = deque()
        start_x, start_y = (self.np_random.integers(0, self.MAZE_W // 2) * 2 + 1, 
                            self.np_random.integers(0, self.MAZE_H // 2) * 2 + 1)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.MAZE_W - 1 and 0 < ny < self.MAZE_H - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors) # Use standard random for this part
                # Carve path
                maze[ny, nx] = 0
                maze[(y + ny) // 2, (x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def _generate_enemy_patrol(self):
        path_cells = np.argwhere(self.maze == 0)
        start_idx = self.np_random.integers(0, len(path_cells))
        start_pos = tuple(path_cells[start_idx])

        path = [start_pos]
        current_pos = start_pos
        
        for _ in range(self.ENEMY_PATROL_LENGTH):
            y, x = current_pos
            possible_moves = []
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.MAZE_H and 0 <= nx < self.MAZE_W and self.maze[ny, nx] == 0:
                    possible_moves.append((ny, nx))
            
            if not possible_moves:
                break
            
            # Avoid immediately going back if possible
            if len(possible_moves) > 1 and len(path) > 1:
                last_pos = path[-2]
                possible_moves = [move for move in possible_moves if move != last_pos]

            move_idx = self.np_random.integers(0, len(possible_moves))
            current_pos = possible_moves[move_idx]
            path.append(current_pos)

        # Create a loop by returning
        if len(path) > 1:
            patrol_loop = path + path[-2:0:-1]
        else:
            patrol_loop = path
        return patrol_loop


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.maze = self._generate_maze()
        
        path_cells = np.argwhere(self.maze == 0)
        
        # Player and Exit positions
        start_idx, exit_idx = self.np_random.choice(len(path_cells), 2, replace=False)
        self.player_pos = tuple(path_cells[start_idx])
        self.exit_pos = tuple(path_cells[exit_idx])

        # Ensure player and exit are far apart
        while np.linalg.norm(np.array(self.player_pos) - np.array(self.exit_pos)) < (self.MAZE_W + self.MAZE_H) / 4:
            exit_idx = self.np_random.integers(0, len(path_cells))
            self.exit_pos = tuple(path_cells[exit_idx])

        # Initialize enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            patrol_path = self._generate_enemy_patrol()
            if not patrol_path: continue
            self.enemies.append({
                "path": patrol_path,
                "path_index": 0,
                "pos": patrol_path[0]
            })

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.1 # Survival reward
        
        # Player Movement
        py, px = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        if 0 <= py < self.MAZE_H and 0 <= px < self.MAZE_W and self.maze[py, px] == 0:
            self.player_pos = (py, px)

        # Enemy Movement
        for enemy in self.enemies:
            if enemy["path"]:
                enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
                enemy["pos"] = enemy["path"][enemy["path_index"]]

        # Update game state
        self.steps += 1
        
        # Check for termination conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 50
            self.game_over = True
            terminated = True
        
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                reward -= 50
                self.game_over = True
                terminated = True
                self._spawn_hit_particles()
                break
        
        truncated = False
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True

        self.score += reward
        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_hit_particles(self):
        player_screen_pos = self._cell_to_screen(self.player_pos[1], self.player_pos[0])
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(player_screen_pos),
                "vel": vel,
                "life": self.np_random.integers(15, 25),
                "color": (255, 50, 50),
                "radius": self.np_random.random() * 2 + 1
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _cell_to_screen(self, x, y):
        return (
            int(x * self.CELL_SIZE + self.MAZE_DRAW_OFFSET_X + self.CELL_SIZE // 2),
            int(y * self.CELL_SIZE + self.MAZE_DRAW_OFFSET_Y + self.CELL_SIZE // 2)
        )

    def _render_game(self):
        # Draw maze
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                color = self.COLOR_WALL if self.maze[y, x] == 1 else self.COLOR_PATH
                rect = pygame.Rect(
                    x * self.CELL_SIZE + self.MAZE_DRAW_OFFSET_X,
                    y * self.CELL_SIZE + self.MAZE_DRAW_OFFSET_Y,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)

        # Draw Exit
        ex, ey = self._cell_to_screen(self.exit_pos[1], self.exit_pos[0])
        pulse = abs(math.sin(self.steps * 0.05))
        for i in range(5, 0, -1):
            alpha = 100 - i * 15
            radius = int(self.CELL_SIZE * 0.5 + i * 2 + pulse * 3)
            pygame.gfxdraw.filled_circle(self.screen, ex, ey, radius, (*self.COLOR_EXIT, alpha))
        pygame.gfxdraw.filled_circle(self.screen, ex, ey, int(self.CELL_SIZE * 0.4), self.COLOR_EXIT)
        
        # Draw Enemies
        for enemy in self.enemies:
            ey, ex = enemy["pos"]
            ecx, ecy = self._cell_to_screen(ex, ey)
            # Flickering shadow effect
            for i in range(3):
                offset_x = (self.np_random.random() - 0.5) * self.CELL_SIZE * 0.6
                offset_y = (self.np_random.random() - 0.5) * self.CELL_SIZE * 0.6
                size = int(self.CELL_SIZE * (self.np_random.random() * 0.4 + 0.8))
                shadow_rect = pygame.Rect(ecx + offset_x - size//2, ecy + offset_y - size//2, size, size)
                pygame.draw.rect(self.screen, (*self.COLOR_ENEMY, 100), shadow_rect)
        
        # Draw Player
        px, py = self._cell_to_screen(self.player_pos[1], self.player_pos[0])
        player_rect = pygame.Rect(px - self.CELL_SIZE//4, py - self.CELL_SIZE//4, self.CELL_SIZE//2, self.CELL_SIZE//2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_BG, player_rect, 1)

        # Draw Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["pos"], p["radius"], p["color"])

    def _render_ui(self):
        # Timer bar
        timer_width = self.WIDTH * max(0, 1 - self.steps / self.MAX_STEPS)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (0, 0, timer_width, 5))
        
        # Score text
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        if self.maze is not None:
             self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset, which initializes the state and returns the first observation
        obs, info = self.reset()
        
        # Test observation space using the result from reset
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Maze Horror")
    
    running = True
    terminated = False
    truncated = False
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        if not (terminated or truncated):
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                pygame.time.wait(100) # Small delay to control speed for human players
        else:
             # Render game over/win screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))

            font_game_over = pygame.font.SysFont("monospace", 50)
            if info["score"] > 0 and terminated:
                text = font_game_over.render("YOU ESCAPED", True, (200, 255, 200))
            else:
                text = font_game_over.render("GAME OVER", True, (255, 100, 100))
            text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
            display_screen.blit(text, text_rect)
            
            font_restart = pygame.font.SysFont("monospace", 20)
            restart_text = font_restart.render("Press 'R' to restart", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(screen_width // 2, screen_height // 2 + 50))
            display_screen.blit(restart_text, restart_rect)

            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                truncated = False
        
        if not (terminated or truncated):
            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)

    env.close()