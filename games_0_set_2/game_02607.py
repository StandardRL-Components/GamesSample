
# Generated: 2025-08-28T05:23:05.317188
# Source Brief: brief_02607.md
# Brief Index: 2607

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Avoid hidden traps. Reach the green exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated dungeon, avoiding deadly traps to find the exit in this top-down horror experience."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 48
        self.GRID_WIDTH = 41 # Must be odd
        self.GRID_HEIGHT = 31 # Must be odd
        
        self.MAX_STEPS = 1000
        self.MAX_TRAP_HITS = 5
        self.TRAP_DENSITY = 0.05

        # --- Colors ---
        self.COLOR_BG = (15, 15, 20)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_TRAP_FLASH = (255, 20, 20)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_DANGER = (255, 80, 80)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        self._create_vignette()
        
        # --- State Variables (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.trap_hits = 0
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.trap_locations = None
        self.particles = []
        self.triggered_traps = {} # {step: pos} for rendering flashes
        
        # Initialize state variables
        self.reset()
    
    def _create_vignette(self):
        """Creates a pre-rendered vignette surface for a spooky atmosphere."""
        self.vignette_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for i in range(max(self.WIDTH, self.HEIGHT) // 2, 0, -2):
            alpha = int(255 * (1 - (i / (max(self.WIDTH, self.HEIGHT) / 2)))**2)
            pygame.gfxdraw.filled_ellipse(
                self.vignette_surface,
                self.WIDTH // 2, self.HEIGHT // 2,
                i, int(i * (self.HEIGHT / self.WIDTH)),
                (0, 0, 0, 4)
            )

    def _generate_maze(self):
        """Generates a perfect maze using recursive backtracking (DFS)."""
        maze = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        stack = []
        
        start_x, start_y = 1, 1
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and maze[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # Use the environment's RNG for deterministic generation
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                
                maze[ny, nx] = 0
                maze[cy + (ny - cy) // 2, cx + (nx - cx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        return maze

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None: # First reset
             self.np_random = np.random.default_rng(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.trap_hits = 0
        self.particles = []
        self.triggered_traps = {}

        self.maze = self._generate_maze()
        
        self.player_pos = np.array([1, 1])
        self.exit_pos = np.array([self.GRID_WIDTH - 2, self.GRID_HEIGHT - 2])

        floor_tiles = np.argwhere(self.maze == 0)
        valid_trap_tiles = [
            tuple(tile) for tile in floor_tiles 
            if not np.array_equal(tile, self.player_pos[::-1]) and not np.array_equal(tile, self.exit_pos[::-1])
        ]
        
        num_traps = int(len(valid_trap_tiles) * self.TRAP_DENSITY)
        trap_indices = self.np_random.choice(len(valid_trap_tiles), num_traps, replace=False)
        self.trap_locations = {tuple(valid_trap_tiles[i][::-1]) for i in trap_indices}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Base penalty for taking a step ---
        reward -= 0.1
        self.steps += 1
        
        # --- Handle Action ---
        if movement != 0: # 0 is no-op
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_map[movement]
            
            next_pos = self.player_pos + np.array([dx, dy])
            
            # Check for wall collision
            if self.maze[next_pos[1], next_pos[0]] == 0:
                self.player_pos = next_pos
                # sound: footstep.wav
            else:
                pass # sound: bump_wall.wav

        # --- Game Logic ---
        player_pos_tuple = tuple(self.player_pos)
        if player_pos_tuple in self.trap_locations:
            self.trap_locations.remove(player_pos_tuple)
            self.trap_hits += 1
            reward -= 10
            self.triggered_traps[self.steps] = player_pos_tuple
            # sound: trap_sprung.wav
            
        if np.array_equal(self.player_pos, self.exit_pos):
            reward += 100
            self.game_over = True
            # sound: win.wav

        if self.trap_hits >= self.MAX_TRAP_HITS or self.steps >= self.MAX_STEPS:
            self.game_over = True
            # sound: lose.wav
        
        self.score += reward
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self.screen.blit(self.vignette_surface, (0, 0))
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
    
    def _render_game(self):
        offset_x = self.WIDTH / 2 - self.player_pos[0] * self.TILE_SIZE - self.TILE_SIZE / 2
        offset_y = self.HEIGHT / 2 - self.player_pos[1] * self.TILE_SIZE - self.TILE_SIZE / 2

        start_col = max(0, int(-offset_x / self.TILE_SIZE))
        end_col = min(self.GRID_WIDTH, int((-offset_x + self.WIDTH) / self.TILE_SIZE) + 2)
        start_row = max(0, int(-offset_y / self.TILE_SIZE))
        end_row = min(self.GRID_HEIGHT, int((-offset_y + self.HEIGHT) / self.TILE_SIZE) + 2)

        for y in range(start_row, end_row):
            for x in range(start_col, end_col):
                if self.maze[y, x] == 1:
                    rect = pygame.Rect(
                        offset_x + x * self.TILE_SIZE,
                        offset_y + y * self.TILE_SIZE,
                        self.TILE_SIZE, self.TILE_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
        
        # A trap flash lasts for 5 steps
        active_flashes = {}
        for step_triggered, pos in self.triggered_traps.items():
            if self.steps < step_triggered + 5:
                active_flashes[step_triggered] = pos
                
                flash_alpha = 1 - (self.steps - step_triggered) / 5
                intensity = 0.6 + 0.4 * math.sin(self.steps * 2) # Flicker
                flash_color = (*self.COLOR_TRAP_FLASH, int(255 * flash_alpha * intensity))

                flash_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                flash_surf.fill(flash_color)
                
                draw_x = offset_x + pos[0] * self.TILE_SIZE
                draw_y = offset_y + pos[1] * self.TILE_SIZE
                self.screen.blit(flash_surf, (draw_x, draw_y))
        self.triggered_traps = active_flashes

        exit_x = offset_x + self.exit_pos[0] * self.TILE_SIZE
        exit_y = offset_y + self.exit_pos[1] * self.TILE_SIZE
        exit_rect = pygame.Rect(exit_x, exit_y, self.TILE_SIZE, self.TILE_SIZE)
        
        glow_size = self.TILE_SIZE * 2.5
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pulse = 0.9 + 0.1 * math.sin(pygame.time.get_ticks() * 0.002)
        for i in range(5):
            alpha = 60 - i * 10
            radius = (glow_size / 2 - i * 2) * pulse
            pygame.gfxdraw.filled_circle(
                glow_surf, int(glow_size / 2), int(glow_size / 2), int(radius),
                (*self.COLOR_EXIT, alpha)
            )
        self.screen.blit(glow_surf, (exit_rect.centerx - glow_size / 2, exit_rect.centery - glow_size / 2))
        
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)

        player_draw_x = self.WIDTH / 2
        player_draw_y = self.HEIGHT / 2
        player_radius = int(self.TILE_SIZE / 3)
        pygame.gfxdraw.filled_circle(self.screen, int(player_draw_x), int(player_draw_y), player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(player_draw_x), int(player_draw_y), player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        trap_text_str = f"TRAPS: {self.trap_hits}/{self.MAX_TRAP_HITS}"
        trap_color = self.COLOR_TEXT if self.trap_hits == 0 else self.COLOR_TEXT_DANGER
        trap_text = self.font_ui.render(trap_text_str, True, trap_color)
        self.screen.blit(trap_text, (10, 10))

        step_text_str = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        step_text = self.font_ui.render(step_text_str, True, self.COLOR_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            won = np.array_equal(self.player_pos, self.exit_pos)
            end_text_str = "ESCAPED" if won else "YOU DIED"
            end_color = self.COLOR_EXIT if won else self.COLOR_TEXT_DANGER
                
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# This block allows for manual play and testing of the environment.
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset(seed=42)
    done = False
    
    pygame.display.set_caption("Dungeon Horror")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action = np.array([0, 0, 0]) # Default no-op
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    done = False
                    continue

                if not done:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    
                    if action[0] != 0:
                        action_taken = True
                
                if action_taken:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()