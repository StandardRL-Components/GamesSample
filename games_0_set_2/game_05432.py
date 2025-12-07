
# Generated: 2025-08-28T04:59:44.799993
# Source Brief: brief_05432.md
# Brief Index: 5432

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a light crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Illuminate a dark cavern by strategically placing light-emitting crystals. "
        "Your goal is to light up 100% of the cavern before you run out of crystals."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.CELL_SIZE = 20

        # Colors
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_LIT_AREA = (60, 65, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.CRYSTAL_COLORS = [
            (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 128)
        ]

        # Game parameters
        self.INITIAL_CRYSTALS = 25
        self.MAX_STEPS = 1000
        self.LIGHT_RADIUS = 7
        self.WALL_DENSITY = 0.15

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
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 60)
        
        # Etc...        
        self.grid = None
        self.illumination_grid = None
        self.total_empty_cells = 0
        self.crystals = []
        self.cursor_pos = [0, 0]
        self.crystals_remaining = 0
        self.illumination_pct = 0.0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.milestone_50_reached = False
        self.milestone_75_reached = False
        self.np_random = None
        
        # Initialize state variables
        self.reset()
    
    def _generate_cavern(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        # Create borders
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # Add random internal walls
        num_walls = int((self.GRID_WIDTH * self.GRID_HEIGHT) * self.WALL_DENSITY)
        for _ in range(num_walls):
            x = self.np_random.integers(1, self.GRID_WIDTH - 1)
            y = self.np_random.integers(1, self.GRID_HEIGHT - 1)
            self.grid[x, y] = 1

        self.total_empty_cells = np.sum(self.grid == 0)
        if self.total_empty_cells == 0: # Failsafe for over-dense maps
            self._generate_cavern()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_cavern()
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        # Ensure cursor starts on an empty space
        while self.grid[self.cursor_pos[0], self.cursor_pos[1]] != 0:
            self.cursor_pos[0] = self.np_random.integers(1, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = self.np_random.integers(1, self.GRID_HEIGHT - 1)

        self.crystals = []
        self.crystals_remaining = self.INITIAL_CRYSTALS
        
        self.illumination_grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=bool)
        self.illumination_pct = 0.0
        
        self.milestone_50_reached = False
        self.milestone_75_reached = False

        self._calculate_illumination()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Actions ---
        # Movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        # Place Crystal
        if space_held:
            cx, cy = self.cursor_pos
            can_place = (
                self.crystals_remaining > 0 and
                self.grid[cx, cy] == 0
            )
            if can_place:
                # SFX: Crystal placement sound
                old_lit_cells = np.sum(self.illumination_grid[self.grid == 0])

                crystal_color_index = len(self.crystals) % len(self.CRYSTAL_COLORS)
                self.crystals.append({"pos": (cx, cy), "color": self.CRYSTAL_COLORS[crystal_color_index]})
                self.grid[cx, cy] = 2 + crystal_color_index
                self.crystals_remaining -= 1
                
                self._calculate_illumination()
                
                new_lit_cells = np.sum(self.illumination_grid[self.grid == 0])
                reward += (new_lit_cells - old_lit_cells) * 0.1
            else:
                # SFX: Error/buzz sound
                pass

        # --- 2. Update State & Calculate Rewards ---
        self.steps += 1
        
        # Milestone rewards
        if self.illumination_pct >= 50 and not self.milestone_50_reached:
            reward += 5
            self.milestone_50_reached = True
        if self.illumination_pct >= 75 and not self.milestone_75_reached:
            reward += 10
            self.milestone_75_reached = True

        # --- 3. Check Termination ---
        win = self.illumination_pct >= 100.0
        loss = self.crystals_remaining <= 0 and not win
        timeout = self.steps >= self.MAX_STEPS
        
        terminated = win or loss or timeout
        if terminated:
            self.game_over = True
            if win:
                reward += 100
            elif loss:
                reward -= 100
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _calculate_illumination(self):
        self.illumination_grid.fill(False)
        if not self.crystals:
            self.illumination_pct = 0.0
            return

        for crystal in self.crystals:
            q = deque([(crystal["pos"], 0)])
            visited = {crystal["pos"]}
            
            while q:
                (x, y), dist = q.popleft()
                
                self.illumination_grid[x, y] = True
                
                if dist < self.LIGHT_RADIUS:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                            if self.grid[nx, ny] != 1:
                                visited.add((nx, ny))
                                q.append(((nx, ny), dist + 1))
        
        lit_cells = np.sum(self.illumination_grid[self.grid == 0])
        self.illumination_pct = (lit_cells / self.total_empty_cells) * 100 if self.total_empty_cells > 0 else 100.0

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid cells (lit areas and walls)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, rect)
                elif self.illumination_grid[x, y]:
                    pygame.draw.rect(self.screen, self.COLOR_LIT_AREA, rect, 1)
        
        # Draw crystals and their glow
        for crystal in self.crystals:
            cx, cy = crystal["pos"]
            px, py = int((cx + 0.5) * self.CELL_SIZE), int((cy + 0.5) * self.CELL_SIZE)
            color = crystal["color"]
            
            # Glow effect using semi-transparent filled circles
            for i in range(7, 0, -1):
                alpha = 40 - i * 5
                glow_color = (color[0], color[1], color[2], alpha)
                radius = int(self.CELL_SIZE * 0.3 + i * 1.5)
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, glow_color, (radius, radius), radius)
                self.screen.blit(s, (px - radius, py - radius))
            
            # Core crystal
            pygame.gfxdraw.aacircle(self.screen, px, py, int(self.CELL_SIZE * 0.4), color)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.4), color)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)

    def _render_ui(self):
        # Illumination percentage
        illum_text = f"Illumination: {self.illumination_pct:.1f}%"
        text_surface = self.font_main.render(illum_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        # Crystals remaining
        crystal_text = f"Crystals: {self.crystals_remaining}"
        text_surface = self.font_main.render(crystal_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            if self.illumination_pct >= 100.0:
                msg = "VICTORY!"
                color = self.CRYSTAL_COLORS[0]
            else:
                msg = "CAVERN REMAINS DARK"
                color = self.COLOR_WALL
            
            text_surface = self.font_large.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_rect = text_rect.inflate(40, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": self.crystals_remaining,
            "illumination_pct": self.illumination_pct,
        }

    def close(self):
        pygame.quit()

# This block allows the game to be run and played directly
if __name__ == "__main__":
    env = GameEnv()
    env.reset()

    running = True
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Cavern Illuminator")

    print(env.user_guide)
    print(env.game_description)

    # Use a clock to prevent spamming actions
    action_timer = 0
    ACTION_COOLDOWN = 100 # milliseconds

    while running:
        time_delta = env.clock.tick(30)
        action_timer += time_delta
        
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                action_timer = 0

        # Only process actions if cooldown has passed
        if action_timer > ACTION_COOLDOWN:
            keys = pygame.key.get_pressed()
            action_taken = False
            if keys[pygame.K_UP]:
                movement = 1
                action_taken = True
            elif keys[pygame.K_DOWN]:
                movement = 2
                action_taken = True
            elif keys[pygame.K_LEFT]:
                movement = 3
                action_taken = True
            elif keys[pygame.K_RIGHT]:
                movement = 4
                action_taken = True
            
            if keys[pygame.K_SPACE]:
                space = 1
                action_taken = True
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
                # This action does nothing, so not considered 'action_taken'

            if action_taken:
                current_action = np.array([movement, space, shift])
                obs, reward, terminated, _, info = env.step(current_action)
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Illum: {info['illumination_pct']:.1f}%")
                if terminated:
                    print("Game Over! Press 'R' to restart.")
                action_timer = 0 # Reset timer after an action

        # Draw the observation from the environment
        obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

    env.close()