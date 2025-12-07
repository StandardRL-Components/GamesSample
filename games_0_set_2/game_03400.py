
# Generated: 2025-08-27T23:15:12.708114
# Source Brief: brief_03400.md
# Brief Index: 3400

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move on the grid. Collect all gems to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Navigate a grid to collect yellow gems while dodging red enemies. "
        "Collect 5 gems to advance to the next stage. You have 3 lives. Good luck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.GRID_SIZE = (16, 10)  # 16 cells wide, 10 cells high
        self.CELL_SIZE = (
            self.screen_size[0] // self.GRID_SIZE[0],
            self.screen_size[1] // self.GRID_SIZE[1],
        ) # (40, 40)
        self.MAX_STEPS = 1000
        self.TOTAL_LIVES = 3
        self.GEMS_PER_STAGE = 5
        self.TOTAL_STAGES = 3

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25) # Dark blue/black
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255, 50)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_GEM_GLOW = (255, 220, 0, 80)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 100, 100, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEART = (255, 20, 80)

        # --- Fonts ---
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.stage = 0
        self.gems_collected_this_stage = 0
        self.total_gems_collected = 0
        self.player_pos = [0, 0]
        self.gem_positions = []
        self.enemies = []
        self.game_over = False
        self.win = False
        self.transition_timer = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset core state
        self.steps = 0
        self.score = 0
        self.lives = self.TOTAL_LIVES
        self.stage = 1
        self.gems_collected_this_stage = 0
        self.total_gems_collected = 0
        self.game_over = False
        self.win = False
        self.transition_timer = 0
        
        # Setup the first stage
        self._setup_stage()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the game elements for the current stage."""
        self.gems_collected_this_stage = 0
        
        # Clear existing elements
        self.gem_positions.clear()
        self.enemies.clear()
        
        # Player position (center-left)
        self.player_pos = [1, self.GRID_SIZE[1] // 2]
        
        # Generate gems in random, non-occupied cells
        occupied_cells = {tuple(self.player_pos)}
        while len(self.gem_positions) < self.GEMS_PER_STAGE:
            pos = [
                self.np_random.integers(0, self.GRID_SIZE[0]),
                self.np_random.integers(0, self.GRID_SIZE[1]),
            ]
            if tuple(pos) not in occupied_cells:
                self.gem_positions.append(pos)
                occupied_cells.add(tuple(pos))

        # Generate enemies based on stage
        if self.stage == 1:
            # Horizontal patrol
            path = [[x, 2] for x in range(self.GRID_SIZE[0] - 4, self.GRID_SIZE[0] - 1)]
            path += reversed(path[1:-1])
            self.enemies.append({"pos": path[0], "path": path, "path_index": 0})
        elif self.stage == 2:
            # Horizontal
            path1 = [[x, 2] for x in range(self.GRID_SIZE[0] - 6, self.GRID_SIZE[0] - 1)]
            path1 += reversed(path1[1:-1])
            self.enemies.append({"pos": path1[0], "path": path1, "path_index": 0})
            # Vertical
            path2 = [[3, y] for y in range(0, self.GRID_SIZE[1] - 2)]
            path2 += reversed(path2[1:-1])
            self.enemies.append({"pos": path2[0], "path": path2, "path_index": 0})
        elif self.stage == 3:
            # Rectangular patrol
            path1 = [[x, 1] for x in range(10, 15)] + [[14, y] for y in range(2, 5)] + \
                    [[x, 4] for x in range(14, 9, -1)] + [[10, y] for y in range(3, 1, -1)]
            self.enemies.append({"pos": path1[0], "path": path1, "path_index": 0})
            # Diagonal-ish patrol
            path2 = [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [5, 5], [4, 4], [3, 3]]
            self.enemies.append({"pos": path2[0], "path": path2, "path_index": 0})
            # Fast horizontal patrol
            path3 = [[x, 8] for x in range(1, self.GRID_SIZE[0] - 1)]
            path3 += reversed(path3[1:-1])
            self.enemies.append({"pos": path3[0], "path": path3, "path_index": 0})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Handle stage transition delay
        if self.transition_timer > 0:
            self.transition_timer -= 1
            if self.transition_timer == 0:
                self._setup_stage()
            return self._get_observation(), 0, False, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean (not used)
        shift_held = action[2] == 1  # Boolean (not used)
        
        reward = 0
        self.steps += 1
        
        # Pre-move state for reward calculation
        dist_before = self._get_nearest_gem_dist()

        # Update player position
        if movement == 1:  # Up
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif movement == 2:  # Down
            self.player_pos[1] = min(self.GRID_SIZE[1] - 1, self.player_pos[1] + 1)
        elif movement == 3:  # Left
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif movement == 4:  # Right
            self.player_pos[0] = min(self.GRID_SIZE[0] - 1, self.player_pos[0] + 1)
        
        # Move enemies
        self._move_enemies()
        
        # --- Check for Interactions and Calculate Rewards ---
        # Gem collection
        if self.player_pos in self.gem_positions:
            self.gem_positions.remove(self.player_pos)
            self.score += 1
            reward += 1
            self.gems_collected_this_stage += 1
            self.total_gems_collected += 1
            # sfx: gem_collect.wav
        
        # Enemy collision
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                self.lives -= 1
                reward -= 5
                self.player_pos = [1, self.GRID_SIZE[1] // 2] # Reset player pos to safe spot
                # sfx: player_hit.wav
                break # Only one hit per step

        # Movement reward/penalty
        dist_after = self._get_nearest_gem_dist()
        if movement == 0: # No-op
             reward -= 0.2
        elif dist_after is not None and dist_before is not None:
             if dist_after > dist_before: # Moved away from nearest gem
                 reward -= 0.2

        # --- Check for Stage/Game End ---
        if self.gems_collected_this_stage >= self.GEMS_PER_STAGE:
            if self.stage < self.TOTAL_STAGES:
                self.stage += 1
                self.transition_timer = 2 # Wait 2 steps for transition screen
                # sfx: stage_clear.wav
            else:
                self.win = True
                self.game_over = True
        
        if self.lives <= 0:
            self.game_over = True
            self.win = False
            # sfx: game_over.wav
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_enemies(self):
        for enemy in self.enemies:
            enemy["path_index"] = (enemy["path_index"] + 1) % len(enemy["path"])
            enemy["pos"] = enemy["path"][enemy["path_index"]]
            
    def _get_nearest_gem_dist(self):
        if not self.gem_positions:
            return None
        return min([
            abs(self.player_pos[0] - gx) + abs(self.player_pos[1] - gy)
            for gx, gy in self.gem_positions
        ])

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_gems()
        self._render_enemies()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        if self.transition_timer > 0:
            self._render_transition_screen()
        
        if self.game_over:
            self._render_game_over_screen()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
            "win": self.win
        }

    # --- Rendering Methods ---
    def _render_grid(self):
        for x in range(0, self.screen_size[0], self.CELL_SIZE[0]):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_size[1]))
        for y in range(0, self.screen_size[1], self.CELL_SIZE[1]):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_size[0], y))

    def _render_player(self):
        px, py = self.player_pos
        rect = pygame.Rect(
            px * self.CELL_SIZE[0],
            py * self.CELL_SIZE[1],
            self.CELL_SIZE[0],
            self.CELL_SIZE[1]
        )
        
        bob = math.sin(self.steps * 0.3) * 2
        center = rect.center
        
        glow_radius = int(self.CELL_SIZE[0] * 0.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius + bob), special_flags=pygame.BLEND_RGBA_ADD)
        
        body_rect = pygame.Rect(0, 0, self.CELL_SIZE[0] * 0.7, self.CELL_SIZE[1] * 0.7)
        body_rect.center = (center[0], center[1] + bob)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
        
    def _render_gems(self):
        for gx, gy in self.gem_positions:
            center_x = int((gx + 0.5) * self.CELL_SIZE[0])
            center_y = int((gy + 0.5) * self.CELL_SIZE[1])
            
            pulse = (math.sin(self.steps * 0.2 + gx) + 1) / 2 # 0 to 1
            size = self.CELL_SIZE[0] * (0.3 + pulse * 0.1)
            
            glow_radius = int(size * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_GEM_GLOW)
            self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            points = [
                (center_x, center_y - size),
                (center_x + size, center_y),
                (center_x, center_y + size),
                (center_x - size, center_y)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)

    def _render_enemies(self):
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            rect = pygame.Rect(
                ex * self.CELL_SIZE[0],
                ey * self.CELL_SIZE[1],
                self.CELL_SIZE[0],
                self.CELL_SIZE[1]
            )
            center = rect.center
            
            glow_radius = int(self.CELL_SIZE[0] * 0.7)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_ENEMY_GLOW)
            self.screen.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            body_size = int(self.CELL_SIZE[0] * 0.7)
            body_rect = pygame.Rect(0, 0, body_size, body_size)
            body_rect.center = center
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, body_rect, border_radius=4)

    def _render_ui(self):
        score_text = self.font_medium.render(f"GEMS: {self.total_gems_collected}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        stage_text = self.font_medium.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.screen_size[0] - stage_text.get_width() - 10, 10))
        
        heart_size = 20
        for i in range(self.lives):
            self._draw_heart(self.screen_size[0] // 2 - (self.TOTAL_LIVES * (heart_size+5)) // 2 + i * (heart_size + 5), 15, heart_size)

    def _draw_heart(self, x, y, size):
        points = [
            (x + size // 2, y + size), (x, y + size // 3), (x + size // 4, y),
            (x + size // 2, y + size // 4), (x + size * 3 // 4, y), (x + size, y + size // 3),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)
        
    def _render_transition_screen(self):
        overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = self.font_large.render(f"STAGE {self.stage-1} CLEAR!", True, self.COLOR_GEM)
        text_rect = text.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2 - 20))
        self.screen.blit(overlay, (0,0))
        self.screen.blit(text, text_rect)
        
        sub_text = self.font_medium.render(f"Next: Stage {self.stage}", True, self.COLOR_TEXT)
        sub_text_rect = sub_text.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2 + 30))
        self.screen.blit(sub_text, sub_text_rect)

    def _render_game_over_screen(self):
        overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_GEM if self.win else self.COLOR_ENEMY
        
        text = self.font_large.render(message, True, color)
        text_rect = text.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2))
        
        self.screen.blit(overlay, (0,0))
        self.screen.blit(text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''Call this at the end of __init__ to verify implementation:'''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly to test and debug.
    import os
    # Set the display driver. Use "x11", "wayland", "windows", "mac", or "dummy".
    # If you are on a headless server, use "dummy".
    # os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Grid Gem Collector")
    
    terminated = False
    
    # Game loop for human play
    while not terminated:
        movement = 0 # No-op
        
        # Event handling for key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_r: # Press 'r' to reset
                    obs, info = env.reset()
                    continue
                else: # Any other key can be a no-op step
                    movement = 0

                # Since auto_advance is False, we only step on a key press
                action = [movement, 0, 0]
                obs, reward, term, trunc, info = env.step(action)
                terminated = term
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the current state from the observation
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

    env.close()
    pygame.quit()