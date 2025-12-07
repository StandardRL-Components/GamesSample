import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:19:42.839621
# Source Brief: brief_01660.md
# Brief Index: 1660
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Shift rows of colored lights to create 3x3 blocks of the same color. "
        "Race against the clock to build up enough energy to win."
    )
    user_guide = (
        "Controls: Use ← and → to shift the selected row. "
        "Use Space and Shift to move the selector down and up."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    CELL_SIZE = 40
    
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID_LINES = (30, 40, 60)
    COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 80, 255),   # Blue
    ]
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Game Parameters
    WIN_ENERGY = 5
    GAME_DURATION_SECONDS = 45
    FPS = 60 # Assumed FPS for step-based timing
    MAX_STEPS = GAME_DURATION_SECONDS * FPS + (5 * FPS) # 45s + 5s buffer
    TIMER_MAX_STEPS = GAME_DURATION_SECONDS * FPS
    COLOR_CHANGE_INTERVAL = 1 * FPS # Colors change every second

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # State variables are initialized in reset()
        self.grid_colors = None
        self.visual_row_offsets = None
        self.selected_row = 0
        self.steps = 0
        self.score = 0
        self.energy = 0
        self.game_over = False
        self.match_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # This will be called once to ensure everything is set up for the first reset
        # self.reset() # reset is called by the wrapper, no need to call it here
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid_colors = self.np_random.integers(0, len(self.COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self.visual_row_offsets = np.zeros(self.GRID_HEIGHT, dtype=float)
        
        self.selected_row = self.GRID_HEIGHT // 2
        self.steps = 0
        self.score = 0
        self.energy = 0
        self.game_over = False
        
        self.match_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if space_pressed:
            self.selected_row = (self.selected_row + 1) % self.GRID_HEIGHT
            # sfx: select_row_sound()
        if shift_pressed:
            self.selected_row = (self.selected_row - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
            # sfx: select_row_sound()
        
        shift_direction = 0
        if movement == 3: # Left
            shift_direction = -1
        elif movement == 4: # Right
            shift_direction = 1

        # --- 2. Calculate Pre-Action State for Rewards ---
        pre_action_matches_2x2 = self._find_matches(2)
        pre_action_matches_3x3 = self._find_matches(3)

        # --- 3. Update Game State ---
        if shift_direction != 0:
            # Shift the logical grid
            self.grid_colors[self.selected_row] = np.roll(self.grid_colors[self.selected_row], shift_direction)
            # Add visual offset for smooth animation
            self.visual_row_offsets[self.selected_row] -= shift_direction * self.CELL_SIZE
            # sfx: row_shift_sound()

        # Update colors periodically
        if self.steps > 0 and self.steps % self.COLOR_CHANGE_INTERVAL == 0:
            # Randomly change a 2x2 block of colors to keep the board fresh
            row = self.np_random.integers(0, self.GRID_HEIGHT - 1)
            col = self.np_random.integers(0, self.GRID_WIDTH - 1)
            for r_offset in range(2):
                for c_offset in range(2):
                    self.grid_colors[row + r_offset, col + c_offset] = self.np_random.integers(0, len(self.COLORS))
        
        # --- 4. Calculate Post-Action State and Rewards ---
        reward = 0
        
        # Partial match reward
        post_action_matches_2x2 = self._find_matches(2)
        new_2x2_matches = len(post_action_matches_2x2 - pre_action_matches_2x2)
        reward += new_2x2_matches * 0.1

        # Full match reward
        post_action_matches_3x3 = self._find_matches(3)
        new_3x3_matches = post_action_matches_3x3 - pre_action_matches_3x3
        if new_3x3_matches:
            # sfx: match_success_sound()
            for match in new_3x3_matches:
                reward += 10
                self.energy += 1
                self._add_match_effect(match[0], match[1])

        self.score += reward

        # --- 5. Check Termination Conditions ---
        time_is_up = self.steps >= self.TIMER_MAX_STEPS
        has_won = self.energy >= self.WIN_ENERGY
        hard_cap_reached = self.steps >= self.MAX_STEPS
        
        terminated = time_is_up or has_won or hard_cap_reached
        truncated = False # This game has a time limit, but termination handles it.
        if terminated and not self.game_over:
            self.game_over = True
            if has_won and not time_is_up:
                reward += 100 # Win bonus
                # sfx: game_win_sound()
            else:
                reward -= 100 # Lose penalty
                # sfx: game_lose_sound()
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _find_matches(self, size):
        matches = set()
        for r in range(self.GRID_HEIGHT - size + 1):
            for c in range(self.GRID_WIDTH - size + 1):
                sub_grid = self.grid_colors[r:r+size, c:c+size]
                first_color = sub_grid[0, 0]
                if np.all(sub_grid == first_color):
                    matches.add((r, c, first_color))
        return matches

    def _add_match_effect(self, r, c):
        center_x = self.GRID_X + (c + 1.5) * self.CELL_SIZE
        center_y = self.GRID_Y + (r + 1.5) * self.CELL_SIZE
        color = self.COLORS[self.grid_colors[r, c]]
        self.match_effects.append({
            "pos": (center_x, center_y),
            "color": color,
            "lifetime": 30, # frames
            "max_lifetime": 30,
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Animate visual offsets back to zero
        self.visual_row_offsets *= 0.8
        self.visual_row_offsets[np.abs(self.visual_row_offsets) < 0.1] = 0

        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, y), 1)
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE), 1)

        # Draw lights
        for r in range(self.GRID_HEIGHT):
            row_offset = self.visual_row_offsets[r]
            for c in range(self.GRID_WIDTH):
                color_index = self.grid_colors[r, c]
                color = self.COLORS[color_index]
                
                # Handle wrapping for smooth animation
                for offset_mult in [-1, 0, 1]:
                    base_x = self.GRID_X + c * self.CELL_SIZE
                    x = base_x + row_offset + offset_mult * self.GRID_WIDTH * self.CELL_SIZE
                    y = self.GRID_Y + r * self.CELL_SIZE

                    if self.GRID_X <= x + self.CELL_SIZE and x <= self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE:
                        # Glow effect
                        glow_color = (*color, 60)
                        glow_surf = pygame.Surface((self.CELL_SIZE * 2, self.CELL_SIZE * 2), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, glow_color, (self.CELL_SIZE, self.CELL_SIZE), self.CELL_SIZE * 0.7)
                        self.screen.blit(glow_surf, (int(x - self.CELL_SIZE/2), int(y - self.CELL_SIZE/2)), special_flags=pygame.BLEND_RGBA_ADD)
                        
                        # Main square
                        rect = pygame.Rect(int(x) + 2, int(y) + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
                        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw selector
        pulse = 1 + 0.1 * math.sin(self.steps * 0.2)
        selector_y = self.GRID_Y + self.selected_row * self.CELL_SIZE
        selector_rect = pygame.Rect(
            self.GRID_X - 4, 
            selector_y - 4, 
            self.GRID_WIDTH * self.CELL_SIZE + 8, 
            self.CELL_SIZE + 8
        )
        # Draw multiple times for a thicker line
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, int(3 * pulse), border_radius=8)
        pygame.draw.rect(self.screen, (*self.COLOR_SELECTOR, 100), selector_rect, int(6 * pulse), border_radius=10)


    def _render_effects(self):
        for effect in self.match_effects[:]:
            effect["lifetime"] -= 1
            if effect["lifetime"] <= 0:
                self.match_effects.remove(effect)
                continue

            progress = 1.0 - (effect["lifetime"] / effect["max_lifetime"])
            radius = (1.5 * self.CELL_SIZE) * math.sqrt(progress) # Sqrt for ease-out
            alpha = 255 * (1.0 - progress)

            pos = (int(effect["pos"][0]), int(effect["pos"][1]))
            color = (*effect["color"], int(alpha))

            # Use a temporary surface for alpha blending
            effect_surf = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
            pygame.draw.circle(effect_surf, color, (int(radius), int(radius)), int(radius))
            self.screen.blit(effect_surf, (int(pos[0] - radius), int(pos[1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Energy Counter
        energy_text = self.font_large.render(f"ENERGY: {self.energy} / {self.WIN_ENERGY}", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (20, 10))

        # Timer
        time_remaining = max(0, self.TIMER_MAX_STEPS - self.steps)
        time_seconds = time_remaining / self.FPS
        timer_text = self.font_large.render(f"TIME: {time_seconds:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 20, 10))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0,0))

            if self.energy >= self.WIN_ENERGY and self.steps < self.TIMER_MAX_STEPS:
                msg = "VICTORY!"
                msg_color = (180, 255, 180)
            else:
                msg = "TIME UP"
                msg_color = (255, 180, 180)
            
            end_text = self.font_large.render(msg, True, msg_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "time_remaining_steps": max(0, self.TIMER_MAX_STEPS - self.steps)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment for manual play
if __name__ == '__main__':
    # The validation code has been removed as it's not part of the required game logic
    # and was for development purposes. It also causes issues when run outside a
    # specific test harness.

    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Color Grid Shift")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Reset action
            action = [0, 0, 0] # No-op, released, released
            
            # Movement (for shifting rows)
            # The brief does not map up/down to anything, but we handle it.
            if keys[pygame.K_UP]:
                action[0] = 1 
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Buttons (for selecting rows)
            if keys[pygame.K_SPACE]:
                action[1] = 1 # Held
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1 # Held

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(GameEnv.FPS)

    env.close()