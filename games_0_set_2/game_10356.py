import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls the gravity of falling blocks
    to keep them balanced within a shrinking circular arena.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Balance a stack of blocks within a shrinking circular arena by adjusting each block's individual gravity."
    )
    user_guide = (
        "Controls: Use ↑/↓ to select a block. Press space to increase its gravity or shift to decrease it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_BLOCKS = 6
    BLOCK_WIDTH = 80
    BLOCK_HEIGHT = 20
    MAX_STEPS = 5000
    GRAVITY_MIN = 1.0
    GRAVITY_MAX = 5.0
    PHYSICS_TIMESTEP = 0.05

    # --- Colors ---
    COLOR_BG = (15, 15, 25)
    COLOR_ARENA = (200, 200, 255)
    COLOR_ARENA_GLOW = (50, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECTED_OUTLINE = (255, 255, 255)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 12)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        
        self.arena_center = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.arena_initial_radius = (self.SCREEN_HEIGHT / 2) * 0.9
        self.arena_final_radius = self.arena_initial_radius * 0.2
        self.arena_radius = self.arena_initial_radius
        self.arena_shrink_rate = (self.arena_initial_radius - self.arena_final_radius) / self.MAX_STEPS
        self.last_arena_percent = 100

        self.blocks = []
        self.selected_block_idx = 0
        self.last_action = np.array([0, 0, 0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False

        self.arena_radius = self.arena_initial_radius
        self.last_arena_percent = 100
        
        self.selected_block_idx = 0
        self.last_action = np.array([0, 0, 0])

        self.blocks = []
        start_y = self.SCREEN_HEIGHT * 0.2
        for i in range(self.NUM_BLOCKS):
            block = {
                "pos": pygame.Vector2(self.SCREEN_WIDTH / 2, start_y + i * (self.BLOCK_HEIGHT + 5)),
                "vel": pygame.Vector2(0, 0),
                "gravity": self.GRAVITY_MIN,  # Set to a constant to ensure initial stability
                "color": self.BLOCK_COLORS[i],
                "rect": pygame.Rect(0, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
            }
            block["rect"].center = block["pos"]
            self.blocks.append(block)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self._handle_actions(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        self.game_over = terminated or truncated
        
        self.steps += 1
        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_press, shift_press = action[0], action[1], action[2]
        
        # --- Block Selection ---
        # Only trigger on change to avoid rapid cycling
        if movement != self.last_action[0]:
            if movement == 1:  # Up
                self.selected_block_idx = (self.selected_block_idx - 1) % self.NUM_BLOCKS
            elif movement == 2:  # Down
                self.selected_block_idx = (self.selected_block_idx + 1) % self.NUM_BLOCKS

        # --- Gravity Adjustment ---
        # Trigger on rising edge (0 -> 1)
        if space_press == 1 and self.last_action[1] == 0:
            block = self.blocks[self.selected_block_idx]
            block["gravity"] = min(self.GRAVITY_MAX, block["gravity"] + 1.0)
        
        if shift_press == 1 and self.last_action[2] == 0:
            block = self.blocks[self.selected_block_idx]
            block["gravity"] = max(self.GRAVITY_MIN, block["gravity"] - 1.0)

    def _update_game_state(self):
        # --- Arena Shrinking ---
        self.arena_radius = max(self.arena_final_radius, self.arena_radius - self.arena_shrink_rate)
        
        # --- Physics Update ---
        for block in self.blocks:
            block["vel"].y += block["gravity"] * self.PHYSICS_TIMESTEP
            block["pos"] += block["vel"] * self.PHYSICS_TIMESTEP * 10 # Scale for visibility
            block["rect"].center = block["pos"]
        
        # --- Collision Detection ---
        # Block-Arena Collision
        for block in self.blocks:
            for corner in [block["rect"].topleft, block["rect"].topright, block["rect"].bottomleft, block["rect"].bottomright]:
                if self.arena_center.distance_to(corner) > self.arena_radius:
                    self.game_over = True
                    return

        # Block-Block Collision
        for i in range(self.NUM_BLOCKS):
            for j in range(i + 1, self.NUM_BLOCKS):
                if self.blocks[i]["rect"].colliderect(self.blocks[j]["rect"]):
                    # Stacking is allowed ONLY between adjacent blocks (i and i+1)
                    if j != i + 1:
                        self.game_over = True
                        return
        
        # --- Win Condition ---
        if self.arena_radius <= self.arena_final_radius and not self.game_over:
            self.win = True
            self.game_over = True

    def _calculate_reward(self):
        if self.game_over:
            if self.win:
                return 100.0
            else:
                return -100.0
        
        reward = 0.0
        
        # Survival reward
        reward += 0.1
        
        # Arena shrink reward
        current_percent = int(100 * (self.arena_initial_radius - self.arena_radius) / (self.arena_initial_radius - self.arena_final_radius))
        if current_percent > self.last_arena_percent:
            reward += 1.0 * (current_percent - self.last_arena_percent)
            self.last_arena_percent = current_percent
            
        return reward

    def _check_termination(self):
        # Termination is caused by game_over events (collisions, wins)
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Arena ---
        arena_pos = (int(self.arena_center.x), int(self.arena_center.y))
        arena_rad = int(self.arena_radius)
        
        # Glow effect
        glow_rad = int(arena_rad * 1.05)
        pygame.gfxdraw.filled_circle(self.screen, arena_pos[0], arena_pos[1], glow_rad, self.COLOR_ARENA_GLOW)
        
        # Main line (anti-aliased)
        pygame.gfxdraw.aacircle(self.screen, arena_pos[0], arena_pos[1], arena_rad, self.COLOR_ARENA)
        pygame.gfxdraw.aacircle(self.screen, arena_pos[0], arena_pos[1], arena_rad-1, self.COLOR_ARENA)

        # --- Blocks ---
        for i, block in enumerate(self.blocks):
            # Outline/glow for selected block
            if i == self.selected_block_idx:
                outline_rect = block["rect"].inflate(8, 8)
                pygame.draw.rect(self.screen, self.COLOR_SELECTED_OUTLINE, outline_rect, border_radius=5)
            
            # Main block
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            
            # Gravity text
            grav_text = f"{block['gravity']:.1f}"
            text_surf = self.font_small.render(grav_text, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(block["pos"].x, block["pos"].y - self.BLOCK_HEIGHT))
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # --- Score ---
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # --- Arena Size ---
        arena_percent = 100 * self.arena_radius / self.arena_initial_radius
        arena_text = f"Arena: {arena_percent:.0f}%"
        arena_surf = self.font_main.render(arena_text, True, self.COLOR_UI_TEXT)
        arena_rect = arena_surf.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(arena_surf, arena_rect)
        
        # --- Steps ---
        steps_text = f"Step: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_main.render(steps_text, True, self.COLOR_UI_TEXT)
        steps_rect = steps_surf.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(steps_surf, steps_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "arena_radius_percent": self.arena_radius / self.arena_initial_radius
        }

    def close(self):
        pygame.quit()