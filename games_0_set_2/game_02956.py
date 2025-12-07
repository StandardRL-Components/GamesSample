
# Generated: 2025-08-27T21:56:19.235779
# Source Brief: brief_02956.md
# Brief Index: 2956

        
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
        "Use arrow keys to move the selected crystal. Press SPACE to cycle selection, SHIFT to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A crystal maze puzzle. Place crystals next to paths to light them up. Light all paths before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.MAX_MOVES = 10

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PATH_UNLIT = (60, 70, 90)
        self.COLOR_PATH_LIT_GLOW = (150, 255, 255)
        self.COLOR_PATH_LIT_CORE = (220, 255, 255)
        self.COLOR_CRYSTALS = [
            (255, 80, 120),  # Hot Pink
            (80, 255, 150),  # Mint Green
            (255, 200, 80),  # Gold
        ]
        self.COLOR_SELECTION = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 15)
        self.COLOR_VICTORY = (180, 255, 180)
        self.COLOR_DEFEAT = (255, 180, 180)

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
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = 0
        self.crystals = []
        self.path_cells = set()
        self.lit_path_cells = set()
        self.selected_crystal_idx = None
        self.victory = False
        
        # For rising-edge detection of actions
        self.space_was_held = False
        self.shift_was_held = False

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.remaining_moves = self.MAX_MOVES
        self.selected_crystal_idx = None
        self.space_was_held = False
        self.shift_was_held = False

        # --- Define Maze Layout ---
        self.path_cells = {
            (2, 2), (3, 2), (4, 2), (5, 2),
            (5, 3), (5, 4), (5, 5), (5, 6),
            (4, 6), (3, 6), (2, 6),
            (8, 4), (9, 4), (10, 4), (11, 4), (12, 4), (13, 4)
        }
        
        # --- Define Crystal Positions ---
        self.crystals = [
            {'pos': np.array([1, 8]), 'color': self.COLOR_CRYSTALS[0]},
            {'pos': np.array([7, 1]), 'color': self.COLOR_CRYSTALS[1]},
            {'pos': np.array([14, 8]), 'color': self.COLOR_CRYSTALS[2]},
        ]
        
        self._update_lit_paths()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action and detect rising edge for space/shift
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        
        self.space_was_held = space_held
        self.shift_was_held = shift_held
        
        action_taken = False

        # --- Handle Actions ---
        if space_pressed:
            if self.selected_crystal_idx is None:
                self.selected_crystal_idx = 0
            else:
                self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystals)
            # sfx_select_crystal()
        
        if shift_pressed:
            self.selected_crystal_idx = None
            # sfx_deselect_crystal()

        if movement != 0 and self.selected_crystal_idx is not None:
            action_taken = True
            
            crystal = self.crystals[self.selected_crystal_idx]
            current_pos = crystal['pos']
            target_pos = current_pos.copy()

            if movement == 1: target_pos[1] -= 1
            elif movement == 2: target_pos[1] += 1
            elif movement == 3: target_pos[0] -= 1
            elif movement == 4: target_pos[0] += 1
            
            is_valid_move = True
            if not (0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H):
                is_valid_move = False
            if is_valid_move:
                for i, other_crystal in enumerate(self.crystals):
                    if i != self.selected_crystal_idx and np.array_equal(target_pos, other_crystal['pos']):
                        is_valid_move = False
                        break
            
            if is_valid_move:
                # sfx_move_crystal()
                paths_before = self.lit_path_cells.copy()
                crystal['pos'] = target_pos
                self._update_lit_paths()
                paths_after = self.lit_path_cells

                newly_lit = paths_after - paths_before
                newly_unlit = paths_before - paths_after
                
                reward += len(paths_after) * 1.0
                reward += len(newly_lit) * 5.0
                reward -= len(newly_unlit) * 0.1
            else:
                # sfx_invalid_move()
                reward -= 1.0

        if action_taken:
            self.remaining_moves -= 1
        
        self.steps += 1
        self.score += reward

        # --- Check Termination Conditions ---
        all_paths_lit = len(self.lit_path_cells) == len(self.path_cells)
        out_of_moves = self.remaining_moves <= 0
        terminated = False

        if all_paths_lit:
            # sfx_victory()
            terminated = True
            self.victory = True
            reward += 50.0
            self.score += 50.0
        elif out_of_moves:
            # sfx_defeat()
            terminated = True
            self.victory = False
            reward -= 50.0
            self.score -= 50.0
            
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _update_lit_paths(self):
        self.lit_path_cells.clear()
        for px, py in self.path_cells:
            is_lit = False
            for crystal in self.crystals:
                cx, cy = crystal['pos']
                if abs(cx - px) <= 1 and abs(cy - py) <= 1:
                    is_lit = True
                    break
            if is_lit:
                self.lit_path_cells.add((px, py))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "lit_paths": len(self.lit_path_cells),
            "total_paths": len(self.path_cells),
        }

    def _render_game(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        for px, py in self.path_cells:
            rect = pygame.Rect(px * self.GRID_SIZE, py * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            if (px, py) in self.lit_path_cells:
                glow_rect = rect.inflate(self.GRID_SIZE * 0.4, self.GRID_SIZE * 0.4)
                pygame.draw.rect(self.screen, self.COLOR_PATH_LIT_GLOW, glow_rect, border_radius=int(self.GRID_SIZE * 0.4))
                core_rect = rect.inflate(self.GRID_SIZE * 0.1, self.GRID_SIZE * 0.1)
                pygame.draw.rect(self.screen, self.COLOR_PATH_LIT_CORE, core_rect, border_radius=int(self.GRID_SIZE * 0.3))
            else:
                pygame.draw.rect(self.screen, self.COLOR_PATH_UNLIT, rect.inflate(-4, -4), border_radius=int(self.GRID_SIZE * 0.2))

        for i, crystal in enumerate(self.crystals):
            center_x = int((crystal['pos'][0] + 0.5) * self.GRID_SIZE)
            center_y = int((crystal['pos'][1] + 0.5) * self.GRID_SIZE)
            radius = int(self.GRID_SIZE * 0.35)

            if i == self.selected_crystal_idx:
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius + 5, self.COLOR_SELECTION)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + 5, self.COLOR_SELECTION)

            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, crystal['color'])
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, tuple(int(c * 0.8) for c in crystal['color']))

    def _render_text(self, text, font, color, pos, shadow_color=None, shadow_offset=(2, 2)):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        moves_text = f"Moves: {self.remaining_moves}"
        score_text = f"Score: {int(self.score)}"
        paths_text = f"Lit: {len(self.lit_path_cells)}/{len(self.path_cells)}"

        self._render_text(moves_text, self.font_medium, self.COLOR_TEXT, (15, 15), self.COLOR_TEXT_SHADOW)
        self._render_text(score_text, self.font_medium, self.COLOR_TEXT, (15, 50), self.COLOR_TEXT_SHADOW)
        
        paths_size = self.font_medium.size(paths_text)
        self._render_text(paths_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH - paths_size[0] - 15, 15), self.COLOR_TEXT_SHADOW)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.victory else "OUT OF MOVES"
            color = self.COLOR_VICTORY if self.victory else self.COLOR_DEFEAT
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    while not done:
        movement, space_held, shift_held = 0, 0, 0
        action_triggered = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                 if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                     action_triggered = True

        if action_triggered:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    pygame.quit()