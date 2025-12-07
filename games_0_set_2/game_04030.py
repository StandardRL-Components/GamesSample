
# Generated: 2025-08-28T01:10:50.381829
# Source Brief: brief_04030.md
# Brief Index: 4030

        
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

    user_guide = (
        "Controls: Use Space/Shift to cycle crystal selection. Use ↑↓←→ to push the selected crystal. "
        "Light up all paths to win."
    )

    game_description = (
        "A turn-based puzzle game. Push colored crystals onto pressure plates to connect and illuminate paths. "
        "Solve each level within the move limit to advance."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        self._define_colors_and_fonts()
        self._define_levels()
        
        self.reset()
        
        self.validate_implementation()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_PLATE_OFF = (50, 50, 60)
        self.COLOR_PLATE_ON = (220, 220, 255)
        self.COLOR_PATH_OFF = (45, 50, 70)
        self.COLOR_PATH_LIT = (255, 255, 255)
        self.COLOR_PATH_GLOW = (150, 200, 255)
        self.COLOR_TEXT = (230, 230, 240)
        self.CRYSTAL_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 150, 255),
            (255, 255, 80), (200, 80, 255), (80, 255, 200)
        ]
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_LEVEL = pygame.font.Font(None, 32)

    def _define_levels(self):
        self.LEVELS = [
            {
                "grid_size": (10, 8), "moves": 25,
                "crystals": [(2, 2), (2, 5)],
                "plates": [(7, 2), (7, 5)],
                "paths": [(0, 1)]
            },
            {
                "grid_size": (12, 10), "moves": 50,
                "crystals": [(2, 2), (5, 4), (2, 7)],
                "plates": [(9, 2), (5, 7), (9, 7)],
                "paths": [(0, 2), (1, 2)]
            },
            {
                "grid_size": (16, 12), "moves": 75,
                "crystals": [(2, 2), (5, 2), (10, 5), (2, 9), (13, 9)],
                "plates": [(8, 2), (2, 5), (13, 5), (8, 9), (5, 9)],
                "paths": [(0, 2), (1, 2), (3, 4), (0, 3), (1, 4)]
            }
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_level = 0
        self.total_score = 0
        self.episode_steps = 0
        self.game_over = False
        
        self._load_level(self.current_level)
        
        return self._get_observation(), self._get_info()

    def _load_level(self, level_idx):
        level_data = self.LEVELS[level_idx]
        self.grid_w, self.grid_h = level_data["grid_size"]
        self.moves_remaining = level_data["moves"]
        
        self.crystal_pos = [list(pos) for pos in level_data["crystals"]]
        self.plate_pos = level_data["plates"]
        self.path_defs = level_data["paths"]
        
        self.num_crystals = len(self.crystal_pos)
        self.selected_crystal_idx = 0
        
        self.plates_on = [False] * len(self.plate_pos)
        self.lit_paths = [False] * len(self.path_defs)
        
        self._update_game_state()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.episode_steps += 1
        reward = 0
        terminated = False
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle selection change
        if space_pressed and not shift_pressed:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % self.num_crystals
        elif shift_pressed and not space_pressed:
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + self.num_crystals) % self.num_crystals
        
        # 2. Handle push action (a "move")
        if movement != 0:
            self.moves_remaining -= 1
            old_lit_paths = sum(self.lit_paths)
            
            # sfx: Crystal push sound
            self._push_crystal(self.selected_crystal_idx, movement)
            
            newly_lit_paths = sum(self.lit_paths) - old_lit_paths
            
            if newly_lit_paths > 0:
                reward += 1.0 * newly_lit_paths
                # sfx: Path activation success chime
            else:
                reward -= 0.1 # Penalty for a non-productive move
        
        # 3. Check for win/loss conditions
        if all(self.lit_paths):
            if self.current_level < len(self.LEVELS) - 1:
                reward += 5.0 # Level complete bonus
                self.total_score += reward
                self.current_level += 1
                self._load_level(self.current_level)
                # sfx: Level complete fanfare
            else:
                reward += 50.0 # Final level complete bonus
                self.game_over = True
                terminated = True
                # sfx: Game win celebration
        elif self.moves_remaining <= 0:
            self.game_over = True
            terminated = True
            # sfx: Game over failure sound

        if self.episode_steps >= 300:
             self.game_over = True
             terminated = True

        self.total_score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _push_crystal(self, crystal_idx, direction):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[direction]
        
        start_pos = self.crystal_pos[crystal_idx]
        current_pos = list(start_pos)
        
        while True:
            next_pos = [current_pos[0] + dx, current_pos[1] + dy]
            
            # Check wall boundaries
            if not (0 <= next_pos[0] < self.grid_w and 0 <= next_pos[1] < self.grid_h):
                break
            
            # Check other crystal collisions
            is_blocked = False
            for i, pos in enumerate(self.crystal_pos):
                if i != crystal_idx and pos[0] == next_pos[0] and pos[1] == next_pos[1]:
                    is_blocked = True
                    break
            if is_blocked:
                break
                
            current_pos = next_pos

        self.crystal_pos[crystal_idx] = current_pos
        self._update_game_state()

    def _update_game_state(self):
        # Update which plates are on
        self.plates_on = [False] * len(self.plate_pos)
        for i, p_pos in enumerate(self.plate_pos):
            for c_pos in self.crystal_pos:
                if c_pos[0] == p_pos[0] and c_pos[1] == p_pos[1]:
                    self.plates_on[i] = True
                    break
    
        # Update which paths are lit
        for i, path in enumerate(self.path_defs):
            plate1_idx, plate2_idx = path
            if self.plates_on[plate1_idx] and self.plates_on[plate2_idx]:
                self.lit_paths[i] = True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.episode_steps,
            "level": self.current_level + 1,
            "moves_remaining": self.moves_remaining,
            "paths_lit": f"{sum(self.lit_paths)}/{len(self.lit_paths)}"
        }

    def _render_game(self):
        # Calculate grid scaling and offsets
        self.cell_size = min(self.width // (self.grid_w + 2), self.height // (self.grid_h + 2))
        offset_x = (self.width - self.grid_w * self.cell_size) / 2
        offset_y = (self.height - self.grid_h * self.cell_size) / 2

        def to_screen_pos(grid_x, grid_y, center=False):
            x = offset_x + grid_x * self.cell_size
            y = offset_y + grid_y * self.cell_size
            if center:
                return int(x + self.cell_size / 2), int(y + self.cell_size / 2)
            return int(x), int(y)

        # Draw faint grid lines
        for i in range(self.grid_w + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, to_screen_pos(i, 0), to_screen_pos(i, self.grid_h))
        for i in range(self.grid_h + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, to_screen_pos(0, i), to_screen_pos(self.grid_w, i))

        # Draw paths
        for i, path in enumerate(self.path_defs):
            p1_idx, p2_idx = path
            start_pos = to_screen_pos(self.plate_pos[p1_idx][0], self.plate_pos[p1_idx][1], center=True)
            end_pos = to_screen_pos(self.plate_pos[p2_idx][0], self.plate_pos[p2_idx][1], center=True)
            
            if self.lit_paths[i]:
                # Glow effect
                pygame.draw.line(self.screen, self.COLOR_PATH_GLOW, start_pos, end_pos, self.cell_size // 4)
                pygame.draw.line(self.screen, self.COLOR_PATH_LIT, start_pos, end_pos, self.cell_size // 8)
            else:
                pygame.draw.line(self.screen, self.COLOR_PATH_OFF, start_pos, end_pos, self.cell_size // 10)

        # Draw pressure plates
        plate_radius = int(self.cell_size * 0.35)
        for i, pos in enumerate(self.plate_pos):
            center = to_screen_pos(pos[0], pos[1], center=True)
            color = self.COLOR_PLATE_ON if self.plates_on[i] else self.COLOR_PLATE_OFF
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], plate_radius, color)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], plate_radius, self.COLOR_WALL)

        # Draw crystals
        crystal_size = int(self.cell_size * 0.8)
        margin = (self.cell_size - crystal_size) // 2
        for i, pos in enumerate(self.crystal_pos):
            rect = pygame.Rect(to_screen_pos(pos[0], pos[1]), (crystal_size, crystal_size))
            rect.move_ip(margin, margin)
            
            color = self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)]
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Draw selector
            if i == self.selected_crystal_idx and not self.game_over:
                pulse = (math.sin(self.episode_steps * 0.2) + 1) / 2 # 0 to 1
                glow_size = int(crystal_size * (1.1 + pulse * 0.2))
                glow_alpha = int(100 + pulse * 100)
                
                glow_color = (*self.COLOR_PATH_LIT, glow_alpha)
                
                glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
                
                glow_rect = glow_surf.get_rect(center=rect.center)
                self.screen.blit(glow_surf, glow_rect)
        
        # Draw walls (border of the grid)
        wall_rect = pygame.Rect(offset_x, offset_y, self.grid_w * self.cell_size, self.grid_h * self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect, 3)

    def _render_ui(self):
        moves_text = self.FONT_UI.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        level_text = self.FONT_LEVEL.render(f"Level: {self.current_level + 1}", True, self.COLOR_TEXT)
        level_rect = level_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(level_text, level_rect)
        
        score_text = self.FONT_UI.render(f"Score: {self.total_score:.1f}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midbottom=(self.width / 2, self.height - 10))
        self.screen.blit(score_text, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if all(self.lit_paths):
                end_text_str = "CONGRATULATIONS!"
            else:
                end_text_str = "GAME OVER"
                
            end_text = pygame.font.Font(None, 60).render(end_text_str, True, self.COLOR_PATH_LIT)
            end_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Crystal Caverns")
    
    terminated = False
    
    # Game loop
    while not terminated:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
                elif event.key == pygame.K_r: # Reset key
                    obs, info = env.reset()
        
        # Only step if an action was taken
        if any(a != 0 for a in action):
             obs, reward, terminated, truncated, info = env.step(action)
             print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over! Final Score:", info['score'])
            # Wait for a moment before closing or allow reset
            pygame.time.wait(3000)
            # You could add a reset prompt here
            
    env.close()