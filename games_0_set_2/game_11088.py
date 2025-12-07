import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:25:37.224738
# Source Brief: brief_01088.md
# Brief Index: 1088
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a geometric puzzle game.
    The goal is to recursively divide the screen to create the maximum number
    of equal-sized regions within a limited number of divisions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A geometric puzzle game where you divide the screen into regions. "
        "Aim to create the largest possible number of equal-sized areas within a limited number of cuts."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to make a vertical cut and shift to make a horizontal cut."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_DIVISIONS = 5
    CURSOR_SPEED = 10
    LINE_WIDTH = 3

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_LINE = (255, 255, 255)
    COLOR_CURSOR = (255, 0, 255)
    COLOR_HIGHLIGHT = (0, 255, 128)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_GAMEOVER_FAIL = (255, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gym Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.divisions = None
        self.score = None
        self.game_over = None
        self.cursor_pos = None
        self.regions = None
        self.lines = None
        self.flash_alpha = 0

        # Per the brief, reset() is called externally after init
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.divisions = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)

        initial_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.regions = [{
            'rect': initial_rect,
            'area': initial_rect.width * initial_rect.height,
            'color': self._get_random_region_color()
        }]

        self.lines = []
        self.flash_alpha = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.steps += 1

        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Update flash effect ---
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 25)

        # --- Handle cursor movement ---
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)

        # --- Handle division action ---
        can_divide = self.divisions < self.MAX_DIVISIONS
        division_action = space_held or shift_held

        if can_divide and division_action:
            target_region_idx, target_region = self._find_region_at_cursor()

            if target_region:
                new_line, new_regions = None, None
                
                # Prioritize vertical split if both are held
                if space_held:
                    new_line, new_regions = self._split_region(target_region, 'vertical')
                elif shift_held:
                    new_line, new_regions = self._split_region(target_region, 'horizontal')
                
                if new_line and new_regions:
                    # SFX: Play "slice" or "thump" sound
                    self.divisions += 1
                    self.flash_alpha = 150

                    self.lines.append(new_line)
                    self.regions.pop(target_region_idx)
                    self.regions.extend(new_regions)
                    
                    step_reward = 1.0  # Base reward for a successful division
                    if new_regions[0]['area'] == new_regions[1]['area']:
                        step_reward += 5.0
                    
                    reward += step_reward
                    self.score += step_reward

        # --- Check for termination ---
        terminated = self.divisions >= self.MAX_DIVISIONS
        if terminated and not self.game_over:
            self.game_over = True
            largest_group_size = self._get_largest_equal_group_size()
            if largest_group_size >= 10:
                terminal_reward = 50.0
                reward += terminal_reward
                self.score += terminal_reward
                # SFX: Play "victory" sound

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_region_at_cursor(self):
        for i, region in enumerate(self.regions):
            if region['rect'].collidepoint(self.cursor_pos):
                return i, region
        return None, None

    def _split_region(self, region_data, orientation):
        r = region_data['rect']
        cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])

        if orientation == 'vertical' and r.left < cx < r.right:
            line = (cx, r.top, cx, r.bottom)
            r1 = pygame.Rect(r.left, r.top, cx - r.left, r.height)
            r2 = pygame.Rect(cx, r.top, r.right - cx, r.height)
        elif orientation == 'horizontal' and r.top < cy < r.bottom:
            line = (r.left, cy, r.right, cy)
            r1 = pygame.Rect(r.left, r.top, r.width, cy - r.top)
            r2 = pygame.Rect(r.left, cy, r.width, r.bottom - cy)
        else:
            return None, None

        new_regions = [
            {'rect': r1, 'area': r1.width * r1.height, 'color': self._get_random_region_color()},
            {'rect': r2, 'area': r2.width * r2.height, 'color': self._get_random_region_color()}
        ]
        return line, new_regions

    def _get_largest_equal_group_size(self):
        if not self.regions: return 0
        areas = defaultdict(int)
        for region in self.regions:
            areas[region['area']] += 1
        return max(areas.values()) if areas else 0

    def _find_equal_sized_regions_map(self):
        areas = defaultdict(int)
        for region in self.regions:
            areas[region['area']] += 1
        return {area: count for area, count in areas.items() if count > 1}

    def _get_random_region_color(self):
        # Generate visually pleasing, desaturated colors using HSV
        hue = self.np_random.uniform(0, 360)
        saturation = self.np_random.uniform(0.2, 0.4)
        value = self.np_random.uniform(0.4, 0.6)
        
        color = pygame.Color(0)
        color.hsva = (hue, saturation * 100, value * 100, 100)
        return (color.r, color.g, color.b)

    def _render_game(self):
        # --- Render Regions ---
        equal_areas = self._find_equal_sized_regions_map()
        largest_equal_group_area = 0
        if equal_areas:
            largest_equal_group_area = max(equal_areas, key=lambda k: equal_areas[k])
        
        for region in self.regions:
            color = region['color']
            if region['area'] == largest_equal_group_area and equal_areas.get(region['area'], 0) > 1:
                color = self.COLOR_HIGHLIGHT
            pygame.draw.rect(self.screen, color, region['rect'])

        # --- Render Lines ---
        for line in self.lines:
            pygame.draw.line(self.screen, self.COLOR_LINE, (line[0], line[1]), (line[2], line[3]), self.LINE_WIDTH)

        # --- Render Cursor Preview ---
        if not self.game_over:
            _, target_region = self._find_region_at_cursor()
            if target_region:
                r = target_region['rect']
                cx, cy = int(self.cursor_pos[0]), int(self.cursor_pos[1])
                
                pygame.gfxdraw.vline(self.screen, cx, r.top, r.bottom, self.COLOR_CURSOR)
                pygame.gfxdraw.hline(self.screen, r.left, r.right, cy, self.COLOR_CURSOR)
                
                pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cx, cy), 6, 2)

    def _render_text(self, text, font, position, color, shadow_color, center=False):
        text_surf = font.render(str(text), True, color)
        shadow_surf = font.render(str(text), True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = position
        else:
            text_rect.topleft = position

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"Score: {int(self.score)}", self.font_small, (10, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Divisions
        div_text = f"Divisions: {self.divisions}/{self.MAX_DIVISIONS}"
        text_surf = self.font_small.render(div_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self._render_text(div_text, self.font_small, text_rect.topleft, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over Message
        if self.game_over:
            largest_group_size = self._get_largest_equal_group_size()
            win = largest_group_size >= 10
            
            end_text = "SUCCESS!" if win else "GAME OVER"
            end_color = self.COLOR_HIGHLIGHT if win else self.COLOR_GAMEOVER_FAIL
            center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
            
            self._render_text(end_text, self.font_large, (center_x, center_y - 30), end_color, self.COLOR_TEXT_SHADOW, center=True)
            
            sub_text = f"Largest group of equal regions: {largest_group_size}"
            self._render_text(sub_text, self.font_small, (center_x, center_y + 20), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

        # --- Render Flash Effect ---
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, self.flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

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
            "divisions": self.divisions,
            "regions": len(self.regions),
            "largest_equal_group": self._get_largest_equal_group_size()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Interactive Play Mode ---
    env = GameEnv()
    env.reset()
    # env.validate_implementation() # Validate after reset # Commented out as not needed for interactive play

    # Un-dummy the video driver for interactive play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()

    running = True
    terminated = False
    
    # Use a separate screen for display to avoid conflicts with env's internal surface
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Geometric Division Puzzle")
    clock = pygame.time.Clock()

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated:
            # On game over, wait for a key press to reset
            keys = pygame.key.get_pressed()
            if any(keys):
                obs, info = env.reset()
                terminated = False
                continue
        else:
            # --- Action Mapping for Human Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Divisions: {info['divisions']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()