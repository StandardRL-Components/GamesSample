
# Generated: 2025-08-27T18:35:18.235422
# Source Brief: brief_01874.md
# Brief Index: 1874

        
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

    user_guide = "Controls: Use arrow keys to move the selector. Press space to rotate the selected tile."
    game_description = "A fast-paced puzzle game. Rotate the tiles to match the target pattern before the time runs out!"
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.fps = 30

        self.GRID_DIM = 4
        self.MAX_STAGES = 3
        self.TIME_PER_STAGE = 60.0
        
        self._define_colors_and_fonts()
        self._generate_patterns()

        self.reset()
        
        # This check is run once to ensure the implementation is correct.
        # self.validate_implementation()

    def _define_colors_and_fonts(self):
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_UI_BG = (30, 40, 50)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.TILE_COLORS = [
            (231, 76, 60),    # Red
            (46, 204, 113),   # Green
            (52, 152, 219),   # Blue
            (241, 196, 15),   # Yellow
            (155, 89, 182),   # Purple
            (26, 188, 156),   # Turquoise
        ]
        self.FONT_LARGE = pygame.font.Font(None, 48)
        self.FONT_MEDIUM = pygame.font.Font(None, 32)
        self.FONT_SMALL = pygame.font.Font(None, 24)

    def _generate_patterns(self):
        # Create a fixed color layout for the puzzle
        base_colors = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_DIM, self.GRID_DIM))
        self.grid_colors = np.array([[self.TILE_COLORS[i] for i in row] for row in base_colors])
        
        # Generate three distinct target patterns of orientations
        self.target_patterns = []
        for i in range(self.MAX_STAGES):
            # Create a pattern with increasing complexity (more non-zero rotations)
            pattern = self.np_random.integers(0, 4, size=(self.GRID_DIM, self.GRID_DIM))
            self.target_patterns.append(pattern)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = -1
        
        self.prev_space_held = False
        self.move_cooldown = 0
        self.animation_state = {}

        # Re-generate patterns if a new seed is provided
        self._generate_patterns()
        self._setup_stage(0)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        self.current_stage = stage_num
        self.time_remaining = self.TIME_PER_STAGE
        self.selector_pos = [self.GRID_DIM // 2, self.GRID_DIM // 2]
        self.animation_state = {}

        # Scramble the target pattern to create the initial state
        target = self.target_patterns[self.current_stage]
        # Difficulty scales with stage number
        scramble_amount = 5 + self.current_stage * 10
        self.grid_orientations = self._scramble_pattern(target, scramble_amount)

    def _scramble_pattern(self, pattern, num_scrambles):
        scrambled = np.copy(pattern)
        for _ in range(num_scrambles):
            x = self.np_random.integers(0, self.GRID_DIM)
            y = self.np_random.integers(0, self.GRID_DIM)
            scrambled[y, x] = (scrambled[y, x] + self.np_random.integers(1, 4)) % 4
        return scrambled

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False

        self.time_remaining -= 1.0 / self.fps
        self.steps += 1
        
        # Update animations
        self._update_animations()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        space_pressed = space_held and not self.prev_space_held
        
        reward += self._handle_input(movement, space_pressed)
        
        self.prev_space_held = space_held

        # Check for stage completion
        if self._check_completion():
            # sfx: stage_complete.wav
            reward += 10.0
            self.score += 10
            if self.current_stage + 1 >= self.MAX_STAGES:
                # sfx: game_win.wav
                reward += 50.0 # Final bonus
                self.score += 50
                self.game_over = True
                terminated = True
            else:
                self._setup_stage(self.current_stage + 1)
        
        # Check for timeout
        if self.time_remaining <= 0:
            # sfx: game_over.wav
            reward = -100.0 # Override other rewards with a large penalty
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_animations(self):
        # Update rotation animations
        keys_to_del = []
        for key, anim in self.animation_state.items():
            anim['progress'] -= 1.0 / (self.fps * 0.15) # 0.15 second animation
            if anim['progress'] <= 0:
                keys_to_del.append(key)
        for key in keys_to_del:
            del self.animation_state[key]

    def _handle_input(self, movement, space_pressed):
        reward = 0
        
        # Handle selector movement
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            # sfx: selector_move.wav
            if movement == 1: self.selector_pos[1] -= 1  # Up
            elif movement == 2: self.selector_pos[1] += 1  # Down
            elif movement == 3: self.selector_pos[0] -= 1  # Left
            elif movement == 4: self.selector_pos[0] += 1  # Right
            
            self.selector_pos[0] %= self.GRID_DIM
            self.selector_pos[1] %= self.GRID_DIM
            self.move_cooldown = 4 # frames

        # Handle tile rotation
        if space_pressed:
            sel_x, sel_y = self.selector_pos
            # Only rotate if not already animating
            if (sel_x, sel_y) not in self.animation_state:
                # sfx: tile_rotate.wav
                target_orientation = self.target_patterns[self.current_stage][sel_y, sel_x]
                current_orientation = self.grid_orientations[sel_y, sel_x]
                new_orientation = (current_orientation + 1) % 4

                is_correct_before = (current_orientation == target_orientation)
                is_correct_after = (new_orientation == target_orientation)

                if not is_correct_before and is_correct_after:
                    reward += 0.1
                    self.score += 0.1
                elif is_correct_before and not is_correct_after:
                    reward -= 0.01
                    self.score -= 0.01

                self.animation_state[(sel_x, sel_y)] = {
                    'from': current_orientation, 
                    'to': new_orientation, 
                    'progress': 1.0
                }
                self.grid_orientations[sel_y, sel_x] = new_orientation
        return reward

    def _check_completion(self):
        return np.array_equal(self.grid_orientations, self.target_patterns[self.current_stage])

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
            "stage": self.current_stage + 1,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        grid_pixel_size = self.GRID_DIM * 90
        offset_x = 40
        offset_y = (400 - grid_pixel_size) // 2 + 20

        # Draw grid background and lines
        pygame.draw.rect(self.screen, self.COLOR_GRID, (offset_x - 5, offset_y - 5, grid_pixel_size + 10, grid_pixel_size + 10), border_radius=5)

        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_x = offset_x + c * 90
                tile_y = offset_y + r * 90
                color = self.grid_colors[r, c]
                orientation = self.grid_orientations[r, c]
                
                anim = self.animation_state.get((c, r))
                self._render_tile(self.screen, tile_x, tile_y, color, orientation, anim)
        
        # Draw selector
        sel_x, sel_y = self.selector_pos
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 5
        selector_rect = pygame.Rect(offset_x + sel_x * 90 - pulse, offset_y + sel_y * 90 - pulse, 80 + 2 * pulse, 80 + 2 * pulse)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width=3, border_radius=8)

    def _render_tile(self, surface, x, y, color, orientation, anim_data):
        tile_size = 80
        rect = pygame.Rect(x, y, tile_size, tile_size)
        pygame.gfxdraw.box(surface, rect, (*color, 200)) # Base with alpha
        pygame.gfxdraw.rectangle(surface, rect, (*[c*0.8 for c in color], 255)) # Border

        # Create a surface for the indicator triangle
        indicator_surf = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
        indicator_color = (255, 255, 255)
        
        points = [(tile_size / 2, 10), (tile_size - 15, tile_size / 2 + 10), (15, tile_size / 2 + 10)]
        pygame.gfxdraw.aapolygon(indicator_surf, points, indicator_color)
        pygame.gfxdraw.filled_polygon(indicator_surf, points, indicator_color)

        # Calculate rotation
        if anim_data:
            progress = 1.0 - anim_data['progress']
            start_angle = anim_data['from'] * 90
            end_angle = anim_data['to'] * 90
            # Handle wrapping from 270 to 0
            if end_angle == 0 and start_angle == 270:
                end_angle = 360
            
            angle = start_angle + (end_angle - start_angle) * progress
        else:
            angle = orientation * 90

        rotated_indicator = pygame.transform.rotate(indicator_surf, -angle)
        new_rect = rotated_indicator.get_rect(center=rect.center)
        surface.blit(rotated_indicator, new_rect.topleft)

    def _render_ui(self):
        ui_x = 420
        # Panel background
        pygame.gfxdraw.box(self.screen, (ui_x - 20, 0, 640 - ui_x + 20, 400), self.COLOR_UI_BG)
        pygame.draw.line(self.screen, self.COLOR_GRID, (ui_x - 20, 0), (ui_x-20, 400), 2)
        
        # Score
        score_text = self.FONT_LARGE.render(f"{int(self.score):05d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 30))
        score_label = self.FONT_SMALL.render("SCORE", True, self.COLOR_SELECTOR)
        self.screen.blit(score_label, (ui_x, 15))

        # Time
        time_color = (255, 100, 100) if self.time_remaining < 10 else self.COLOR_TEXT
        time_text = self.FONT_LARGE.render(f"{max(0, self.time_remaining):.1f}", True, time_color)
        self.screen.blit(time_text, (ui_x, 100))
        time_label = self.FONT_SMALL.render("TIME", True, self.COLOR_SELECTOR)
        self.screen.blit(time_label, (ui_x, 85))

        # Stage
        stage_text = self.FONT_LARGE.render(f"{self.current_stage + 1} / {self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (ui_x, 170))
        stage_label = self.FONT_SMALL.render("STAGE", True, self.COLOR_SELECTOR)
        self.screen.blit(stage_label, (ui_x, 155))

        # Target Pattern Preview
        target_label = self.FONT_MEDIUM.render("TARGET", True, self.COLOR_SELECTOR)
        self.screen.blit(target_label, (ui_x, 230))
        
        preview_size = 35
        preview_offset_x = ui_x + (220 - self.GRID_DIM * preview_size) // 2
        preview_offset_y = 265
        
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_x = preview_offset_x + c * preview_size
                tile_y = preview_offset_y + r * preview_size
                color = self.grid_colors[r, c]
                orientation = self.target_patterns[self.current_stage][r, c]
                self._render_tile(self.screen, tile_x, tile_y, color, orientation, None)

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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tile Rotator")
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    while running:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to reset.")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.fps)
        
    env.close()