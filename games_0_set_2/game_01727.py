
# Generated: 2025-08-28T02:31:06.669835
# Source Brief: brief_01727.md
# Brief Index: 1727

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to 'click' on the selected tile and find the hidden objects."
    )

    game_description = (
        "An isometric hidden object game. Search a cluttered scene to find all 15 hidden objects before the 60-second timer runs out. Each click costs a small amount of time."
    )

    auto_advance = False

    # --- Colors ---
    COLOR_BG = (44, 62, 80)  # Dark blue-gray
    COLOR_GRID = (52, 73, 94)
    COLOR_CURSOR = (241, 196, 15, 150)  # Yellow, semi-transparent
    COLOR_TEXT = (236, 240, 241)
    COLOR_SUCCESS_FLASH = (46, 204, 113, 200)  # Green
    COLOR_FAIL_FLASH = (231, 76, 60, 200)   # Red
    COLOR_FOUND_OUTLINE = (149, 165, 166) # Gray
    
    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    TILE_W = 32
    TILE_H = 16
    NUM_OBJECTS = 15
    GAME_DURATION = 60.0  # seconds
    MAX_STEPS = 1200 # Approx 60s at 20 actions/sec

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except pygame.error:
            # Fallback if font system fails in headless mode
            self.font_large = self.font_medium = self.font_small = None

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 80
        
        self.clutter_items = []
        self.hidden_objects = []
        self.cursor_pos = [0, 0]
        self.time_remaining = self.GAME_DURATION
        self.score = 0
        self.steps = 0
        self.found_count = 0
        self.game_over = False
        
        self.click_feedback = [] # {'pos': [gx,gy], 'color': C, 'life': L}
        self.particles = [] # {'pos': [x,y], 'vel': [vx,vy], 'life': L, 'color': C}

        self.reset()
        
        # This check is commented out as it is for the user's local validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.found_count = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self.click_feedback = []
        self.particles = []
        
        self._generate_scene()

        return self._get_observation(), self._get_info()

    def _generate_scene(self):
        # Create a pool of all possible grid locations
        all_locations = [(gx, gy) for gx in range(self.GRID_WIDTH) for gy in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_locations)
        
        # Place hidden objects
        self.hidden_objects = []
        object_locations = all_locations[:self.NUM_OBJECTS]
        
        object_shapes = ['sphere', 'pyramid', 'gem']
        object_colors = [(52, 152, 219), (155, 89, 182), (26, 188, 156), (241, 196, 15), (230, 126, 34)]

        for i, pos in enumerate(object_locations):
            self.hidden_objects.append({
                'pos': pos,
                'found': False,
                'shape': self.np_random.choice(object_shapes),
                'color': random.choice(object_colors),
                'found_anim': 0 # Animation timer for when found
            })

        # Place background clutter
        self.clutter_items = []
        clutter_locations = all_locations[self.NUM_OBJECTS : self.NUM_OBJECTS + 40]
        clutter_colors = [(127, 140, 141), (90, 90, 90)]
        for pos in clutter_locations:
            self.clutter_items.append({
                'pos': pos,
                'height': self.np_random.integers(1, 4),
                'color': random.choice(clutter_colors)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        
        # Movement is discrete, one tile per step with action
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # Space press ("click") action
        if space_press:
            self.time_remaining -= 0.5 # Clicking costs time
            clicked_gx, clicked_gy = self.cursor_pos
            found_item = False
            for obj in self.hidden_objects:
                if not obj['found'] and obj['pos'] == (clicked_gx, clicked_gy):
                    obj['found'] = True
                    obj['found_anim'] = 30 # Start found animation
                    self.found_count += 1
                    self.score += 1
                    reward += 1
                    
                    # Sound effect placeholder
                    # play_sound('found_item.wav')

                    self.click_feedback.append({'pos': self.cursor_pos, 'color': self.COLOR_SUCCESS_FLASH, 'life': 15})
                    screen_pos = self._iso_to_screen(clicked_gx, clicked_gy)
                    self._create_sparkle_effect(screen_pos)
                    found_item = True
                    break
            
            if not found_item:
                self.score -= 0.1
                reward -= 0.1
                # Sound effect placeholder
                # play_sound('click_fail.wav')
                self.click_feedback.append({'pos': self.cursor_pos, 'color': self.COLOR_FAIL_FLASH, 'life': 15})

        # Update animations and timers
        self._update_effects()
        if not space_press and movement == 0: # A no-op action still costs time
            self.time_remaining -= 0.1

        # --- Check for Termination ---
        terminated = False
        if self.found_count == self.NUM_OBJECTS:
            win_bonus = 50 + max(0, self.time_remaining)
            self.score += win_bonus
            reward += win_bonus
            terminated = True
            self.game_over = True
        
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.score -= 100
            reward = -100 # Overwrite other rewards on timeout
            terminated = True
            self.game_over = True

        if self.steps >= self.MAX_STEPS and not terminated:
             self.score -= 100
             reward = -100
             terminated = True
             self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_effects(self):
        # Update click feedback animations
        self.click_feedback = [f for f in self.click_feedback if f['life'] > 0]
        for f in self.click_feedback:
            f['life'] -= 1
            
        # Update found object animations
        for obj in self.hidden_objects:
            if obj['found_anim'] > 0:
                obj['found_anim'] -= 1

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity

    def _iso_to_screen(self, gx, gy):
        sx = self.origin_x + (gx - gy) * (self.TILE_W / 2)
        sy = self.origin_y + (gx + gy) * (self.TILE_H / 2)
        return int(sx), int(sy)

    def _render_iso_tile(self, surface, gx, gy, color, filled=True):
        sx, sy = self._iso_to_screen(gx, gy)
        points = [
            (sx, sy),
            (sx + self.TILE_W / 2, sy + self.TILE_H / 2),
            (sx, sy + self.TILE_H),
            (sx - self.TILE_W / 2, sy + self.TILE_H / 2)
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_iso_cube(self, surface, gx, gy, color, height):
        sx, sy = self._iso_to_screen(gx, gy)
        tile_h = self.TILE_H
        tile_w = self.TILE_W
        
        top_color = color
        side_color_1 = tuple(max(0, c - 20) for c in color)
        side_color_2 = tuple(max(0, c - 40) for c in color)

        for i in range(height):
            z_offset = i * tile_h
            # Top face
            top_points = [
                (sx, sy - z_offset),
                (sx + tile_w / 2, sy + tile_h / 2 - z_offset),
                (sx, sy + tile_h - z_offset),
                (sx - tile_w / 2, sy + tile_h / 2 - z_offset)
            ]
            # Left face
            left_points = [
                (sx - tile_w / 2, sy + tile_h / 2 - z_offset),
                (sx, sy + tile_h - z_offset),
                (sx, sy + tile_h - z_offset + tile_h),
                (sx - tile_w / 2, sy + tile_h / 2 - z_offset + tile_h / 2 + tile_h / 2)
            ]
            # Right face
            right_points = [
                (sx + tile_w / 2, sy + tile_h / 2 - z_offset),
                (sx, sy + tile_h - z_offset),
                (sx, sy + tile_h - z_offset + tile_h),
                (sx + tile_w / 2, sy + tile_h / 2 - z_offset + tile_h / 2 + tile_h / 2)
            ]
            
            if i == height - 1: # Only draw top on the highest block
                 pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
                 pygame.gfxdraw.aapolygon(surface, top_points, side_color_2)
            
            pygame.gfxdraw.filled_polygon(surface, left_points, side_color_2)
            pygame.gfxdraw.filled_polygon(surface, right_points, side_color_1)

    def _render_iso_object(self, surface, obj):
        gx, gy = obj['pos']
        sx, sy = self._iso_to_screen(gx, gy)
        sy += self.TILE_H / 2 # Center on tile
        
        color = obj['color']
        outline_color = self.COLOR_FOUND_OUTLINE

        if obj['found']:
            if obj['found_anim'] > 0:
                # Flash white when found
                t = obj['found_anim'] / 30.0
                flash_color = (
                    int(color[0] + (255 - color[0]) * t),
                    int(color[1] + (255 - color[1]) * t),
                    int(color[2] + (255 - color[2]) * t),
                )
                color = flash_color
            else:
                color = outline_color

        if obj['shape'] == 'sphere':
            radius = int(self.TILE_W / 3)
            pygame.gfxdraw.filled_circle(surface, sx, int(sy - radius/2), radius, color)
            if not obj['found'] or obj['found_anim'] > 0:
                highlight_color = (255, 255, 255, 100)
                pygame.gfxdraw.filled_circle(surface, int(sx + radius/3), int(sy - radius/1.5), int(radius/4), highlight_color)
            pygame.gfxdraw.aacircle(surface, sx, int(sy - radius/2), radius, outline_color)

        elif obj['shape'] == 'pyramid':
            base_w, base_h = self.TILE_W * 0.6, self.TILE_H * 0.6
            height = self.TILE_H * 1.2
            points = [
                (sx, sy - height),
                (sx + base_w / 2, sy + base_h / 2),
                (sx - base_w / 2, sy + base_h / 2)
            ]
            if obj['found'] and obj['found_anim'] == 0:
                pygame.gfxdraw.aapolygon(surface, points, outline_color)
            else:
                pygame.gfxdraw.filled_trigon(surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), color)
                pygame.gfxdraw.aatrigon(surface, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), outline_color)

        elif obj['shape'] == 'gem':
            w, h = self.TILE_W * 0.4, self.TILE_H * 0.8
            points = [
                (sx, sy - h),
                (sx + w, sy),
                (sx, sy + h),
                (sx - w, sy)
            ]
            if obj['found'] and obj['found_anim'] == 0:
                pygame.gfxdraw.aapolygon(surface, points, outline_color)
            else:
                pygame.gfxdraw.filled_polygon(surface, points, color)
                pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render scene from back to front
        for gy in range(self.GRID_HEIGHT):
            for gx in range(self.GRID_WIDTH):
                # Draw grid tile
                self._render_iso_tile(self.screen, gx, gy, self.COLOR_GRID, filled=False)
                
                # Draw clutter
                for item in self.clutter_items:
                    if item['pos'] == (gx, gy):
                        self._render_iso_cube(self.screen, gx, gy, item['color'], item['height'])
                
                # Draw hidden objects
                for obj in self.hidden_objects:
                    if obj['pos'] == (gx, gy):
                        self._render_iso_object(self.screen, obj)
        
        # Draw cursor
        cursor_gx, cursor_gy = self.cursor_pos
        self._render_iso_tile(self.screen, cursor_gx, cursor_gy, self.COLOR_CURSOR)

        # Draw click feedback
        for f in self.click_feedback:
            gx, gy = f['pos']
            alpha = int(f['color'][3] * (f['life'] / 15.0))
            color_with_alpha = f['color'][:3] + (alpha,)
            self._render_iso_tile(self.screen, gx, gy, color_with_alpha)

        # Draw particles
        self._render_particles()

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _create_sparkle_effect(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': life,
                'max_life': life,
                'color': (255, 255, 150)
            })

    def _render_ui(self):
        if not self.font_medium: return
        
        # Timer display
        time_str = f"{self.time_remaining:.1f}"
        time_color = self.COLOR_TEXT
        if self.time_remaining < 10:
            time_color = (231, 76, 60) # Red
        elif self.time_remaining < 20:
            time_color = (241, 196, 15) # Yellow
        
        time_surf = self.font_large.render(time_str, True, time_color)
        time_rect = time_surf.get_rect(center=(self.SCREEN_WIDTH / 2, 30))
        self.screen.blit(time_surf, time_rect)

        # Found objects progress bar
        bar_width = 400
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - 35
        
        progress = self.found_count / self.NUM_OBJECTS
        fill_width = int(bar_width * progress)

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        if fill_width > 0:
            pygame.draw.rect(self.screen, (46, 204, 113), (bar_x, bar_y, fill_width, bar_height), border_radius=5)
        
        found_text = f"Found: {self.found_count} / {self.NUM_OBJECTS}"
        found_surf = self.font_small.render(found_text, True, self.COLOR_TEXT)
        found_rect = found_surf.get_rect(center=(self.SCREEN_WIDTH / 2, bar_y + bar_height / 2))
        self.screen.blit(found_surf, found_rect)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.found_count == self.NUM_OBJECTS:
                msg = "YOU WIN!"
                color = (46, 204, 113)
            else:
                msg = "TIME UP!"
                color = (231, 76, 60)
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)
            
            score_text = f"Final Score: {self.score:.1f}"
            score_surf = self.font_medium.render(score_text, True, self.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))
            self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "objects_found": self.found_count,
        }

    def close(self):
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Hidden Object Game")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
        space = 0 # 0: released, 1: pressed
        shift = 0 # 0: released, 1: pressed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(20) # Limit frame rate for playability

    env.close()