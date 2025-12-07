
# Generated: 2025-08-27T12:56:03.884234
# Source Brief: brief_00202.md
# Brief Index: 202

        
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

    user_guide = (
        "Controls: Use ←, ↑, → to jump left, up, and right. ↓ jumps down. Match the platform color to survive."
    )

    game_description = (
        "Climb a procedurally generated tower of colored platforms, jumping only on matching colors to reach the top."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TARGET_FPS = 30
        self.MAX_STEPS = 2000
        self.GOAL_HEIGHT = -10000 # World y-coordinate to win

        # Colors
        self.COLOR_BG_TOP = (120, 180, 255)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_PLAYER = (255, 255, 255)
        self.PLATFORM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 120, 255),  # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 255),  # Purple
        ]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (30, 30, 30)

        # Player constants
        self.PLAYER_RADIUS = 10
        self.JUMP_DURATION = 15  # frames
        self.JUMP_V_DISTANCE = 100
        self.JUMP_H_DISTANCE = 90
        self.JUMP_APEX_HEIGHT = 30
        self.FALL_SPEED = 6

        # Platform constants
        self.PLATFORM_WIDTH = 60
        self.PLATFORM_HEIGHT = 20
        self.PLATFORM_SPACING_Y = self.JUMP_V_DISTANCE
        self.PLATFORM_SPAWN_COUNT = 3 # Number of platforms per row

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Internal state variables (initialized in reset)
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_state = {}
        self.platforms = deque()
        self.particles = []
        self.camera_y = 0
        self.successful_jumps = 0
        self.max_height_level = 0
        self.current_h_spacing = 0 # for difficulty scaling

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.successful_jumps = 0
        self.max_height_level = 0
        
        # Difficulty scaling parameter
        self.base_h_spacing = 40
        self.min_h_spacing = 20
        self.h_spacing_reduction_rate = 1
        self.h_spacing_reduction_interval = 20

        # Player state
        self.player_state = {
            'pos': np.array([self.WIDTH / 2, self.HEIGHT - 80], dtype=float),
            'start_pos': np.array([0, 0], dtype=float),
            'target_pos': np.array([0, 0], dtype=float),
            'jump_timer': 0,
            'is_jumping': False,
            'is_falling': False,
            'color_idx': 0,
            'on_platform_idx': 0,
            'scale': 1.0
        }
        
        # World state
        self.platforms = deque()
        self.particles = []
        self.next_platform_y = self.HEIGHT - 80 - self.PLATFORM_SPACING_Y
        
        # Generate initial platforms
        start_platform = {
            'rect': pygame.Rect(
                self.player_state['pos'][0] - self.PLATFORM_WIDTH / 2,
                self.player_state['pos'][1],
                self.PLATFORM_WIDTH,
                self.PLATFORM_HEIGHT
            ),
            'color_idx': self.np_random.integers(0, len(self.PLATFORM_COLORS)),
            'id': 0
        }
        self.platforms.append(start_platform)
        self.player_state['color_idx'] = start_platform['color_idx']

        while self.next_platform_y > -self.HEIGHT:
            self._generate_new_platforms()

        self.camera_y = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update player scale for animation
        self.player_state['scale'] = max(1.0, self.player_state['scale'] - 0.1)

        # Handle player input if not busy
        if not self.player_state['is_jumping'] and not self.player_state['is_falling']:
            if movement == 0: # no-op
                reward -= 0.2
            elif 1 <= movement <= 4: # Jump actions
                self._start_jump(movement)
                # Sound: Jump

        # Update game logic
        self._update_player_movement()
        landing_info = self._handle_landing()
        if landing_info['landed']:
            if landing_info['match']:
                # Sound: Success
                reward += 1.0
                self.score += 10
                self.successful_jumps += 1
                
                # Check for new height level reward
                current_height_level = self.successful_jumps // 10
                if current_height_level > self.max_height_level:
                    self.max_height_level = current_height_level
                    reward += 10.0
                    self.score += 100
                
                # Difficulty scaling
                self.current_h_spacing = max(
                    self.min_h_spacing,
                    self.base_h_spacing - (self.successful_jumps // self.h_spacing_reduction_interval) * self.h_spacing_reduction_rate
                )
            else:
                # Sound: Mismatch
                self.player_state['is_falling'] = True

        self._update_camera()
        self._manage_platforms()
        self._update_particles()
        
        # Check termination conditions
        terminated = False
        if self.player_state['pos'][1] - self.camera_y > self.HEIGHT + self.PLAYER_RADIUS:
            terminated = True
            reward = -10.0
            self.score -= 500
        elif len(self.platforms) > 0 and self.player_state['on_platform_idx'] != -1 and self.platforms[self.player_state['on_platform_idx']]['rect'].y < self.GOAL_HEIGHT:
            terminated = True
            reward = 100.0
            self.score += 10000
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _start_jump(self, movement):
        self.player_state['is_jumping'] = True
        self.player_state['jump_timer'] = self.JUMP_DURATION
        self.player_state['start_pos'] = self.player_state['pos'].copy()
        target = self.player_state['start_pos'].copy()
        
        if movement == 1: # Up
            target[1] -= self.JUMP_V_DISTANCE
        elif movement == 2: # Down
            target[1] += self.JUMP_V_DISTANCE
        elif movement == 3: # Left
            target[0] -= self.JUMP_H_DISTANCE
            target[1] -= self.JUMP_V_DISTANCE
        elif movement == 4: # Right
            target[0] += self.JUMP_H_DISTANCE
            target[1] -= self.JUMP_V_DISTANCE
            
        self.player_state['target_pos'] = target
        self.player_state['on_platform_idx'] = -1
        self.player_state['scale'] = 1.5 # Squash and stretch

    def _update_player_movement(self):
        if self.player_state['is_jumping']:
            self.player_state['jump_timer'] -= 1
            t = 1.0 - (self.player_state['jump_timer'] / self.JUMP_DURATION)
            
            # Linear interpolation for position
            self.player_state['pos'] = self.player_state['start_pos'] * (1-t) + self.player_state['target_pos'] * t
            
            # Sinusoidal arc for visual appeal
            arc_height = math.sin(t * math.pi) * self.JUMP_APEX_HEIGHT
            self.player_state['pos'][1] -= arc_height

            if self.player_state['jump_timer'] <= 0:
                self.player_state['is_jumping'] = False
                # If no landing, start falling
                if self.player_state['on_platform_idx'] == -1:
                    self.player_state['is_falling'] = True
                    # Sound: Fall

        elif self.player_state['is_falling']:
            self.player_state['pos'][1] += self.FALL_SPEED

    def _handle_landing(self):
        if self.player_state['is_jumping']:
            return {'landed': False}

        for i, platform in enumerate(self.platforms):
            # Simple collision check: player center is within platform bounds and y is close
            is_above = platform['rect'].top - 5 < self.player_state['pos'][1] < platform['rect'].top + 5
            is_horizontally_aligned = platform['rect'].left < self.player_state['pos'][0] < platform['rect'].right
            
            if is_above and is_horizontally_aligned:
                self.player_state['is_falling'] = False
                self.player_state['pos'][1] = platform['rect'].top
                self.player_state['on_platform_idx'] = i
                self.player_state['scale'] = 0.7 # Squash and stretch

                match = self.player_state['color_idx'] == platform['color_idx']
                if match:
                    self.player_state['color_idx'] = platform['color_idx']
                    self._create_particles(self.player_state['pos'], self.PLATFORM_COLORS[platform['color_idx']])
                
                return {'landed': True, 'match': match}
        
        return {'landed': False}
        
    def _update_camera(self):
        # Camera follows player, keeping them in the lower part of the screen
        target_camera_y = self.player_state['pos'][1] - self.HEIGHT * 0.7
        # Smooth camera movement
        self.camera_y = self.camera_y * 0.9 + target_camera_y * 0.1

    def _manage_platforms(self):
        # Remove platforms that are off-screen below
        while self.platforms and self.platforms[0]['rect'].top - self.camera_y > self.HEIGHT:
            self.platforms.popleft()
        
        # Generate new platforms that are about to come on screen
        while self.next_platform_y > self.camera_y - self.HEIGHT:
            self._generate_new_platforms()

    def _generate_new_platforms(self):
        num_platforms = self.np_random.integers(2, self.PLATFORM_SPAWN_COUNT + 1)
        possible_x_positions = np.linspace(
            self.PLATFORM_WIDTH, self.WIDTH - self.PLATFORM_WIDTH, num_platforms + 2
        )[1:-1]
        
        chosen_x_indices = self.np_random.choice(len(possible_x_positions), num_platforms, replace=False)
        
        for i in chosen_x_indices:
            x = possible_x_positions[i]
            platform = {
                'rect': pygame.Rect(
                    x - self.PLATFORM_WIDTH / 2,
                    self.next_platform_y,
                    self.PLATFORM_WIDTH,
                    self.PLATFORM_HEIGHT
                ),
                'color_idx': self.np_random.integers(0, len(self.PLATFORM_COLORS)),
                'id': self.steps + i
            }
            self.platforms.append(platform)
        
        self.next_platform_y -= self.PLATFORM_SPACING_Y
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                'pos': list(pos),
                'vel': velocity,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self._draw_background()
        self._draw_platforms()
        self._draw_particles()
        self._draw_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "successful_jumps": self.successful_jumps,
            "player_y": self.player_state['pos'][1]
        }
        
    def _draw_background(self):
        # Efficient gradient by drawing lines
        for y in range(self.HEIGHT):
            t = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_BOTTOM[0] * (1 - t) + self.COLOR_BG_TOP[0] * t),
                int(self.COLOR_BG_BOTTOM[1] * (1 - t) + self.COLOR_BG_TOP[1] * t),
                int(self.COLOR_BG_BOTTOM[2] * (1 - t) + self.COLOR_BG_TOP[2] * t),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _draw_platforms(self):
        for i, p in enumerate(self.platforms):
            rect_on_screen = p['rect'].copy()
            rect_on_screen.y -= int(self.camera_y)
            
            color = self.PLATFORM_COLORS[p['color_idx']]
            pygame.draw.rect(self.screen, color, rect_on_screen, border_radius=3)
            
            # Glow for the current platform
            if not self.player_state['is_jumping'] and not self.player_state['is_falling'] and i == self.player_state['on_platform_idx']:
                self._draw_glow(self.screen, rect_on_screen, (255, 255, 255), 15)

    def _draw_player(self):
        pos_on_screen = (
            int(self.player_state['pos'][0]),
            int(self.player_state['pos'][1] - self.camera_y)
        )
        radius = int(self.PLAYER_RADIUS * self.player_state['scale'])
        
        # Player is white, but has a border of their current color for clarity
        border_color = self.PLATFORM_COLORS[self.player_state['color_idx']]
        pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], radius, border_color)
        pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], radius - 3, self.COLOR_PLAYER)

    def _draw_particles(self):
        for p in self.particles:
            pos_on_screen = (
                int(p['pos'][0]),
                int(p['pos'][1] - self.camera_y)
            )
            pygame.gfxdraw.filled_circle(
                self.screen, pos_on_screen[0], pos_on_screen[1],
                max(0, int(p['radius'])), p['color']
            )

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_shadow = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(score_shadow, (12, 12))
        self.screen.blit(score_text, (10, 10))

        # Height
        height = -int(self.player_state['pos'][1] / 10)
        height_text = self.font_small.render(f"Height: {height}m", True, self.COLOR_TEXT)
        height_shadow = self.font_small.render(f"Height: {height}m", True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(height_shadow, (12, 52))
        self.screen.blit(height_text, (10, 50))
        
        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            
            end_text_str = "YOU WON!" if self.score > 1000 else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_TEXT)
            end_shadow = self.font_large.render(end_text_str, True, self.COLOR_TEXT_SHADOW)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_shadow, text_rect.move(2, 2))
            self.screen.blit(end_text, text_rect)


    def _draw_glow(self, surface, rect, color, radius):
        for i in range(radius, 0, -2):
            alpha = int(100 * (1 - (i / radius)))
            glow_surf = pygame.Surface((rect.width + i*2, rect.height + i*2), pygame.SRCALPHA)
            glow_color = (*color, alpha)
            pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=i+3, width=2)
            surface.blit(glow_surf, (rect.x - i, rect.y - i))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
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