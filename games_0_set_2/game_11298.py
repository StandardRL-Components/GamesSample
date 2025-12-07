import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:55:06.794913
# Source Brief: brief_01298.md
# Brief Index: 1298
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A rhythmic stealth game where the player controls a geometric shape.
    The goal is to stay within moving shadows to avoid detection by sweeping lights.
    The player can move and change their size to fit into safe zones.
    The game's state updates on a rhythmic beat, requiring timed actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A rhythmic stealth game where you must stay within moving shadows to avoid detection, "
        "changing size to fit into safe zones."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold space to grow and shift to shrink your shape."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 60)
    COLOR_SHADOW = (40, 50, 90)
    COLOR_LIGHT = (255, 255, 200)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (100, 255, 255)
    COLOR_DETECTION = (255, 50, 50)
    COLOR_TEXT = (220, 220, 240)
    
    # Player settings
    PLAYER_MOVE_SPEED = 4.0
    PLAYER_MIN_SIZE = 10.0
    PLAYER_MAX_SIZE = 60.0
    PLAYER_GROW_RATE = 1.0
    PLAYER_SHRINK_RATE = 1.0
    
    # Game rules
    MAX_STEPS = 1500
    BEAT_DURATION_FRAMES = 20 # Game logic updates every N frames
    BEATS_PER_LEVEL = 15

    # Rewards
    REWARD_DETECTED = -50.0
    REWARD_LEVEL_COMPLETE = 50.0
    REWARD_BEAT_SURVIVED = 1.0
    REWARD_IN_SHADOW_PER_STEP = 0.05
    REWARD_TIME_PENALTY_PER_STEP = -0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.player_pos = None
        self.player_size = None
        self.player_sides = 0
        self.lights = []
        self.shadows = []
        self.particles = []
        self.beat_timer = 0
        self.level_beat_counter = 0
        self.total_reward = 0.0
        
        # This is reset in the first call to reset()
        self._first_reset = True
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self._first_reset or self.level == 0:
            self.level = 1
            self._first_reset = False
        
        self.steps = 0
        self.score = 0
        self.total_reward = 0.0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_size = (self.PLAYER_MIN_SIZE + self.PLAYER_MAX_SIZE) / 2
        
        self.particles = []
        self.beat_timer = 0
        self.level_beat_counter = 0
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = self.REWARD_TIME_PENALTY_PER_STEP
        terminated = False
        truncated = False

        self._handle_player_input(movement, space_held, shift_held)
        self._update_particles()
        
        self.beat_timer += 1
        if self.beat_timer >= self.BEAT_DURATION_FRAMES:
            self.beat_timer = 0
            self.level_beat_counter += 1
            self._update_world_on_beat()

            # --- Detection and Reward Logic on Beat ---
            in_light = self._is_player_in_light()
            in_shadow = self._is_player_in_shadow()

            if in_light and not in_shadow:
                # sound_placeholder: play('detection_fail')
                reward += self.REWARD_DETECTED
                self.game_over = True
                terminated = True
                self._create_detection_effect()
            else:
                # sound_placeholder: play('beat_survived_tick')
                reward += self.REWARD_BEAT_SURVIVED

            # --- Level Completion Logic on Beat ---
            if not terminated and self.level_beat_counter >= self.BEATS_PER_LEVEL:
                # sound_placeholder: play('level_complete_chime')
                reward += self.REWARD_LEVEL_COMPLETE
                self.level += 1
                self.score += 1
                self.level_beat_counter = 0
                self._setup_level()

        # Continuous reward for being in shadow
        if self._is_player_in_shadow():
            reward += self.REWARD_IN_SHADOW_PER_STEP

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        terminated = self.game_over or terminated
        
        self.total_reward += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _setup_level(self):
        """Initializes lights and shadows based on the current level."""
        self.lights = []
        self.shadows = []
        self.player_sides = 3 + ((self.level - 1) // 10)
        
        # Difficulty scaling
        num_lights = 1 + (self.level - 1) // 3
        light_speed = 0.02 + ((self.level - 1) // 5) * 0.005
        
        num_shadows = max(1, 5 - ((self.level - 1) // 4))
        min_shadow_size = max(40, 150 - self.level * 4)
        max_shadow_size = max(80, 250 - self.level * 5)

        for _ in range(num_lights):
            self.lights.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)),
                "radius": self.np_random.uniform(80, 150),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "speed": self.np_random.uniform(light_speed * 0.8, light_speed * 1.2),
                "angle_vel": self.np_random.uniform(-0.02, 0.02)
            })
            
        for _ in range(num_shadows):
            size = self.np_random.uniform(min_shadow_size, max_shadow_size)
            self.shadows.append(pygame.Rect(
                self.np_random.uniform(0, self.SCREEN_WIDTH - size),
                self.np_random.uniform(0, self.SCREEN_HEIGHT - size),
                size,
                size
            ))

    def _handle_player_input(self, movement, space_held, shift_held):
        """Updates player state based on actions."""
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1  # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_MOVE_SPEED

        # --- Boundary checks ---
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size, self.SCREEN_WIDTH - self.player_size)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size, self.SCREEN_HEIGHT - self.player_size)
        
        # --- Size change ---
        if space_held: # Grow
            # sound_placeholder: play('grow_sound', loop=True)
            self.player_size += self.PLAYER_GROW_RATE
        if shift_held: # Shrink
            # sound_placeholder: play('shrink_sound', loop=True)
            self.player_size -= self.PLAYER_SHRINK_RATE
            
        self.player_size = np.clip(self.player_size, self.PLAYER_MIN_SIZE, self.PLAYER_MAX_SIZE)

    def _update_world_on_beat(self):
        """Updates light and shadow positions."""
        for light in self.lights:
            light['angle'] += light['angle_vel']
            light['pos'].x += math.cos(light['angle']) * light['speed'] * self.BEAT_DURATION_FRAMES
            light['pos'].y += math.sin(light['angle']) * light['speed'] * self.BEAT_DURATION_FRAMES
            
            # Bounce off walls
            if not (light['radius'] < light['pos'].x < self.SCREEN_WIDTH - light['radius']):
                light['angle'] = math.pi - light['angle']
            if not (light['radius'] < light['pos'].y < self.SCREEN_HEIGHT - light['radius']):
                light['angle'] = -light['angle']
            
            light['pos'].x = np.clip(light['pos'].x, light['radius'], self.SCREEN_WIDTH - light['radius'])
            light['pos'].y = np.clip(light['pos'].y, light['radius'], self.SCREEN_HEIGHT - light['radius'])

    def _is_player_in_light(self):
        for light in self.lights:
            if self.player_pos.distance_to(light['pos']) < light['radius']:
                return True
        return False

    def _is_player_in_shadow(self):
        # Check if player's center is within any shadow rect
        player_rect = pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)
        for shadow in self.shadows:
            if shadow.colliderect(player_rect):
                return True
        return False

    def _create_detection_effect(self):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": self.player_pos.copy(),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifetime": self.np_random.integers(20, 40),
                "color": self.COLOR_DETECTION,
                "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['vel'] *= 0.95 # friction

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
            "level": self.level,
            "total_reward": self.total_reward
        }

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw shadows
        for shadow in self.shadows:
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow)

        # Draw lights
        for light in self.lights:
            self._draw_soft_circle(
                self.screen, light['pos'], light['radius'], self.COLOR_LIGHT
            )

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw player
        if not self.game_over:
            self._draw_polygon(self.screen, self.player_pos, self.player_size, self.player_sides, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Level
        level_text = self.font_main.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        
        # Score (Completed Levels)
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Beat Meter
        beat_progress = self.beat_timer / self.BEAT_DURATION_FRAMES
        meter_width = 200
        meter_height = 10
        meter_x = (self.SCREEN_WIDTH - meter_width) / 2
        meter_y = self.SCREEN_HEIGHT - 25
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (meter_x, meter_y, meter_width, meter_height), 1)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (meter_x, meter_y, meter_width * beat_progress, meter_height))

        # Level progress
        level_progress = self.level_beat_counter / self.BEATS_PER_LEVEL
        level_meter_y = self.SCREEN_HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_GRID, (meter_x, level_meter_y, meter_width, 5), 1)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, (meter_x, level_meter_y, meter_width * level_progress, 5))

    @staticmethod
    def _draw_polygon(surface, center, radius, n_sides, color, glow_color):
        """Draws a regular polygon with a glow effect."""
        if n_sides < 3: return
        angle_offset = math.pi / 2 # Point upwards
        points = []
        for i in range(n_sides):
            angle = i * 2 * math.pi / n_sides + angle_offset
            x = center.x + radius * math.cos(angle)
            y = center.y - radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        # Glow effect by drawing a thicker line behind
        pygame.draw.polygon(surface, glow_color, points, width=max(1, int(radius * 0.15)))
        pygame.gfxdraw.aapolygon(surface, points, glow_color)
        
        # Main polygon
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    @staticmethod
    def _draw_soft_circle(surface, pos, radius, color):
        """Draws a circle with a soft, glowing edge."""
        pos_int = (int(pos.x), int(pos.y))
        rad_int = int(radius)
        
        # Create a temporary surface for blending
        target_rect = pygame.Rect(pos_int[0] - rad_int, pos_int[1] - rad_int, rad_int * 2, rad_int * 2)
        temp_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        
        # Draw circles with decreasing alpha for the glow
        num_layers = 10
        for i in range(num_layers, 0, -1):
            alpha = int(255 * (i / num_layers)**2 * 0.1) # 10% of base color for glow
            pygame.gfxdraw.filled_circle(
                temp_surf, rad_int, rad_int, 
                int(rad_int * (i / num_layers)),
                (*color[:3], alpha)
            )

        # Draw the solid center
        pygame.gfxdraw.filled_circle(temp_surf, rad_int, rad_int, int(rad_int * 0.8), color)
        surface.blit(temp_surf, target_rect, special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Manual Play ---
    # The following code is for human-in-the-loop testing and is not part of the Gymnasium environment
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a specific driver for display
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override pygame screen for direct display
    pygame.display.init()
    env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythmic Stealth")
    
    total_reward = 0
    total_steps = 0
    
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Render the observation to the display
        pygame.display.flip()
        env.clock.tick(30) # Run at 30 FPS

        if done:
            print(f"Episode finished in {total_steps} steps.")
            print(f"Final Score (Levels): {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")

    env.close()