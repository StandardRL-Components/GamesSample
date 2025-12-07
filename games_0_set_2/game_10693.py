import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:50:35.308227
# Source Brief: brief_00693.md
# Brief Index: 693
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-designed Gymnasium environment where the agent must maintain a growing
    constellation of satellites in orbit. The goal is to prevent collisions by
    adjusting the speed of individual satellites. The environment prioritizes high-quality
    visuals and a satisfying gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage a growing constellation of satellites. Adjust individual satellite speeds "
        "to prevent collisions and maintain a stable orbital system."
    )
    user_guide = (
        "Controls: Use ↑ and ↓ to select a satellite. Use ← and → to decrease or "
        "increase its speed. Avoid collisions."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_PLANET = (70, 100, 200)
    COLOR_PLANET_GLOW = (40, 60, 120)
    COLOR_ORBIT = (50, 60, 80)
    COLOR_SELECT_HIGHLIGHT = (100, 255, 100)
    COLOR_COLLISION = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    SATELLITE_COLORS = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (80, 255, 255), (255, 80, 255),
        (255, 165, 0), (128, 0, 128), (0, 128, 128),
        (255, 20, 147), (0, 255, 127), (218, 112, 214),
        (240, 230, 140), (173, 216, 230), (244, 164, 96)
    ]

    # Game parameters
    MAX_STEPS = 10000
    WIN_STEPS = 2400  # 120 seconds at 20 steps/sec
    INITIAL_SATELLITES = 7
    MAX_SATELLITES = 15
    ADD_SATELLITE_INTERVAL = 1200  # 60 seconds at 20 steps/sec
    MIN_SPEED, MAX_SPEED = 1, 5
    BASE_ANGULAR_VELOCITY = 0.008

    PLANET_RADIUS = 30
    SATELLITE_SIZE = 8
    COLLISION_DISTANCE = SATELLITE_SIZE
    MIN_SAFE_DISTANCE = SATELLITE_SIZE * 4

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
            self.font_ui = pygame.font.SysFont("Consolas", 18)
            self.font_info = pygame.font.SysFont("Consolas", 14)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_info = pygame.font.SysFont(None, 20)

        self.ORBITAL_RADII = [
            self.PLANET_RADIUS + 25 + i * 15 for i in range(self.MAX_SATELLITES)
        ]

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.satellites = []
        self.selected_satellite_idx = 0
        self.last_satellite_add_step = 0
        self.collision_info = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_satellite_idx = 0
        self.last_satellite_add_step = 0
        self.collision_info = None

        self.satellites = []
        available_orbits = list(self.ORBITAL_RADII)
        available_colors = list(self.SATELLITE_COLORS)
        
        self.np_random.shuffle(available_orbits)
        self.np_random.shuffle(available_colors)

        for i in range(self.INITIAL_SATELLITES):
            self._add_satellite(
                radius=available_orbits.pop(),
                color=available_colors.pop(),
                angle=self.np_random.uniform(0, 2 * math.pi),
                speed=self.np_random.integers(self.MIN_SPEED, self.MAX_SPEED + 1)
            )
        
        self.satellites.sort(key=lambda s: s['radius'])

        return self._get_observation(), self._get_info()

    def _add_satellite(self, radius, color, angle, speed):
        new_sat = {
            'radius': radius,
            'angle': angle,
            'speed': speed,
            'color': color,
            'pos': (0, 0),
            'trail_end': (0, 0)
        }
        self.satellites.append(new_sat)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        
        movement, _, _ = action
        
        num_sats = len(self.satellites)
        if num_sats > 0:
            if movement == 1:  # Up: Select next satellite
                self.selected_satellite_idx = (self.selected_satellite_idx + 1) % num_sats
            elif movement == 2:  # Down: Select previous satellite
                self.selected_satellite_idx = (self.selected_satellite_idx - 1 + num_sats) % num_sats
            
            selected_sat = self.satellites[self.selected_satellite_idx]
            if movement == 3:  # Left: Decrease speed
                selected_sat['speed'] = max(self.MIN_SPEED, selected_sat['speed'] - 1)
                # SFX: Speed down sound
            elif movement == 4:  # Right: Increase speed
                selected_sat['speed'] = min(self.MAX_SPEED, selected_sat['speed'] + 1)
                # SFX: Speed up sound

        for sat in self.satellites:
            sat['angle'] = (sat['angle'] + self.BASE_ANGULAR_VELOCITY * sat['speed']) % (2 * math.pi)
            x = self.CENTER_X + sat['radius'] * math.cos(sat['angle'])
            y = self.CENTER_Y + sat['radius'] * math.sin(sat['angle'])
            sat['pos'] = (x, y)
            
            trail_length = 5 + sat['speed'] * 4
            trail_angle = sat['angle'] + math.pi / 2
            tx = x - trail_length * math.cos(trail_angle)
            ty = y - trail_length * math.sin(trail_angle)
            sat['trail_end'] = (tx, ty)

        reward = self._calculate_reward_and_check_termination()
        self.score += reward

        self._update_progression()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_reward_and_check_termination(self):
        # Check for collisions
        for i in range(len(self.satellites)):
            for j in range(i + 1, len(self.satellites)):
                sat1, sat2 = self.satellites[i], self.satellites[j]
                dist = math.hypot(sat1['pos'][0] - sat2['pos'][0], sat1['pos'][1] - sat2['pos'][1])
                if dist < self.COLLISION_DISTANCE:
                    self.game_over = True
                    # SFX: Explosion sound
                    self.collision_info = {
                        'pos': ((sat1['pos'][0] + sat2['pos'][0]) / 2, (sat1['pos'][1] + sat2['pos'][1]) / 2)
                    }
                    return -100.0

        # Win condition
        if self.steps >= self.WIN_STEPS:
            self.game_over = True
            # SFX: Victory fanfare
            return 100.0
        
        # Continuous reward for maintaining safe distance
        reward = 0.0
        safe_sats = 0
        if len(self.satellites) > 0:
            for i in range(len(self.satellites)):
                min_dist_to_other = float('inf')
                for j in range(len(self.satellites)):
                    if i == j: continue
                    dist = math.hypot(self.satellites[i]['pos'][0] - self.satellites[j]['pos'][0],
                                     self.satellites[i]['pos'][1] - self.satellites[j]['pos'][1])
                    min_dist_to_other = min(min_dist_to_other, dist)
                
                if min_dist_to_other > self.MIN_SAFE_DISTANCE:
                    safe_sats += 1
            reward += 0.1 * (safe_sats / len(self.satellites)) # Normalized continuous reward

        # Event-based reward for adding a satellite
        if self.steps > 0 and self.steps % self.ADD_SATELLITE_INTERVAL == 0 and self.steps != self.last_satellite_add_step:
             if len(self.satellites) < self.MAX_SATELLITES:
                reward += 1.0
                # SFX: Success chime
        
        return reward

    def _update_progression(self):
        if not self.game_over and self.steps > 0 and self.steps % self.ADD_SATELLITE_INTERVAL == 0 and self.steps != self.last_satellite_add_step:
            if len(self.satellites) < self.MAX_SATELLITES:
                used_orbits = {s['radius'] for s in self.satellites}
                used_colors = {tuple(s['color']) for s in self.satellites}
                
                available_orbits = [r for r in self.ORBITAL_RADII if r not in used_orbits]
                available_colors = [c for c in self.SATELLITE_COLORS if c not in used_colors]
                
                if available_orbits and available_colors:
                    self._add_satellite(
                        radius=self.np_random.choice(available_orbits),
                        color=random.choice(available_colors), # Use python random for color tuple
                        angle=self.np_random.uniform(0, 2 * math.pi),
                        speed=self.np_random.integers(self.MIN_SPEED, self.MAX_SPEED + 1)
                    )
                    self.satellites.sort(key=lambda s: s['radius'])
                    self.last_satellite_add_step = self.steps

    def _render_game(self):
        # Planet with glow
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, self.PLANET_RADIUS + 5, self.COLOR_PLANET_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, self.CENTER_X, self.CENTER_Y, self.PLANET_RADIUS, self.COLOR_PLANET)
        pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, self.PLANET_RADIUS, self.COLOR_PLANET)

        # Orbits
        for sat in self.satellites:
            pygame.gfxdraw.aacircle(self.screen, self.CENTER_X, self.CENTER_Y, int(sat['radius']), self.COLOR_ORBIT)

        # Satellites and trails
        for i, sat in enumerate(self.satellites):
            pos_int = (int(sat['pos'][0]), int(sat['pos'][1]))
            
            trail_end_int = (int(sat['trail_end'][0]), int(sat['trail_end'][1]))
            pygame.draw.aaline(self.screen, sat['color'], pos_int, trail_end_int)
            
            sat_size = self.SATELLITE_SIZE // 2
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], sat_size, sat['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], sat_size, sat['color'])

            if i == self.selected_satellite_idx and not self.game_over:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], sat_size + 4, self.COLOR_SELECT_HIGHLIGHT)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], sat_size + 5, self.COLOR_SELECT_HIGHLIGHT)

        # Collision effect
        if self.collision_info:
            pos_int = (int(self.collision_info['pos'][0]), int(self.collision_info['pos'][1]))
            for i in range(10):
                angle = i * (2 * math.pi / 10) + (self.steps % 5) * 0.1
                end_x = pos_int[0] + 20 * math.cos(angle)
                end_y = pos_int[1] + 20 * math.sin(angle)
                pygame.draw.aaline(self.screen, self.COLOR_COLLISION, pos_int, (int(end_x), int(end_y)))

    def _render_ui(self):
        time_text = f"TIME: {self.steps / 20.0:.1f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        sat_count_text = f"SATELLITES: {len(self.satellites)}/{self.MAX_SATELLITES}"
        sat_count_surf = self.font_ui.render(sat_count_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(sat_count_surf, (self.SCREEN_WIDTH - sat_count_surf.get_width() - 10, 10))
        
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH // 2 - score_surf.get_width() // 2, 10))

        if len(self.satellites) > 0 and not self.game_over:
            selected_sat = self.satellites[self.selected_satellite_idx]
            speed_bar = '❚' * selected_sat['speed'] + ' ' * (self.MAX_SPEED - selected_sat['speed'])
            info_text = f"SELECTED: SAT {self.selected_satellite_idx + 1} | SPEED: [{speed_bar}]"
            info_surf = self.font_info.render(info_text, True, self.COLOR_SELECT_HIGHLIGHT)
            self.screen.blit(info_surf, (10, self.SCREEN_HEIGHT - info_surf.get_height() - 10))

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
            "satellites": len(self.satellites),
            "game_over": self.game_over,
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the game and play it manually.
    # It is not part of the Gymnasium environment definition.
    # Set the environment variable to run with a display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Orbital Command")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset the game
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_q: # Quit the game
                    running = False

        # Key presses for actions
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        action[0] = movement

        # Step the environment
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()
    pygame.quit()