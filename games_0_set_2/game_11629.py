import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:20:00.627318
# Source Brief: brief_01629.md
# Brief Index: 1629
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Teleport between matching colored handholds to climb a treacherous mountain. "
        "Survive the weather and reach the summit."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to target a new handhold. "
        "Press space to teleport to the targeted handhold."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.SUMMIT_ALTITUDE = 4000
        self.MAX_STEPS = 2000
        self.FPS = 30

        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_MOUNTAIN = (35, 40, 55)
        self.HANDHOLD_COLORS = {
            "red": {"base": (255, 50, 50), "glow": (255, 100, 100)},
            "green": {"base": (50, 255, 50), "glow": (100, 255, 100)},
            "blue": {"base": (50, 50, 255), "glow": (100, 100, 255)},
        }
        self.COLOR_NAMES = list(self.HANDHOLD_COLORS.keys())
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 150)
        self.COLOR_TARGET = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_WEATHER_SNOW = (200, 200, 220)
        self.COLOR_WEATHER_WIND = (150, 160, 170)
        self.COLOR_WEATHER_LIGHTNING = (255, 255, 100)

        # --- Gymnasium Spaces ---
        # The observation space is a 3D array representing the game screen.
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        # Action space: [movement, teleport, (unused)]
        # 0: No-op, 1: Up, 2: Down, 3: Left, 4: Right
        # 0: No teleport, 1: Teleport
        # 0: No shift, 1: Shift (unused)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_weather = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 100.0
        self.altitude = 0
        self.camera_y = 0.0
        self.player_pos = [0, 0]
        self.player_color_name = "red"
        self.current_handhold_idx = 0
        self.targeted_handhold_idx = None
        self.teleport_cooldown = 0
        self.handholds = []
        self.mountain_poly = []
        self.particles = []
        self.weather_type = "clear"
        self.weather_duration = 0
        self.wind_direction = 0
        self.lightning_flash = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 100.0
        self.altitude = 0
        self.camera_y = 0.0
        self.targeted_handhold_idx = None
        self.teleport_cooldown = 0
        self.particles = []
        self.lightning_flash = 0
        
        # Reset weather
        self.weather_type = "clear"
        self.weather_duration = self.np_random.integers(100, 200)
        self.wind_direction = self.np_random.choice([-1, 1])

        # Generate world
        self._generate_mountain_and_handholds()

        # Place player on the starting handhold
        start_hold = self.handholds[0]
        self.player_pos = list(start_hold['pos'])
        self.player_color_name = start_hold['color_name']
        self.current_handhold_idx = 0
        self.camera_y = self.player_pos[1] - self.HEIGHT * 0.8

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        reward = 0
        self.teleport_cooldown = max(0, self.teleport_cooldown - 1)

        # 1. Update Game Logic
        self._update_weather()
        reward += self._get_weather_reward()

        # 2. Handle Player Actions
        self._handle_targeting(movement)
        teleport_reward = self._handle_teleport(space_pressed)
        reward += teleport_reward

        # 3. Update World State
        self._cycle_handhold_colors()
        self._update_player_fall()
        self._update_particles()
        self._update_camera()

        self.steps += 1
        
        # 4. Check for Termination
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Gymnasium standard

        # Final return tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_mountain_and_handholds(self):
        self.handholds = []
        
        # Start handhold
        start_pos = (self.WIDTH // 2, self.HEIGHT - 50)
        self.handholds.append({
            'pos': start_pos,
            'color_name': self.np_random.choice(self.COLOR_NAMES),
            'radius': 12
        })
        
        # Procedural generation
        current_y = start_pos[1]
        mountain_points_left = [(0, self.HEIGHT)]
        mountain_points_right = [(self.WIDTH, self.HEIGHT)]
        
        while current_y > -self.SUMMIT_ALTITUDE:
            # Difficulty scaling
            altitude_progress = 1 - (abs(current_y) / self.SUMMIT_ALTITUDE)
            handhold_prob = 0.3 + 0.6 * altitude_progress
            max_dist_x = 150 - 80 * altitude_progress
            max_dist_y = 180 - 100 * altitude_progress
            
            if self.np_random.random() < handhold_prob:
                last_hold = self.handholds[-1]
                angle = self.np_random.uniform(-math.pi, 0)
                dist = self.np_random.uniform(50, max_dist_y)
                
                new_x = last_hold['pos'][0] + math.cos(angle) * self.np_random.uniform(30, max_dist_x)
                new_y = last_hold['pos'][1] + math.sin(angle) * dist
                
                # Mountain shape for visuals and bounds
                mountain_width_at_y = self.WIDTH * (0.3 + 0.6 * (abs(new_y) / self.SUMMIT_ALTITUDE))
                min_x = self.WIDTH/2 - mountain_width_at_y/2
                max_x = self.WIDTH/2 + mountain_width_at_y/2
                new_x = np.clip(new_x, min_x + 20, max_x - 20)
                
                if new_y < last_hold['pos'][1]: # Ensure we are going up
                    self.handholds.append({
                        'pos': (new_x, new_y),
                        'color_name': self.np_random.choice(self.COLOR_NAMES),
                        'radius': 10
                    })
                    current_y = new_y
            else: # If no handhold, ensure path continues
                current_y -= 100

            # Add points for mountain polygon
            m_width = self.WIDTH * (0.3 + 0.6 * (abs(current_y) / self.SUMMIT_ALTITUDE))
            mountain_points_left.append((self.WIDTH/2 - m_width/2, current_y))
            mountain_points_right.insert(0, (self.WIDTH/2 + m_width/2, current_y))

        self.mountain_poly = mountain_points_left + mountain_points_right

    def _update_weather(self):
        self.weather_duration -= 1
        if self.lightning_flash > 0: self.lightning_flash -= 1

        if self.weather_duration <= 0:
            # Choose new weather based on altitude
            altitude_progress = self.altitude / self.SUMMIT_ALTITUDE
            weather_chance = self.np_random.random()
            
            if weather_chance < 0.3 + altitude_progress * 0.4: # Chance of bad weather
                self.weather_type = self.np_random.choice(["snow", "wind", "lightning"])
            else:
                self.weather_type = "clear"

            base_duration = 150 + 150 * altitude_progress
            self.weather_duration = self.np_random.integers(base_duration * 0.8, base_duration * 1.2)
            self.wind_direction = self.np_random.choice([-1, 1])

        # Apply weather effects
        if self.weather_type == "snow":
            self.player_health = max(0, self.player_health - 0.1)
            if self.np_random.random() < 0.8:
                self._add_particles(1, "snow")
        elif self.weather_type == "wind":
            self.player_pos[0] += self.wind_direction * (1.0 + 2.0 * (self.altitude/self.SUMMIT_ALTITUDE))
            if self.np_random.random() < 0.6:
                self._add_particles(1, "wind")
        elif self.weather_type == "lightning":
            if self.np_random.random() < (0.005 + 0.015 * (self.altitude/self.SUMMIT_ALTITUDE)):
                self.player_health = max(0, self.player_health - 10)
                self.lightning_flash = 3
                # sfx: lightning_strike.wav
                self._add_particles(50, "lightning_strike", pos=self.player_pos)


    def _get_weather_reward(self):
        if self.weather_type == "snow": return -0.01
        if self.weather_type == "wind": return -0.1
        if self.lightning_flash == 2: return 1.0 # Rewarded once for surviving strike
        return 0

    def _handle_targeting(self, movement_action):
        if movement_action == 0: return # No change in target

        direction_vectors = {
            1: (0, -1),  # Up
            2: (0, 1),   # Down
            3: (-1, 0),  # Left
            4: (1, 0),   # Right
        }
        target_dir = direction_vectors[movement_action]
        
        best_target_idx = None
        min_score = float('inf')

        for i, hold in enumerate(self.handholds):
            if i == self.current_handhold_idx: continue
            
            dx = hold['pos'][0] - self.player_pos[0]
            dy = hold['pos'][1] - self.player_pos[1]
            
            # Check if handhold is generally in the right direction
            if np.sign(dx) != target_dir[0] and target_dir[0] != 0: continue
            if np.sign(dy) != target_dir[1] and target_dir[1] != 0: continue

            # Score based on weighted distance
            dist_sq = dx**2 + dy**2
            if dist_sq == 0: continue

            # Penalize targets not aligned with the desired direction
            dir_dot = (dx * target_dir[0] + dy * target_dir[1]) / math.sqrt(dist_sq)
            if dir_dot < 0.3: continue # Must be at least somewhat in the right direction

            score = dist_sq / (dir_dot**2) # Prioritize alignment
            
            if score < min_score:
                min_score = score
                best_target_idx = i
        
        if best_target_idx is not None:
            self.targeted_handhold_idx = best_target_idx
            # sfx: target_select.wav

    def _handle_teleport(self, space_pressed):
        if not space_pressed or self.targeted_handhold_idx is None or self.teleport_cooldown > 0:
            return 0
        
        target_hold = self.handholds[self.targeted_handhold_idx]
        current_hold = self.handholds[self.current_handhold_idx]
        
        reward = 0
        if target_hold['color_name'] == current_hold['color_name']:
            # sfx: teleport_success.wav
            old_pos = list(self.player_pos)
            self._add_particles(30, "teleport_out", pos=old_pos, color=self.HANDHOLD_COLORS[current_hold['color_name']]['base'])
            
            self.player_pos = list(target_hold['pos'])
            self.current_handhold_idx = self.targeted_handhold_idx
            self.player_color_name = target_hold['color_name']
            self.altitude = max(0, self.HEIGHT - 50 - self.player_pos[1])

            self._add_particles(30, "teleport_in", pos=self.player_pos, color=self.HANDHOLD_COLORS[self.player_color_name]['base'])
            
            # Reward based on vertical movement
            reward = 0.1 if self.player_pos[1] < old_pos[1] else -0.1
            
            self.teleport_cooldown = 5
            self.targeted_handhold_idx = None
        else:
            # sfx: teleport_fail.wav
            self._add_particles(10, "fizzle", pos=target_hold['pos'])
            self.teleport_cooldown = 10
        
        return reward

    def _cycle_handhold_colors(self):
        if self.steps > 0 and self.steps % 15 == 0:
            # sfx: color_cycle.wav
            for i, hold in enumerate(self.handholds):
                if i != self.current_handhold_idx: # Don't change the player's current handhold
                    hold['color_name'] = self.np_random.choice(self.COLOR_NAMES)

    def _update_player_fall(self):
        # Gravity effect if not on a handhold (e.g. pushed off by wind)
        current_hold_pos = self.handholds[self.current_handhold_idx]['pos']
        dist_from_hold_sq = (self.player_pos[0] - current_hold_pos[0])**2 + (self.player_pos[1] - current_hold_pos[1])**2
        if dist_from_hold_sq > 20**2:
            self.player_pos[1] += 5 # Fall
            self.altitude = max(0, self.HEIGHT - 50 - self.player_pos[1])


    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _add_particles(self, num, p_type, pos=None, color=None):
        for _ in range(num):
            if p_type == "snow":
                p = {
                    'pos': [self.np_random.uniform(0, self.WIDTH), -self.camera_y + self.np_random.uniform(-10, 10)],
                    'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(1, 3)],
                    'life': self.np_random.integers(50, 150),
                    'type': 'snow'
                }
            elif p_type == "wind":
                start_x = self.WIDTH if self.wind_direction < 0 else 0
                p = {
                    'pos': [start_x, -self.camera_y + self.np_random.uniform(0, self.HEIGHT)],
                    'vel': [self.wind_direction * self.np_random.uniform(5, 10), self.np_random.uniform(-0.5, 0.5)],
                    'life': self.np_random.integers(80, 150),
                    'type': 'wind'
                }
            elif p_type in ["teleport_out", "teleport_in", "fizzle", "lightning_strike"]:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5) if p_type != "fizzle" else self.np_random.uniform(0.5, 2)
                p = {
                    'pos': list(pos),
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': self.np_random.integers(15, 30),
                    'type': 'burst',
                    'color': color if color else (200,200,200)
                }
            self.particles.append(p)

    def _update_camera(self):
        target_y = self.player_pos[1] - self.HEIGHT * 0.5
        self.camera_y += (target_y - self.camera_y) * 0.08

    def _check_termination(self):
        terminated = False
        reward = 0
        
        if self.player_health <= 0:
            terminated = True
            reward = -100
            # sfx: game_over_health.wav
        elif self.altitude >= self.SUMMIT_ALTITUDE:
            terminated = True
            reward = 100
            # sfx: victory.wav
        elif self.player_pos[1] > self.handholds[self.current_handhold_idx]['pos'][1] + self.HEIGHT * 0.6:
            terminated = True
            reward = -100
            # sfx: game_over_fall.wav
        
        if terminated:
            self.game_over = True
            
        return terminated, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw mountain background
        if self.mountain_poly:
            points = [(x, y - self.camera_y) for x, y in self.mountain_poly]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MOUNTAIN)

        # Draw weather particles
        self._draw_weather()
        
        # Draw handholds
        for i, hold in enumerate(self.handholds):
            pos_on_screen = (int(hold['pos'][0]), int(hold['pos'][1] - self.camera_y))
            if -20 < pos_on_screen[1] < self.HEIGHT + 20:
                color_data = self.HANDHOLD_COLORS[hold['color_name']]
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], hold['radius'], color_data['glow'])
                pygame.gfxdraw.filled_circle(self.screen, pos_on_screen[0], pos_on_screen[1], hold['radius']-2, color_data['base'])
                
                # Draw target indicator
                if i == self.targeted_handhold_idx:
                    pulse = abs(math.sin(self.steps * 0.3))
                    pygame.gfxdraw.aacircle(self.screen, pos_on_screen[0], pos_on_screen[1], int(hold['radius'] + 4 + pulse * 4), self.COLOR_TARGET)

        # Draw player
        player_screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y))
        self._draw_player(player_screen_pos)

        # Draw burst particles
        for p in self.particles:
            if p['type'] == 'burst':
                pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                pygame.draw.circle(self.screen, color, pos, 2)
        
        # Draw lightning flash
        if self.lightning_flash > 0:
            alpha = 150 * (self.lightning_flash / 3.0)
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 220, alpha))
            self.screen.blit(flash_surface, (0,0))


    def _draw_player(self, pos):
        size = 12
        glow_size = 20 + 5 * abs(math.sin(self.steps * 0.1))
        
        # Player color is tinted by the handhold they are on
        hold_color = self.HANDHOLD_COLORS[self.player_color_name]['base']
        player_tinted_color = (
            (self.COLOR_PLAYER[0] + hold_color[0]) // 2,
            (self.COLOR_PLAYER[1] + hold_color[1]) // 2,
            (self.COLOR_PLAYER[2] + hold_color[2]) // 2,
        )

        # Glow
        p1_glow = (pos[0], pos[1] - glow_size)
        p2_glow = (pos[0] - glow_size * 0.8, pos[1] + glow_size * 0.5)
        p3_glow = (pos[0] + glow_size * 0.8, pos[1] + glow_size * 0.5)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1_glow[0]), int(p1_glow[1]), int(p2_glow[0]), int(p2_glow[1]), int(p3_glow[0]), int(p3_glow[1]), (*self.COLOR_PLAYER_GLOW, 50))

        # Main triangle
        p1 = (pos[0], pos[1] - size)
        p2 = (pos[0] - size * 0.8, pos[1] + size * 0.5)
        p3 = (pos[0] + size * 0.8, pos[1] + size * 0.5)
        pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), player_tinted_color)
        pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _draw_weather(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1] + self.camera_y))
            if p['type'] == 'snow':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, (*self.COLOR_WEATHER_SNOW, 150))
            elif p['type'] == 'wind':
                end_pos = (pos[0] + p['vel'][0] * 0.5, pos[1] + p['vel'][1] * 0.5)
                pygame.draw.line(self.screen, (*self.COLOR_WEATHER_WIND, 100), pos, end_pos, 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / 100.0)
        health_bar_width = int(200 * health_ratio)
        health_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 50)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, health_bar_width, 20))
        self._draw_text(f"HP: {int(self.player_health)}", (110, 20), center=True)

        # Altitude
        alt_text = f"ALT: {int(self.altitude)} m / {self.SUMMIT_ALTITUDE} m"
        self._draw_text(alt_text, (self.WIDTH - 10, 20), align="right")

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (self.WIDTH - 10, 45), align="right")
        
        # Weather status
        weather_surf = self.font_weather.render(self.weather_type.upper(), True, self.COLOR_UI_TEXT)
        self.screen.blit(weather_surf, (self.WIDTH//2 - weather_surf.get_width()//2, 10))

    def _draw_text(self, text, pos, color=None, center=False, align="left"):
        if color is None: color = self.COLOR_UI_TEXT
        text_surface = self.font_ui.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "altitude": self.altitude,
            "health": self.player_health,
            "weather": self.weather_type,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block is for manual play and debugging.
    # It will not be executed by the evaluation script.
    # To use, you might need to unset the SDL_VIDEODRIVER dummy variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Configuration ---
    pygame.display.set_caption("Mountain Climber")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Action state
    movement = 0
    space_pressed = 0
    
    print(GameEnv.user_guide)
    print("  R: Reset environment, Q: Quit")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key == pygame.K_q:
                    env.close()
                    exit()

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_pressed = 1 if keys[pygame.K_SPACE] else 0
            shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_pressed, shift_pressed]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)