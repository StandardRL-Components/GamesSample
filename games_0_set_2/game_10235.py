import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:09:25.346318
# Source Brief: brief_00235.md
# Brief Index: 235
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Use powerful magnets to sort colored scrap metal into the correct recycling bins against the clock."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the active magnet. "
        "Press Shift to cycle between magnets and Space to activate a temporary power boost."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GAME_DURATION_SECONDS = 90
        self.FPS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.BIN_CAPACITY = 20 # Scraps to fill a bin

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.SCRAP_COLORS = {
            'R': (255, 80, 80), 'G': (80, 255, 80), 'B': (80, 80, 255)
        }
        self.BIN_COLORS = {
            'R': (180, 40, 40), 'G': (40, 180, 40), 'B': (40, 40, 180)
        }

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.magnets = []
        self.active_magnet_idx = 0
        self.scraps = []
        self.particles = []
        self.bins = {}
        self.scrap_spawn_timer = 0
        self.scrap_spawn_rate = 0.5 # scraps per second
        self.last_space_state = 0
        self.last_shift_state = 0

        # --- Magnet Properties ---
        self.MAGNET_PROPS = [
            {'size': 15, 'base_strength': 12000, 'color': (255, 255, 0)},
            {'size': 25, 'base_strength': 28000, 'color': (255, 165, 0)},
            {'size': 35, 'base_strength': 60000, 'color': (255, 69, 0)}
        ]
        
        # --- Bin Properties ---
        bin_width = 100
        bin_height = 80
        bin_y = self.SCREEN_HEIGHT - bin_height
        self.BIN_PROPS = {
            'R': pygame.Rect((self.SCREEN_WIDTH / 2) - 1.5 * bin_width - 20, bin_y, bin_width, bin_height),
            'G': pygame.Rect((self.SCREEN_WIDTH / 2) - 0.5 * bin_width, bin_y, bin_width, bin_height),
            'B': pygame.Rect((self.SCREEN_WIDTH / 2) + 0.5 * bin_width + 20, bin_y, bin_width, bin_height),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_STEPS
        
        self.magnets = []
        for i, props in enumerate(self.MAGNET_PROPS):
            self.magnets.append({
                'pos': pygame.Vector2(self.SCREEN_WIDTH / 3 * (i + 1), self.SCREEN_HEIGHT / 3),
                'vel': pygame.Vector2(0, 0),
                'size': props['size'],
                'base_strength': props['base_strength'],
                'color': props['color'],
                'boost_timer': 0,
                'is_chained': False,
            })
        self.active_magnet_idx = 0
        
        self.scraps = []
        self.particles = []
        
        self.bins = { 'R': 0, 'G': 0, 'B': 0 } # Fill count, not percentage

        self.scrap_spawn_rate = 0.5
        self.scrap_spawn_timer = 1 / self.scrap_spawn_rate * self.FPS

        self.last_space_state = 0
        self.last_shift_state = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1
        
        # Update difficulty
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.scrap_spawn_rate += 0.01

        chain_reward = self._update_magnets()
        reward += chain_reward
        self._update_spawner()
        
        dist_reward, score_reward, particles_to_add = self._update_scraps_and_bins()
        reward += dist_reward + score_reward
        self.particles.extend(particles_to_add)

        self._update_particles()
        
        # --- Termination and Terminal Rewards ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(val >= self.BIN_CAPACITY for val in self.bins.values()):
                reward += 100 # Win bonus
            else:
                reward += -50 # Lose penalty

        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        active_magnet = self.magnets[self.active_magnet_idx]
        move_speed = 5
        
        if movement == 1: active_magnet['pos'].y -= move_speed
        elif movement == 2: active_magnet['pos'].y += move_speed
        elif movement == 3: active_magnet['pos'].x -= move_speed
        elif movement == 4: active_magnet['pos'].x += move_speed

        # Clamp magnet position
        active_magnet['pos'].x = np.clip(active_magnet['pos'].x, 0, self.SCREEN_WIDTH)
        active_magnet['pos'].y = np.clip(active_magnet['pos'].y, 0, self.SCREEN_HEIGHT)

        # Handle boost (rising edge of space)
        if space_held and not self.last_space_state:
            active_magnet['boost_timer'] = 5 # 5 frames of boost
            # SFX: Boost activate
        self.last_space_state = space_held

        # Handle magnet switch (rising edge of shift)
        if shift_held and not self.last_shift_state:
            self.active_magnet_idx = (self.active_magnet_idx + 1) % len(self.magnets)
            # SFX: Switch magnet
        self.last_shift_state = shift_held

    def _update_magnets(self):
        # Update boost timers
        for m in self.magnets:
            if m['boost_timer'] > 0:
                m['boost_timer'] -= 1
        
        # Check for chain reactions
        chain_reward = 0
        for m in self.magnets: m['is_chained'] = False
        
        is_chaining = False
        for i in range(len(self.magnets)):
            for j in range(i + 1, len(self.magnets)):
                m1 = self.magnets[i]
                m2 = self.magnets[j]
                dist = m1['pos'].distance_to(m2['pos'])
                if dist < m1['size'] + m2['size']:
                    m1['is_chained'] = True
                    m2['is_chained'] = True
                    is_chaining = True
        
        if is_chaining:
            chain_reward = 5.0 # Chain reaction bonus reward
        
        return chain_reward

    def _update_spawner(self):
        self.scrap_spawn_timer -= 1
        if self.scrap_spawn_timer <= 0:
            scrap_type = random.choice(list(self.SCRAP_COLORS.keys()))
            self.scraps.append({
                'pos': pygame.Vector2(random.randint(20, self.SCREEN_WIDTH - 20), random.randint(20, 100)),
                'vel': pygame.Vector2(0, 0),
                'type': scrap_type,
                'color': self.SCRAP_COLORS[scrap_type],
                'size': 6
            })
            self.scrap_spawn_timer = (1 / self.scrap_spawn_rate) * self.FPS
            # SFX: Scrap spawn

    def _update_scraps_and_bins(self):
        dist_reward = 0
        score_reward = 0
        new_particles = []
        
        scraps_to_remove = []
        for i, scrap in enumerate(self.scraps):
            total_force = pygame.Vector2(0, 0)
            
            # Calculate old distance for reward
            target_bin_rect = self.BIN_PROPS[scrap['type']]
            old_dist = scrap['pos'].distance_to(target_bin_rect.center)
            
            # Calculate forces from magnets
            for magnet in self.magnets:
                dist_vec = magnet['pos'] - scrap['pos']
                dist_sq = dist_vec.length_squared()
                if dist_sq < 1: dist_sq = 1
                
                strength = magnet['base_strength']
                if magnet['boost_timer'] > 0: strength *= 1.5
                if magnet['is_chained']: strength *= 1.5
                    
                force_mag = strength / dist_sq
                total_force += dist_vec.normalize() * force_mag

            # Update velocity and position
            scrap['vel'] += total_force
            scrap['vel'] *= 0.9 # Damping/friction
            scrap['pos'] += scrap['vel']
            
            # Clamp position
            scrap['pos'].x = np.clip(scrap['pos'].x, 0, self.SCREEN_WIDTH)
            scrap['pos'].y = np.clip(scrap['pos'].y, 0, self.SCREEN_HEIGHT)
            
            # Calculate new distance for reward
            new_dist = scrap['pos'].distance_to(target_bin_rect.center)
            dist_reward += (old_dist - new_dist) * 0.001 # Scaled reward for moving closer

            # Check for bin collision
            if target_bin_rect.collidepoint(scrap['pos']):
                scraps_to_remove.append(i)
                if self.bins[scrap['type']] < self.BIN_CAPACITY:
                    self.bins[scrap['type']] += 1
                    self.score += 10
                    score_reward += 1.0
                    # SFX: Scrap collected
                    # Create particles
                    for _ in range(20):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(1, 4)
                        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                        new_particles.append({
                            'pos': scrap['pos'].copy(), 'vel': vel, 
                            'life': random.randint(20, 40), 'color': scrap['color']
                        })

        # Remove collected scraps
        for i in sorted(scraps_to_remove, reverse=True):
            del self.scraps[i]
            
        return dist_reward, score_reward, new_particles

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        win = all(val >= self.BIN_CAPACITY for val in self.bins.values())
        lose = self.timer <= 0
        return win or lose

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.timer / self.FPS,
            "bin_R%": self.bins['R'] / self.BIN_CAPACITY * 100,
            "bin_G%": self.bins['G'] / self.BIN_CAPACITY * 100,
            "bin_B%": self.bins['B'] / self.BIN_CAPACITY * 100,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_bins()
        self._render_attraction_lines()
        self._render_scraps()
        self._render_magnets()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
    
    def _render_bins(self):
        for type_char, rect in self.BIN_PROPS.items():
            # Draw bin container
            pygame.draw.rect(self.screen, self.BIN_COLORS[type_char], rect, border_radius=5)
            pygame.draw.rect(self.screen, self.SCRAP_COLORS[type_char], rect, width=2, border_radius=5)

            # Draw fill level
            fill_ratio = self.bins[type_char] / self.BIN_CAPACITY
            fill_height = int(rect.height * fill_ratio)
            fill_rect = pygame.Rect(rect.left, rect.bottom - fill_height, rect.width, fill_height)
            fill_color = self.SCRAP_COLORS[type_char]
            pygame.draw.rect(self.screen, fill_color, fill_rect, border_bottom_left_radius=5, border_bottom_right_radius=5)

            # Draw percentage text
            percent_text = f"{int(fill_ratio * 100)}%"
            self._draw_text(percent_text, self.font_small, rect.centerx, rect.top - 15)

    def _render_attraction_lines(self):
        for magnet in self.magnets:
            if magnet['boost_timer'] > 0 or magnet['is_chained']:
                for scrap in self.scraps:
                    dist = magnet['pos'].distance_to(scrap['pos'])
                    if dist < magnet['size'] * 8:
                        alpha = int(max(0, 255 * (1 - dist / (magnet['size'] * 8))))
                        if alpha > 20:
                            line_color = magnet['color'] + (alpha,)
                            pygame.draw.aaline(self.screen, line_color, magnet['pos'], scrap['pos'])

    def _render_scraps(self):
        for scrap in self.scraps:
            rect = pygame.Rect(scrap['pos'].x - scrap['size']/2, scrap['pos'].y - scrap['size']/2, scrap['size'], scrap['size'])
            pygame.draw.rect(self.screen, scrap['color'], rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, width=1)

    def _render_magnets(self):
        for i, magnet in enumerate(self.magnets):
            pos = (int(magnet['pos'].x), int(magnet['pos'].y))
            size = int(magnet['size'])
            
            # Chain reaction glow
            if magnet['is_chained']:
                for j in range(10, 0, -1):
                    alpha = 100 - j * 10
                    glow_color = magnet['color']
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * (1 + j/15)), (*glow_color, alpha))

            # Boost glow
            if magnet['boost_timer'] > 0:
                pulse = abs(math.sin(self.steps * 0.5))
                glow_size = int(size * (1.2 + pulse * 0.4))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, (*magnet['color'], 50))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_size, (*magnet['color'], 100))
            
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, magnet['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (255, 255, 255))
            
            # Active indicator
            if i == self.active_magnet_idx:
                angle = (self.steps * 4) % 360
                for k in range(3):
                    a = math.radians(angle + k * 120)
                    p1 = (pos[0] + math.cos(a) * (size + 4), pos[1] + math.sin(a) * (size + 4))
                    p2 = (pos[0] + math.cos(a) * (size + 10), pos[1] + math.sin(a) * (size + 10))
                    pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 2)

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['life'] * 0.1))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Timer
        time_left_sec = self.timer / self.FPS
        timer_text = f"TIME: {time_left_sec:.1f}"
        self._draw_text(timer_text, self.font_main, self.SCREEN_WIDTH - 100, 25)

        # Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, self.font_main, 100, 25)

    def _draw_text(self, text, font, x, y, color=None):
        if color is None: color = self.COLOR_TEXT
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        shadow_rect = shadow_surf.get_rect(center=(x + 2, y + 2))
        text_rect = text_surf.get_rect(center=(x, y))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you must comment out the os.environ line at the top
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Magnet Mayhem")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    while not done:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        env.clock.tick(env.FPS)
        
    env.close()