import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:21:39.300595
# Source Brief: brief_00943.md
# Brief Index: 943
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent defends a central base from projectiles
    by placing and modifying energy portals.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your central base by placing and modifying energy portals to deflect waves of incoming projectiles."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place or upgrade a portal, and press shift over a portal to remove it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500  # Extended from brief for more gameplay
        self.MAX_WAVES = 50

        # Colors (Vibrant Neon on Dark)
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_BASE_GLOW = (0, 100, 200)
        self.COLOR_PROJECTILE = (255, 50, 50)
        self.COLOR_PORTAL = (0, 200, 255)
        self.COLOR_PORTAL_GLOW = (0, 150, 200)
        self.COLOR_PORTAL_EMPOWERED = (50, 255, 50)
        self.COLOR_PORTAL_EMPOWERED_GLOW = (50, 200, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)

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
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_pos = None
        self.base_health = None
        self.base_radius = None
        self.cursor_pos = None
        self.cursor_speed = 10
        self.cursor_radius = 15
        self.portal_radius = 20
        self.projectiles = []
        self.portals = []
        self.particles = []
        self.current_wave = 0
        self.wave_timer = 0
        self.wave_duration = 300 # steps
        self.empowered_portals_unlocked = False
        self.last_space_held = False
        self.last_shift_held = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Note: Pygame's randomness is not controlled by np.random.default_rng
            # For full determinism, one would need to seed Python's `random` module
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.base_health = 100.0
        self.base_radius = 30
        
        self.cursor_pos = self.base_pos.copy()
        
        self.projectiles = []
        self.portals = []
        self.particles = []
        
        self.current_wave = 1
        self.wave_timer = self.wave_duration - 60 # Start first wave quickly
        self.empowered_portals_unlocked = False
        
        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            self._handle_input(action)
            
            wave_reward = self._update_wave_logic()
            reward += wave_reward
            
            projectile_updates = self._update_projectiles()
            reward += projectile_updates['deflection_reward']
            self.base_health += projectile_updates['damage'] # damage is negative
            
            self._update_particles()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if self.base_health <= 0:
                reward = -100.0
            elif self.current_wave > self.MAX_WAVES:
                reward = 100.0
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos.y -= self.cursor_speed
        elif movement == 2: self.cursor_pos.y += self.cursor_speed
        elif movement == 3: self.cursor_pos.x -= self.cursor_speed
        elif movement == 4: self.cursor_pos.x += self.cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, self.cursor_radius, self.WIDTH - self.cursor_radius)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, self.cursor_radius, self.HEIGHT - self.cursor_radius)

        # Place/modify portal (on space press)
        if space_held and not self.last_space_held:
            portal_at_cursor = next((p for p in self.portals if p['pos'].distance_to(self.cursor_pos) < self.portal_radius), None)
            
            if portal_at_cursor:
                if self.empowered_portals_unlocked:
                    portal_at_cursor['state'] = 1 - portal_at_cursor['state']
                    # sound: portal_switch.wav
            elif self.cursor_pos.distance_to(self.base_pos) > self.base_radius + self.portal_radius:
                if len(self.portals) < 5: # Max 5 portals
                    normal = (self.base_pos - self.cursor_pos).normalize()
                    self.portals.append({'pos': self.cursor_pos.copy(), 'state': 0, 'normal': normal, 'last_hit_time': 0})
                    # sound: portal_place.wav

        # Remove portal (on shift press)
        if shift_held and not self.last_shift_held:
            initial_count = len(self.portals)
            self.portals = [p for p in self.portals if p['pos'].distance_to(self.cursor_pos) >= self.portal_radius]
            if len(self.portals) < initial_count:
                # sound: portal_remove.wav
                pass

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
    def _update_wave_logic(self):
        reward = 0
        self.wave_timer += 1
        if self.wave_timer >= self.wave_duration and self.current_wave <= self.MAX_WAVES:
            self.wave_timer = 0
            if self.current_wave > 0:
                reward = 1.0  # Wave survived reward
                # sound: wave_complete.wav
            
            self._spawn_wave()
            self.current_wave += 1
            
            if self.current_wave >= 10:
                self.empowered_portals_unlocked = True
        return reward

    def _spawn_wave(self):
        num_projectiles = 2 + self.current_wave // 2
        speed = 2.0 + (self.current_wave // 5) * 0.4
        
        for _ in range(num_projectiles):
            edge = random.randint(0, 3)
            if edge == 0: spawn_pos = pygame.Vector2(random.uniform(0, self.WIDTH), -10)
            elif edge == 1: spawn_pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            elif edge == 2: spawn_pos = pygame.Vector2(-10, random.uniform(0, self.HEIGHT))
            else: spawn_pos = pygame.Vector2(self.WIDTH + 10, random.uniform(0, self.HEIGHT))
            
            vel = (self.base_pos - spawn_pos).normalize() * speed
            self.projectiles.append({'pos': spawn_pos, 'vel': vel, 'trail': [spawn_pos.copy() for _ in range(10)]})

    def _update_projectiles(self):
        updates = {'deflection_reward': 0, 'damage': 0}
        projectiles_to_remove = []

        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            proj['trail'].pop(0)
            proj['trail'].append(proj['pos'].copy())

            if proj['pos'].distance_to(self.base_pos) < self.base_radius:
                updates['damage'] -= 15 # Base hit damage
                self._create_particles(proj['pos'], self.COLOR_PROJECTILE, 30, 4.0)
                projectiles_to_remove.append(proj)
                # sound: base_hit.wav
                continue
            
            collided_this_frame = False
            for portal in self.portals:
                if proj['pos'].distance_to(portal['pos']) < self.portal_radius and self.steps > portal['last_hit_time'] + 5:
                    updates['deflection_reward'] += 0.1
                    
                    speed_multiplier = 1.1 if portal['state'] == 1 else 1.0
                    proj['vel'] = proj['vel'].reflect(portal['normal']) * speed_multiplier
                    proj['pos'] += proj['vel'] # Nudge out of portal
                    
                    color = self.COLOR_PORTAL_EMPOWERED if portal['state'] == 1 else self.COLOR_PORTAL
                    self._create_particles(proj['pos'], color, 20, 3.0)
                    collided_this_frame = True
                    portal['last_hit_time'] = self.steps
                    # sound: deflect.wav
                    break
            if collided_this_frame:
                continue

            if not self.screen.get_rect().inflate(20, 20).collidepoint(proj['pos']):
                projectiles_to_remove.append(proj)
        
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return updates

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.96
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = random.uniform(1, 4)
            lifetime = random.randint(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'radius': radius, 'color': color, 'lifetime': lifetime})

    def _check_termination(self):
        return (
            self.base_health <= 0 or
            self.current_wave > self.MAX_WAVES
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Base
        self._render_glow_circle(self.base_pos, self.base_radius, self.COLOR_BASE_GLOW, self.COLOR_BASE, 15)

        # Portals
        for p in self.portals:
            color = self.COLOR_PORTAL_EMPOWERED if p['state'] == 1 else self.COLOR_PORTAL
            glow_color = self.COLOR_PORTAL_EMPOWERED_GLOW if p['state'] == 1 else self.COLOR_PORTAL_GLOW
            self._render_glow_circle(p['pos'], self.portal_radius, glow_color, color, 10)
            
            # Directional indicator
            start_pos = p['pos'] - p['normal'] * (self.portal_radius * 0.5)
            end_pos = p['pos'] + p['normal'] * (self.portal_radius * 0.7)
            pygame.draw.line(self.screen, (255, 255, 255), start_pos, end_pos, 2)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 40.0))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(p['radius'], p['radius']))

        # Projectiles
        for proj in self.projectiles:
            points = [(int(p.x), int(p.y)) for p in proj['trail']]
            pygame.draw.aalines(self.screen, self.COLOR_PROJECTILE, False, points)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(proj['pos'].x), int(proj['pos'].y)), 3)

        # Cursor
        if not self.game_over:
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), int(self.cursor_radius), self.COLOR_CURSOR)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x - 5, self.cursor_pos.y), (self.cursor_pos.x + 5, self.cursor_pos.y), 1)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x, self.cursor_pos.y - 5), (self.cursor_pos.x, self.cursor_pos.y + 5), 1)

    def _render_glow_circle(self, pos, radius, glow_color, core_color, glow_size):
        pos_int = (int(pos.x), int(pos.y))
        for i in range(glow_size, 0, -2):
            alpha = 40 - i * (40 / glow_size)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius + i), (*glow_color, alpha))
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(radius), core_color)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(radius), core_color)

    def _render_ui(self):
        # Wave number
        wave_text_str = f"Wave: {min(self.current_wave-1, self.MAX_WAVES)}/{self.MAX_WAVES}" if self.current_wave > 1 else f"Wave: 0/{self.MAX_WAVES}"
        wave_text = self.font_ui.render(wave_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Score
        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Base Health Bar
        health_percent = max(0, self.base_health / 100.0)
        bar_w, bar_h = 200, 20
        bar_x, bar_y = (self.WIDTH - bar_w) / 2, self.HEIGHT - bar_h - 10
        health_color = (int(255 * (1 - health_percent)), int(255 * health_percent), 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, bar_w * health_percent, bar_h))
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_w, bar_h), 2)
        
        if self.game_over:
            msg = "VICTORY" if self.base_health > 0 else "BASE DESTROYED"
            color = (50, 255, 50) if self.base_health > 0 else (255, 50, 50)
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "health": self.base_health}

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # Manual play example
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Portal Defender")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Info: {info}")
    env.close()