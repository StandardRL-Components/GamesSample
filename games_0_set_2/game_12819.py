import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:29:48.230819
# Source Brief: brief_02819.md
# Brief Index: 2819
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Survive waves of geometric projectiles by strategically magnetizing them,
    manipulating time, and flipping gravity within a visually stunning arena.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Survive waves of incoming geometric projectiles by using your abilities. "
        "Magnetize projectiles, slow down time, and flip gravity to stay alive."
    )
    user_guide = (
        "Hold Space to activate the magnet. Press Shift to slow time. "
        "Press ↑ to flip the gravity of green projectiles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CENTER = pygame.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    CENTER_RADIUS = 15

    # Colors
    COLOR_BG = (15, 19, 26)
    COLOR_GRID = (30, 35, 46)
    COLOR_CENTER = (230, 255, 255)
    COLOR_CENTER_GLOW = (60, 180, 255)
    COLOR_PROJ_SLOW = (0, 150, 255)
    COLOR_PROJ_FAST = (255, 80, 100)
    COLOR_PROJ_GRAV = (80, 255, 150)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TIME_SLOW_OVERLAY = (50, 100, 200, 100) # RGBA

    # Game Parameters
    MAX_STEPS = 1800 # 60 seconds at 30 FPS
    WIN_TIME = 60.0 # Seconds

    # Initial Difficulty
    INITIAL_PROJECTILE_SPEED = 1.5
    INITIAL_SPAWN_RATE = 2.0 # projectiles per second

    # Abilities
    TIME_SLOW_FACTOR = 0.4
    TIME_SLOW_DURATION = 3.0 # seconds
    TIME_SLOW_COOLDOWN = 10.0 # seconds
    GRAVITY_FLIP_COOLDOWN = 5.0 # seconds
    MAGNET_STRENGTH = 60.0
    STRONG_MAGNET_STRENGTH = 120.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_time = 0.0
        self.projectiles = []
        self.particles = []
        
        self.magnet_active = False
        self.magnet_strength_current = self.MAGNET_STRENGTH
        
        self.time_slow_active = False
        self.time_slow_timer = 0.0
        self.time_slow_cooldown_timer = 0.0
        
        self.gravity_flip_cooldown_timer = 0.0
        
        self.spawn_timer = 0.0
        self.current_projectile_speed = self.INITIAL_PROJECTILE_SPEED
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.last_difficulty_increase = 0.0
        
        self.unlocks = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_time = 0.0
        
        self.projectiles = []
        self.particles = []
        
        self.magnet_active = False
        self.magnet_strength_current = self.MAGNET_STRENGTH
        
        self.time_slow_active = False
        self.time_slow_timer = 0.0
        self.time_slow_cooldown_timer = 0.0
        
        self.gravity_flip_cooldown_timer = 0.0
        
        self.spawn_timer = 0.0
        self.current_projectile_speed = self.INITIAL_PROJECTILE_SPEED
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.last_difficulty_increase = 0.0

        self.unlocks = {
            'stronger_magnet': False,
            'extended_slowdown': False
        }

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = 1.0 / self.metadata["render_fps"]
        reward = 0.0

        # --- Handle Actions ---
        self._handle_actions(action)
        
        # --- Update Game State ---
        time_multiplier = self.TIME_SLOW_FACTOR if self.time_slow_active else 1.0
        game_dt = dt * time_multiplier
        self.game_time += game_dt

        self._update_cooldowns(game_dt)
        self._update_difficulty()
        self._update_unlocks()

        self._spawn_projectiles(game_dt)
        reward += self._update_projectiles(game_dt, time_multiplier)
        self._update_particles(dt) # Particles move at normal speed for visual effect

        # --- Calculate Rewards & Check Termination ---
        reward += 0.1  # Survival reward
        
        collision, projectile_hit = self._check_collisions()
        if collision:
            self.game_over = True
            reward = -100.0 # Loss penalty
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not terminated and self.game_time >= self.WIN_TIME:
            terminated = True
            reward = 100.0 # Win bonus
        
        self.score += reward
        self.steps += 1
        
        truncated = False # This environment does not truncate based on time limits
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action
        
        # Action: Magnet (Space)
        self.magnet_active = (space_held == 1)
        
        # Action: Time Slow (Shift)
        if shift_held == 1 and self.time_slow_cooldown_timer <= 0:
            self.time_slow_active = True
            self.time_slow_timer = self.TIME_SLOW_DURATION
            if self.unlocks['extended_slowdown']:
                 self.time_slow_timer *= 1.5
            self.time_slow_cooldown_timer = self.TIME_SLOW_COOLDOWN
            # sfx: time_slow_activate

        # Action: Gravity Flip (Movement Key 1 = Up)
        if movement == 1 and self.gravity_flip_cooldown_timer <= 0:
            flipped_any = False
            for p in self.projectiles:
                if p['type'] == 'grav':
                    p['vel'].y *= -1
                    self._create_particles(p['pos'], p['color'], 10, 2.0)
                    flipped_any = True
            if flipped_any:
                self.score += 2.0 # Reward for using gravity flip effectively
                self.gravity_flip_cooldown_timer = self.GRAVITY_FLIP_COOLDOWN
                # sfx: gravity_flip

    def _update_cooldowns(self, game_dt):
        if self.time_slow_cooldown_timer > 0:
            self.time_slow_cooldown_timer -= game_dt
        if self.gravity_flip_cooldown_timer > 0:
            self.gravity_flip_cooldown_timer -= game_dt
            
        if self.time_slow_active:
            self.time_slow_timer -= game_dt
            if self.time_slow_timer <= 0:
                self.time_slow_active = False
                # sfx: time_slow_deactivate

    def _update_difficulty(self):
        if self.game_time - self.last_difficulty_increase >= 10.0:
            self.last_difficulty_increase = self.game_time
            self.current_projectile_speed *= 1.10
            self.current_spawn_rate *= 1.05

    def _update_unlocks(self):
        if not self.unlocks['stronger_magnet'] and self.game_time >= 30.0:
            self.unlocks['stronger_magnet'] = True
            self.magnet_strength_current = self.STRONG_MAGNET_STRENGTH
        if not self.unlocks['extended_slowdown'] and self.game_time >= 60.0:
            self.unlocks['extended_slowdown'] = True

    def _spawn_projectiles(self, game_dt):
        self.spawn_timer += game_dt
        if self.spawn_timer >= 1.0 / self.current_spawn_rate:
            self.spawn_timer = 0
            
            # Choose spawn edge and position
            edge = random.randint(0, 3)
            if edge == 0: # Top
                pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), -20)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
            elif edge == 2: # Left
                pos = pygame.Vector2(-20, random.uniform(0, self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + 20, random.uniform(0, self.SCREEN_HEIGHT))

            # Choose type
            type_roll = random.random()
            if type_roll < 0.4:
                proj_type, color, size = 'slow', self.COLOR_PROJ_SLOW, 7
            elif type_roll < 0.8:
                proj_type, color, size = 'fast', self.COLOR_PROJ_FAST, 5
            else:
                proj_type, color, size = 'grav', self.COLOR_PROJ_GRAV, 6
            
            speed_multiplier = 1.5 if proj_type == 'fast' else 1.0
            vel = (self.CENTER - pos).normalize() * self.current_projectile_speed * speed_multiplier
            
            self.projectiles.append({
                'pos': pos, 'vel': vel, 'type': proj_type, 'color': color, 
                'size': size, 'magnetized': False
            })

    def _update_projectiles(self, game_dt, time_multiplier):
        reward = 0.0
        for p in self.projectiles[:]:
            # Magnetism
            if self.magnet_active:
                to_center = self.CENTER - p['pos']
                dist_sq = to_center.length_squared()
                if dist_sq > 1:
                    accel = to_center.normalize() * (self.magnet_strength_current / dist_sq) * 1000
                    p['vel'] += accel * game_dt
                    if not p['magnetized']:
                        reward += 1.0 # Reward for first-time magnetization
                        p['magnetized'] = True
            
            p['pos'] += p['vel'] * game_dt
            
            # Remove if off-screen
            if not self.screen.get_rect().inflate(50, 50).collidepoint(p['pos']):
                self.projectiles.remove(p)
        return reward

    def _update_particles(self, dt):
        for particle in self.particles[:]:
            particle['pos'] += particle['vel'] * dt
            particle['life'] -= dt
            if particle['life'] <= 0:
                self.particles.remove(particle)

    def _check_collisions(self):
        for p in self.projectiles[:]:
            if p['pos'].distance_to(self.CENTER) < self.CENTER_RADIUS + p['size']:
                self._create_particles(self.CENTER, (255, 255, 100), 100, 8.0)
                # sfx: player_hit_explosion
                return True, p
        return False, None

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(max_speed * 0.2, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': random.uniform(0.5, 1.5),
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_center_point()
        self._render_projectiles()
        self._render_particles()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_center_point(self):
        # Pulsing glow effect
        glow_size = self.CENTER_RADIUS + 15 + math.sin(self.game_time * 4) * 5
        if self.magnet_active:
            glow_size *= 1.5
        
        # Draw a transparent, larger circle for the glow
        temp_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        glow_color = self.COLOR_CENTER_GLOW
        if self.magnet_active:
            glow_color = (255, 100, 60)
        pygame.draw.circle(temp_surf, glow_color + (50,), (glow_size, glow_size), glow_size)
        self.screen.blit(temp_surf, (self.CENTER.x - glow_size, self.CENTER.y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core circle
        pygame.gfxdraw.aacircle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.CENTER_RADIUS, self.COLOR_CENTER)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CENTER.x), int(self.CENTER.y), self.CENTER_RADIUS, self.COLOR_CENTER)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'])
            if p['type'] == 'grav': # Triangle for gravity-affected
                points = [
                    (pos[0], pos[1] - size),
                    (pos[0] - size, pos[1] + size),
                    (pos[0] + size, pos[1] + size)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, p['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, p['color'])
            else: # Circle for others
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, p['color'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 255)))
            color = p['color']
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), int(p['size']))

    def _render_effects(self):
        if self.time_slow_active:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIME_SLOW_OVERLAY)
            self.screen.blit(overlay, (0, 0))

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.WIN_TIME - self.game_time)
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(timer_text, timer_rect)
        
        # Ability Cooldowns
        self._render_cooldown_icon(
            (self.SCREEN_WIDTH - 150, self.SCREEN_HEIGHT - 40), "SLOW", 
            self.time_slow_cooldown_timer, self.TIME_SLOW_COOLDOWN, self.time_slow_active
        )
        self._render_cooldown_icon(
            (self.SCREEN_WIDTH - 80, self.SCREEN_HEIGHT - 40), "FLIP",
            self.gravity_flip_cooldown_timer, self.GRAVITY_FLIP_COOLDOWN, False
        )

    def _render_cooldown_icon(self, pos, text, timer, max_cooldown, is_active):
        rect = pygame.Rect(pos[0], pos[1], 60, 30)
        
        # Background
        bg_color = (100, 100, 100) if timer > 0 else (150, 150, 150)
        if is_active:
            bg_color = (200, 200, 100)
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
        
        # Cooldown overlay
        if timer > 0:
            cooldown_height = rect.height * (timer / max_cooldown)
            overlay_rect = pygame.Rect(rect.left, rect.top, rect.width, cooldown_height)
            pygame.draw.rect(self.screen, (0, 0, 0, 150), overlay_rect, border_radius=5)
            
        # Text
        text_surf = self.font_small.render(text, True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_time": self.game_time,
            "projectiles": len(self.projectiles),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Geometric Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Time Survived: {info['game_time']:.2f}s")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.metadata["render_fps"])
        
    env.close()