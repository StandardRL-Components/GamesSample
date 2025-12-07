import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:55:23.302485
# Source Brief: brief_02981.md
# Brief Index: 2981
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a visually-striking, time-bending first-person shooter.
    The player navigates a pseudo-3D space, fighting color-coded fractal enemies.
    The core mechanics involve matching weapon color to enemy weaknesses and strategically
    using a time-dilation ability.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A time-bending first-person shooter where you fight color-coded fractal enemies. "
        "Match your weapon color to enemy weaknesses and use time-dilation to gain an advantage."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire. "
        "Hold shift to activate time-dilation (Time Warp)."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Assumed frame rate for smooth visuals

        # --- Colors ---
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_ENEMY_RED = (255, 50, 50)
        self.COLOR_ENEMY_GREEN = (50, 255, 50)
        self.COLOR_ENEMY_BLUE = (50, 100, 255)
        self.ENEMY_COLORS = [self.COLOR_ENEMY_RED, self.COLOR_ENEMY_GREEN, self.COLOR_ENEMY_BLUE]
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_TIME_DILATION = (100, 100, 255, 100) # RGBA for transparency

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = None
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.time_dilation_gauge = None
        self.time_dilation_active = False
        self.current_weapon_idx = 0
        self.space_was_held = False
        self.enemy_spawn_timer = 0
        self.difficulty_tier = 0
        self.max_enemies = 5 # Initial max enemies

        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)


        # --- Reset Player State ---
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_health = 100.0

        # --- Reset Game Systems ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.time_dilation_gauge = 100.0
        self.time_dilation_active = False
        self.current_weapon_idx = 0
        self.space_was_held = False
        self.difficulty_tier = 0
        self.enemy_spawn_timer = 0
        self.max_enemies = 5

        # --- Procedural Background ---
        self.stars = []
        for i in range(200):
            depth = self.np_random.uniform(0.1, 1.0)
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'depth': depth,
                'color': tuple(int(c * (0.5 + depth * 0.5)) for c in self.COLOR_UI_TEXT)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        self.steps += 1
        reward = 0.0

        # --- Handle Time Dilation ---
        if shift_held and self.time_dilation_gauge > 0:
            self.time_dilation_active = True
            self.time_dilation_gauge = max(0, self.time_dilation_gauge - 1.5) # Drain rate
        else:
            self.time_dilation_active = False
            self.time_dilation_gauge = min(100, self.time_dilation_gauge + 0.5) # Recharge rate
        
        time_scale = 0.3 if self.time_dilation_active else 1.0

        # --- Update Game Logic ---
        self._update_player(movement, time_scale)
        self._handle_firing(space_held)
        self._update_difficulty()
        self._spawn_enemies(time_scale)

        reward += self._update_projectiles(time_scale)
        self._update_enemies(time_scale)
        self._update_particles(time_scale)

        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, self.COLOR_WHITE, 100)
            # // Play player death sound
        
        if self.steps >= 2000:
            reward += 50 # Survival bonus
            truncated = True # Use truncated for time limit
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_player(self, movement, time_scale):
        speed = 5 * time_scale
        if movement == 1: # Up/Forward
            self.player_pos[1] -= speed
        elif movement == 2: # Down/Backward
            self.player_pos[1] += speed
        elif movement == 3: # Left
            self.player_pos[0] -= speed
        elif movement == 4: # Right
            self.player_pos[0] += speed

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # Auto-switch weapon based on enemy nearest to crosshair
        crosshair_pos = (self.WIDTH / 2, self.HEIGHT / 2)
        min_dist = float('inf')
        target_enemy = None
        for enemy in self.enemies:
            dist = math.hypot(enemy['pos'][0] - crosshair_pos[0], enemy['pos'][1] - crosshair_pos[1])
            if dist < min_dist:
                min_dist = dist
                target_enemy = enemy
        
        if target_enemy and min_dist < 200: # Targeting threshold
            self.current_weapon_idx = target_enemy['color_idx']


    def _handle_firing(self, space_held):
        if space_held and not self.space_was_held:
            # // Play fire sound
            direction = np.array([self.WIDTH / 2, self.HEIGHT / 2]) - np.array(self.player_pos)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            
            self.projectiles.append({
                'pos': list(self.player_pos),
                'vel': direction * 15,
                'color_idx': self.current_weapon_idx,
                'size': 5,
                'glow': 10
            })
        self.space_was_held = space_held
        
    def _update_difficulty(self):
        new_tier = self.steps // 200
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.max_enemies = min(15, 5 + self.difficulty_tier)
            # print(f"Difficulty increased to tier {self.difficulty_tier}") # For debugging

    def _spawn_enemies(self, time_scale):
        self.enemy_spawn_timer -= 1 * time_scale
        spawn_interval = max(15, 60 - self.difficulty_tier * 5)
        if self.enemy_spawn_timer <= 0 and len(self.enemies) < self.max_enemies:
            self.enemy_spawn_timer = spawn_interval
            
            side = self.np_random.choice([0, 1, 2, 3]) # 0:top, 1:right, 2:bottom, 3:left
            if side == 0: pos = [self.np_random.integers(0, self.WIDTH), -20]
            elif side == 1: pos = [self.WIDTH + 20, self.np_random.integers(0, self.HEIGHT)]
            elif side == 2: pos = [self.np_random.integers(0, self.WIDTH), self.HEIGHT + 20]
            else: pos = [-20, self.np_random.integers(0, self.HEIGHT)]
            
            health = 20 * (1.1 ** self.difficulty_tier)
            self.enemies.append({
                'pos': pos,
                'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)],
                'health': health,
                'max_health': health,
                'color_idx': self.np_random.integers(0, 3),
                'size': 15,
                'patrol_timer': self.np_random.integers(60, 121)
            })

    def _update_projectiles(self, time_scale):
        collision_reward = 0
        for p in self.projectiles[:]:
            p['pos'][0] += p['vel'][0] * time_scale
            p['pos'][1] += p['vel'][1] * time_scale

            # Check for collision with enemies
            hit = False
            for e in self.enemies[:]:
                dist = math.hypot(p['pos'][0] - e['pos'][0], p['pos'][1] - e['pos'][1])
                if dist < e['size'] + p['size']:
                    hit = True
                    is_match = p['color_idx'] == e['color_idx']
                    damage = 50 if is_match else 5
                    e['health'] -= damage
                    
                    collision_reward += 0.1 if is_match else -0.1
                    # // Play hit sound (different for match/mismatch)
                    self._create_explosion(p['pos'], self.ENEMY_COLORS[p['color_idx']], 10)

                    if e['health'] <= 0:
                        collision_reward += 1.0
                        self._create_explosion(e['pos'], self.ENEMY_COLORS[e['color_idx']], 40)
                        self.enemies.remove(e)
                        # // Play enemy death sound
                    break
            
            if hit or not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                if p in self.projectiles:
                    self.projectiles.remove(p)
        return collision_reward

    def _update_enemies(self, time_scale):
        for e in self.enemies[:]:
            e['patrol_timer'] -= 1 * time_scale
            if e['patrol_timer'] <= 0:
                # Simple AI: move towards player's general area
                direction = np.array(self.player_pos) - np.array(e['pos'])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    e['vel'] = (direction / norm) * self.np_random.uniform(1.5, 2.5)
                e['patrol_timer'] = self.np_random.integers(90, 181)

            e['pos'][0] += e['vel'][0] * time_scale
            e['pos'][1] += e['vel'][1] * time_scale
            
            # Simple collision with player
            dist = math.hypot(e['pos'][0] - self.player_pos[0], e['pos'][1] - self.player_pos[1])
            if dist < e['size'] + 10: # Player has a small collision radius
                self.player_health -= 10
                self._create_explosion(e['pos'], self.ENEMY_COLORS[e['color_idx']], 20)
                if e in self.enemies:
                    self.enemies.remove(e)


    def _update_particles(self, time_scale):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * time_scale
            p['pos'][1] += p['vel'][1] * time_scale
            p['life'] -= 1 * time_scale
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 31),
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_crosshair()
        if self.time_dilation_active:
            self._render_time_dilation_effect()
        if self.game_over:
            self._render_game_over()

    def _render_background(self):
        for star in self.stars:
            # Parallax effect: stars move opposite to player, faster with depth
            star['pos'][0] = (star['pos'][0] - 0.1 * star['depth']) % self.WIDTH
            star['pos'][1] = (star['pos'][1] - 0.05 * star['depth']) % self.HEIGHT
            
            size = int(1.5 * star['depth'])
            pygame.draw.rect(self.screen, star['color'], (int(star['pos'][0]), int(star['pos'][1]), size, size))

    def _render_enemies(self):
        for e in self.enemies:
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            color = self.ENEMY_COLORS[e['color_idx']]
            size = int(e['size'])

            # Glow effect
            glow_surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (size * 2, size * 2), size * 2)
            self.screen.blit(glow_surf, (pos[0] - size * 2, pos[1] - size * 2))

            # Main body
            points = []
            for i in range(6):
                angle = (math.pi / 3 * i) + (self.steps / 20.0) # Rotation
                points.append((
                    pos[0] + size * math.cos(angle),
                    pos[1] + size * math.sin(angle)
                ))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_WHITE) # White outline

            # Health bar
            if e['health'] < e['max_health']:
                health_pct = e['health'] / e['max_health']
                bar_width = size * 1.5
                bar_pos_y = pos[1] - size - 8
                pygame.draw.rect(self.screen, (100, 0, 0), (pos[0] - bar_width/2, bar_pos_y, bar_width, 4))
                pygame.draw.rect(self.screen, (0, 200, 0), (pos[0] - bar_width/2, bar_pos_y, bar_width * health_pct, 4))

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            color = self.ENEMY_COLORS[p['color_idx']]
            
            # Glow
            glow_surf = pygame.Surface((p['glow'] * 2, p['glow'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 100), (p['glow'], p['glow']), p['glow'])
            self.screen.blit(glow_surf, (pos[0] - p['glow'], pos[1] - p['glow']))
            
            # Core
            pygame.draw.circle(self.screen, self.COLOR_WHITE, pos, p['size'])

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, p['life'] / 30.0)
            color = (*p['color'], int(255 * alpha))
            size = int(p['size'] * alpha)
            if size > 0:
                # Use a surface for per-pixel alpha
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(particle_surf, color, (0,0,size*2,size*2))
                self.screen.blit(particle_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_crosshair(self):
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        color = self.ENEMY_COLORS[self.current_weapon_idx]
        length, gap = 8, 4
        
        # Glow
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 80), (20, 20), 20)
        self.screen.blit(glow_surf, (center_x - 20, center_y - 20))
        
        # Lines
        pygame.draw.line(self.screen, color, (center_x - gap - length, center_y), (center_x - gap, center_y), 2)
        pygame.draw.line(self.screen, color, (center_x + gap, center_y), (center_x + gap + length, center_y), 2)
        pygame.draw.line(self.screen, color, (center_x, center_y - gap - length), (center_x, center_y - gap), 2)
        pygame.draw.line(self.screen, color, (center_x, center_y + gap), (center_x, center_y + gap + length), 2)

    def _render_time_dilation_effect(self):
        vignette_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for i in range(150, 0, -2):
            alpha = int(100 * (1 - (i / 150)))
            pygame.gfxdraw.ellipse(vignette_surf, self.WIDTH//2, self.HEIGHT//2, 
                                   self.WIDTH//2 + i, self.HEIGHT//2 + i, (0,0,20,alpha))
        self.screen.blit(vignette_surf, (0,0))
        
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_TIME_DILATION)
        self.screen.blit(overlay, (0,0))

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # --- Health Gauge ---
        self._draw_gauge(10, self.HEIGHT - 30, 200, 20, self.player_health / 100.0, (200, 0, 0), (0, 200, 0), "HEALTH")

        # --- Time Dilation Gauge ---
        self._draw_gauge(self.WIDTH - 210, self.HEIGHT - 30, 200, 20, self.time_dilation_gauge / 100.0, (50, 50, 50), (100, 100, 255), "TIME WARP")

    def _draw_gauge(self, x, y, w, h, pct, bg_color, fg_color, label):
        pct = np.clip(pct, 0, 1)
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h))
        pygame.draw.rect(self.screen, fg_color, (x, y, int(w * pct), h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (x, y, w, h), 1)
        label_text = self.font_small.render(label, True, self.COLOR_UI_TEXT)
        self.screen.blit(label_text, (x + w/2 - label_text.get_width()/2, y + h/2 - label_text.get_height()/2))
        
    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        status_text = "MISSION COMPLETE" if self.player_health > 0 else "AGENT COMPROMISED"
        text = self.font_large.render(status_text, True, self.COLOR_WHITE)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        self.screen.blit(text, text_rect)
        
        score_text = self.font_small.render(f"FINAL SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "player_health": self.player_health}

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fractal FPS Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Manual Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Render to Screen ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # We need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(env.FPS)
        
    env.close()