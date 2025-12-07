import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:50:36.019656
# Source Brief: brief_01250.md
# Brief Index: 1250
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for particles to create explosion and trail effects
class Particle:
    def __init__(self, x, y, color, life, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.initial_life = life
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        # Add some drag/decay to velocity
        self.vx *= 0.95
        self.vy *= 0.95

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            current_size = int(self.size * (self.life / self.initial_life))
            if current_size > 0:
                # Simple square particle for performance and retro feel
                rect = pygame.Rect(int(self.x - current_size / 2), int(self.y - current_size / 2), current_size, current_size)
                temp_surf = pygame.Surface((current_size, current_size), pygame.SRCALPHA)
                temp_surf.fill((*self.color, alpha))
                surface.blit(temp_surf, rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your position against incoming waves of enemies in this retro-style, top-down arcade shooter."
    )
    user_guide = (
        "Use the arrow keys to aim. Press space to fire and shift to switch weapons."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 40
    PLAYER_POS = (WIDTH // 2, HEIGHT - 30)
    MAX_STEPS = 2500

    # --- Colors (Synthwave Palette) ---
    COLOR_BG = (13, 1, 38)
    COLOR_GRID = (48, 10, 99)
    COLOR_PLAYER = (0, 255, 255)  # Cyan
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_ENEMY_A = (255, 0, 128)  # Magenta
    COLOR_ENEMY_A_GLOW = (150, 0, 75)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_HEALTH_BAR = (0, 255, 0)
    COLOR_HEALTH_BAR_BG = (255, 0, 0)

    WEAPONS = [
        {
            "name": "Laser",
            "color": (255, 255, 0), # Yellow
            "cooldown": 8,
            "projectile_speed": 15,
            "damage": 1,
            "fire": lambda self, aim_vec: self._fire_straight(aim_vec, (255, 255, 0), 1),
        },
        {
            "name": "Spread",
            "color": (0, 255, 128), # Spring Green
            "cooldown": 15,
            "projectile_speed": 12,
            "damage": 1,
            "fire": lambda self, aim_vec: self._fire_spread(aim_vec, (0, 255, 128), 1),
        },
        {
            "name": "Heavy",
            "color": (255, 128, 0), # Orange
            "cooldown": 25,
            "projectile_speed": 8,
            "damage": 3,
            "fire": lambda self, aim_vec: self._fire_straight(aim_vec, (255, 128, 0), 3, size=8),
        },
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # --- Persistent State (across resets) ---
        self.unlocked_weapons_indices = [0]
        
        # --- Initialize state variables ---
        self._initialize_state()

    def _initialize_state(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_health = 100
        self.player_fire_cooldown = 0
        
        self.aim_direction = 0 # 0:Up, 1:Down, 2:Left, 3:Right
        
        self.current_weapon_idx = 0
        
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.enemy_spawn_timer = 0
        self.wave_cleared_timer = 0
        
        self.last_space_held = 0
        self.last_shift_held = 0
        
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        self._start_new_wave()
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action
        
        # 1. Aiming
        if movement > 0: # 1=up, 2=down, 3=left, 4=right
            self.aim_direction = movement - 1

        # 2. Firing (on rising edge of space button)
        fire_pressed = space_held and not self.last_space_held
        if fire_pressed and self.player_fire_cooldown <= 0 and not self.game_over:
            reward += self._fire_weapon()
        self.last_space_held = space_held

        # 3. Weapon Switch (on rising edge of shift button)
        switch_pressed = shift_held and not self.last_shift_held
        if switch_pressed and not self.game_over:
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.unlocked_weapons_indices)
            # sfx: WeaponSwitch
        self.last_shift_held = shift_held

        # --- Update Game State ---
        if not self.game_over:
            # Update player cooldown
            if self.player_fire_cooldown > 0:
                self.player_fire_cooldown -= 1

            # Update projectiles
            for p in self.projectiles[:]:
                p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
                # Trail effect
                if self.steps % 2 == 0:
                    self.particles.append(Particle(p['pos'][0], p['pos'][1], p['color'], 10, p['size']//2, random.uniform(0, 2*math.pi), 0.5))
                # Remove if off-screen
                if not (0 < p['pos'][0] < self.WIDTH and 0 < p['pos'][1] < self.HEIGHT):
                    self.projectiles.remove(p)
                    reward -= 0.01 # Penalty for missing

            # Update enemies
            for e in self.enemies:
                e['pos'] = (e['pos'][0], e['pos'][1] + e['speed'])
                if e['pos'][1] > self.HEIGHT:
                    self.player_health -= 10
                    self.screen_shake = 10
                    # sfx: PlayerDamage
                    e['health'] = 0 # Mark for removal
            
            # Update particles
            for p in self.particles[:]:
                p.update()
                if p.life <= 0:
                    self.particles.remove(p)

            # Collision detection: Projectiles vs Enemies
            for p in self.projectiles[:]:
                proj_rect = pygame.Rect(p['pos'][0] - p['size']/2, p['pos'][1] - p['size']/2, p['size'], p['size'])
                for e in self.enemies[:]:
                    if proj_rect.colliderect(e['rect']):
                        e['health'] -= p['damage']
                        reward += 0.1 # Reward for hitting
                        self._create_explosion(p['pos'], p['color'], 10)
                        # sfx: Hit
                        if p in self.projectiles: self.projectiles.remove(p)
                        break

            # Remove dead enemies
            for e in self.enemies[:]:
                e['rect'] = pygame.Rect(e['pos'][0] - e['size']/2, e['pos'][1] - e['size']/2, e['size'], e['size'])
                if e['health'] <= 0:
                    self.score += 10
                    reward += 1.0 # Reward for destroying
                    self._create_explosion(e['pos'], self.COLOR_ENEMY_A, 30)
                    self.screen_shake = 5
                    # sfx: Explosion
                    self.enemies.remove(e)

            # Wave management
            if not self.enemies and self.wave_cleared_timer == 0:
                self.wave_cleared_timer = 90 # 3 seconds at 30fps
                self.score += self.wave_number * 10
                reward += 5.0 # Reward for clearing wave
            
            if self.wave_cleared_timer > 0:
                self.wave_cleared_timer -= 1
                if self.wave_cleared_timer == 0:
                    self._start_new_wave()
        
        # --- Check Termination ---
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            reward -= 50.0
        
        if self.wave_number > 15 and not self.enemies and not self.game_over:
            self.game_over = True
            self.game_won = True
            reward += 50.0

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _start_new_wave(self):
        self.wave_number += 1
        num_enemies = 3 + self.wave_number
        enemy_health = 1 + (self.wave_number // 3)
        enemy_speed = 0.5 + (self.wave_number * 0.05)
        
        for _ in range(num_enemies):
            size = self.GRID_SIZE * 0.8
            pos = (random.randint(int(size), self.WIDTH - int(size)), random.randint(-self.HEIGHT, -int(size)))
            self.enemies.append({
                'pos': pos,
                'health': enemy_health,
                'max_health': enemy_health,
                'speed': enemy_speed,
                'size': size,
                'rect': pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
            })
            
        # Unlock weapons
        if self.wave_number >= 5 and 1 not in self.unlocked_weapons_indices:
            self.unlocked_weapons_indices.append(1)
        if self.wave_number >= 10 and 2 not in self.unlocked_weapons_indices:
            self.unlocked_weapons_indices.append(2)

    def _fire_weapon(self):
        weapon_template = self.WEAPONS[self.unlocked_weapons_indices[self.current_weapon_idx]]
        self.player_fire_cooldown = weapon_template['cooldown']
        
        aim_vectors = [
            (0, -1), # Up
            (0, 1),  # Down
            (-1, 0), # Left
            (1, 0)   # Right
        ]
        aim_vec = aim_vectors[self.aim_direction]
        
        # Muzzle flash
        self._create_explosion(self.PLAYER_POS, weapon_template['color'], 15, particle_speed=3)
        # sfx: LaserFire
        
        weapon_template['fire'](self, aim_vec)
        return 0 # No immediate reward for firing, only for results

    def _fire_straight(self, aim_vec, color, damage, size=4):
        speed = self.WEAPONS[0]['projectile_speed']
        vel = (aim_vec[0] * speed, aim_vec[1] * speed)
        self.projectiles.append({
            'pos': self.PLAYER_POS, 'vel': vel, 'color': color, 'damage': damage, 'size': size
        })

    def _fire_spread(self, aim_vec, color, damage):
        speed = self.WEAPONS[1]['projectile_speed']
        angle = math.atan2(aim_vec[1], aim_vec[0])
        angles = [angle - 0.2, angle, angle + 0.2]
        for a in angles:
            vel = (math.cos(a) * speed, math.sin(a) * speed)
            self.projectiles.append({
                'pos': self.PLAYER_POS, 'vel': vel, 'color': color, 'damage': damage, 'size': 4
            })
    
    def _create_explosion(self, pos, color, num_particles, particle_life=20, particle_size=4, particle_speed=5):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, particle_speed)
            self.particles.append(Particle(pos[0], pos[1], color, random.randint(10, particle_life), particle_size, angle, speed))

    def _get_observation(self):
        # --- Main Render Loop ---
        render_offset = (0, 0)
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset = (random.randint(-self.screen_shake, self.screen_shake), random.randint(-self.screen_shake, self.screen_shake))

        # 1. Background
        self.screen.fill(self.COLOR_BG)
        self._render_grid(render_offset)

        # 2. Game elements
        self._render_particles(render_offset)
        self._render_projectiles(render_offset)
        self._render_enemies(render_offset)
        self._render_player(render_offset)
        
        # 3. UI
        self._render_ui()
        
        # 4. Game Over Screen
        if self.game_over:
            self._render_game_over()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self, offset):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x + offset[0], 0), (x + offset[0], self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y + offset[1]), (self.WIDTH, y + offset[1]), 1)

    def _render_player(self, offset):
        x, y = self.PLAYER_POS
        x += offset[0]
        y += offset[1]

        # Player base (a triangle)
        points = [(x, y-15), (x-15, y+10), (x+15, y+10)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Aiming indicator
        aim_vectors = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        aim_vec = aim_vectors[self.aim_direction]
        end_pos = (x + aim_vec[0] * 40, y + aim_vec[1] * 40)
        
        # Draw a dashed line for aim
        dash_length = 5
        gap_length = 3
        line_length = math.hypot(end_pos[0] - x, end_pos[1] - y)
        num_dashes = int(line_length / (dash_length + gap_length))
        
        for i in range(num_dashes):
            start = (x + aim_vec[0] * i * (dash_length + gap_length), 
                     y + aim_vec[1] * i * (dash_length + gap_length))
            end = (x + aim_vec[0] * (i * (dash_length + gap_length) + dash_length),
                   y + aim_vec[1] * (i * (dash_length + gap_length) + dash_length))
            pygame.draw.line(self.screen, self.COLOR_PLAYER, start, end, 2)


    def _render_enemies(self, offset):
        for e in self.enemies:
            x, y = int(e['pos'][0] + offset[0]), int(e['pos'][1] + offset[1])
            size = int(e['size'])
            
            # Glow effect
            glow_size = int(size * 1.5)
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_ENEMY_A_GLOW, 50), (0,0, glow_size, glow_size), border_radius=5)
            self.screen.blit(glow_surf, (x - glow_size//2, y - glow_size//2))
            
            # Main body
            rect = pygame.Rect(x - size//2, y - size//2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_A, rect, border_radius=3)
            
            # Health bar
            if e['health'] < e['max_health']:
                health_pct = e['health'] / e['max_health']
                pygame.draw.rect(self.screen, (255,0,0), (x - size//2, y - size//2 - 8, size, 5))
                pygame.draw.rect(self.screen, (0,255,0), (x - size//2, y - size//2 - 8, int(size * health_pct), 5))

    def _render_projectiles(self, offset):
        for p in self.projectiles:
            x, y = int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1])
            size = int(p['size'])
            # Simple rect for projectile core
            pygame.draw.rect(self.screen, (255,255,255), (x-size//2, y-size//2, size, size))
            pygame.draw.rect(self.screen, p['color'], (x-size//2+1, y-size//2+1, size-2, size-2))

    def _render_particles(self, offset):
        # Particles are rendered in world space, not affected by screen shake offset
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / 100)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_pct), 20))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 35))
        
        # Weapon
        weapon = self.WEAPONS[self.unlocked_weapons_indices[self.current_weapon_idx]]
        weapon_text = self.font_ui.render(f"WEAPON: {weapon['name']}", True, weapon['color'])
        self.screen.blit(weapon_text, (10, self.HEIGHT - 30))
        
        if self.wave_cleared_timer > 0:
            clear_text = self.font_game_over.render("WAVE CLEARED", True, self.COLOR_UI_TEXT)
            pos = (self.WIDTH//2 - clear_text.get_width()//2, self.HEIGHT//2 - clear_text.get_height()//2)
            self.screen.blit(clear_text, pos)


    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        text_str = "YOU WIN" if self.game_won else "GAME OVER"
        color = (0, 255, 0) if self.game_won else (255, 0, 0)
        
        game_over_text = self.font_game_over.render(text_str, True, color)
        pos = (self.WIDTH // 2 - game_over_text.get_width() // 2, self.HEIGHT // 2 - game_over_text.get_height() // 2 - 20)
        self.screen.blit(game_over_text, pos)
        
        final_score_text = self.font_ui.render(f"FINAL SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        pos = (self.WIDTH // 2 - final_score_text.get_width() // 2, self.HEIGHT // 2 + 30)
        self.screen.blit(final_score_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
        }

    def close(self):
        pygame.quit()

# Example usage to test the environment
if __name__ == '__main__':
    # To run with display, comment out the os.environ line at the top
    # and uncomment the following line
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Neon Grid Defender")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0

    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- Keyboard to MultiDiscrete Action Mapping ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()