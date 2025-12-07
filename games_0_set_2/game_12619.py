import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:09:41.357147
# Source Brief: brief_02619.md
# Brief Index: 2619
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your mystical orbs from waves of incoming enemies. "
        "Charge up and unleash powerful swipes to destroy attackers and achieve a high score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to swipe at enemies. Hold 'space' to charge your attack and shrink orbs for defense. Hold 'shift' to expand orbs."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG_TOP = (10, 5, 25)
    COLOR_BG_BOTTOM = (2, 0, 10)
    COLOR_ORB_BLUE = (0, 150, 255)
    COLOR_ORB_GREEN = (50, 255, 150)
    COLOR_ORB_PURPLE = (200, 100, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_SWIPE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_COMBO = (255, 200, 0)
    
    MAX_STEPS = 3000
    MAX_WAVES = 10

    # --- ENTITY CLASSES ---
    class Orb:
        def __init__(self, pos, color, pulse_speed, pulse_magnitude):
            self.pos = np.array(pos, dtype=float)
            self.color = color
            self.max_health = 100
            self.health = self.max_health
            self.base_size = 25
            self.size_modifier = 1.0  # Player controlled
            self.current_size = self.base_size
            self.pulse_timer = random.uniform(0, 2 * math.pi)
            self.pulse_speed = pulse_speed
            self.pulse_magnitude = pulse_magnitude
            self.defense = 1.0 # 1.0 = normal damage, >1.0 = less damage

        def update(self, dt):
            self.pulse_timer += self.pulse_speed * dt
            pulse_effect = 1.0 + math.sin(self.pulse_timer) * self.pulse_magnitude
            self.current_size = self.base_size * self.size_modifier * pulse_effect
            self.defense = 1.0 + max(0, 1.0 - self.size_modifier) * 2.0 # Smaller = more defense

        def take_damage(self, amount):
            self.health = max(0, self.health - amount / self.defense)
            return amount / self.defense

        def draw(self, surface):
            # Health ring
            if self.health < self.max_health:
                health_angle = 360 * (self.health / self.max_health)
                arc_rect = pygame.Rect(
                    self.pos[0] - self.current_size - 6, self.pos[1] - self.current_size - 6,
                    (self.current_size + 6) * 2, (self.current_size + 6) * 2
                )
                pygame.draw.arc(surface, (100, 100, 100), arc_rect, math.radians(90), math.radians(450), 3)
                pygame.draw.arc(surface, self.color, arc_rect, math.radians(90), math.radians(90 + health_angle), 3)

            # Glow effect
            for i in range(10, 0, -2):
                alpha = 50 - i * 5
                glow_color = (*self.color, alpha)
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos[0]), int(self.pos[1]),
                    int(self.current_size + i), glow_color
                )
            # Main orb
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_size), self.color)
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_size), self.color)

    class Enemy:
        def __init__(self, pos, speed, target_orb_idx):
            self.pos = np.array(pos, dtype=float)
            self.max_health = 10
            self.health = self.max_health
            self.speed = speed
            self.size = 8
            self.target_orb_idx = target_orb_idx
            self.hit_timer = 0

        def update(self, dt, target_pos):
            direction = target_pos - self.pos
            dist = np.linalg.norm(direction)
            if dist > 1:
                direction /= dist
            self.pos += direction * self.speed * dt
            if self.hit_timer > 0:
                self.hit_timer -= dt

        def draw(self, surface):
            color = GameEnv.COLOR_ENEMY
            if self.hit_timer > 0:
                # Flash white when hit
                t = self.hit_timer / 0.2
                color = (
                    int(GameEnv.COLOR_ENEMY[0] + (255 - GameEnv.COLOR_ENEMY[0]) * t),
                    int(GameEnv.COLOR_ENEMY[1] + (255 - GameEnv.COLOR_ENEMY[1]) * t),
                    int(GameEnv.COLOR_ENEMY[2] + (255 - GameEnv.COLOR_ENEMY[2]) * t),
                )
            
            # Body
            points = [
                (self.pos[0], self.pos[1] - self.size),
                (self.pos[0] - self.size * 0.8, self.pos[1] + self.size * 0.8),
                (self.pos[0] + self.size * 0.8, self.pos[1] + self.size * 0.8),
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

            # Health bar
            if self.health < self.max_health:
                bar_width = 20
                bar_height = 4
                bar_x = self.pos[0] - bar_width / 2
                bar_y = self.pos[1] - self.size - 10
                health_ratio = self.health / self.max_health
                pygame.draw.rect(surface, (80, 0, 0), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(surface, color, (bar_x, bar_y, bar_width * health_ratio, bar_height))

    class Swipe:
        def __init__(self, start_pos, end_pos, damage, combo):
            self.start_pos = np.array(start_pos, dtype=float)
            self.end_pos = np.array(end_pos, dtype=float)
            self.damage = damage * (1 + 0.1 * combo)
            self.lifetime = 0.2
            self.hit_enemies = set()

        def update(self, dt):
            self.lifetime -= dt

        def draw(self, surface):
            alpha = max(0, 255 * (self.lifetime / 0.2))
            width = int(5 + 15 * (self.lifetime / 0.2))
            pygame.draw.line(surface, (*GameEnv.COLOR_SWIPE, alpha), self.start_pos, self.end_pos, width)

    class Particle:
        def __init__(self, pos, vel, size, color, lifetime):
            self.pos = np.array(pos, dtype=float)
            self.vel = np.array(vel, dtype=float)
            self.size = size
            self.color = color
            self.lifetime = lifetime
            self.initial_lifetime = lifetime

        def update(self, dt):
            self.pos += self.vel * dt
            self.lifetime -= dt
            self.vel *= 0.95 # Damping

        def draw(self, surface):
            alpha = max(0, 255 * (self.lifetime / self.initial_lifetime))
            size = int(self.size * (self.lifetime / self.initial_lifetime))
            if size > 0:
                rect = pygame.Rect(self.pos[0] - size/2, self.pos[1] - size/2, size, size)
                pygame.draw.rect(surface, (*self.color, alpha), rect)

    # --- GYM ENV IMPLEMENTATION ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_combo = pygame.font.SysFont("Verdana", 32, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.orbs = []
        self.enemies = []
        self.swipes = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_transition_timer = 0
        
        self.charge_level = 0.0
        self.combo_count = 0
        self.combo_timer = 0
        self.avg_orb_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2])
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.orbs = [
            self.Orb((self.WIDTH * 0.5, self.HEIGHT * 0.5), self.COLOR_ORB_BLUE, 0.05, 0.1),
            self.Orb((self.WIDTH * 0.3, self.HEIGHT * 0.6), self.COLOR_ORB_GREEN, 0.07, 0.08),
            self.Orb((self.WIDTH * 0.7, self.HEIGHT * 0.4), self.COLOR_ORB_PURPLE, 0.04, 0.12),
        ]
        self.avg_orb_pos = np.mean([o.pos for o in self.orbs], axis=0)

        self.enemies = []
        self.swipes = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_transition_timer = 3.0 # Start first wave after 3 seconds
        
        self.charge_level = 0.0
        self.combo_count = 0
        self.combo_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        dt = 1 / 30.0 # Assume 30 FPS for physics
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle Player Input
        orb_size_mod_change = 0
        if space_held:
            orb_size_mod_change = -0.05
            self.charge_level = min(1.0, self.charge_level + 0.02)
        elif shift_held:
            orb_size_mod_change = 0.05
        
        for orb in self.orbs:
            orb.size_modifier = np.clip(orb.size_modifier + orb_size_mod_change, 0.5, 1.5)

        if movement != 0:
            # Trigger swipe
            swipe_len = 150
            start_pos = self.avg_orb_pos
            if movement == 1: end_pos = start_pos + np.array([0, -swipe_len]) # Up
            elif movement == 2: end_pos = start_pos + np.array([0, swipe_len]) # Down
            elif movement == 3: end_pos = start_pos + np.array([-swipe_len, 0]) # Left
            else: end_pos = start_pos + np.array([swipe_len, 0]) # Right
            
            base_damage = 5 + 15 * self.charge_level
            self.swipes.append(self.Swipe(start_pos, end_pos, base_damage, self.combo_count))
            self.charge_level = 0.0 # Consume charge
            # sound: player_swipe.wav

        # 2. Update Game Entities
        for orb in self.orbs:
            orb.update(dt)
        
        for swipe in self.swipes[:]:
            swipe.update(dt)
            if swipe.lifetime <= 0:
                self.swipes.remove(swipe)

        for enemy in self.enemies:
            if enemy.target_orb_idx < len(self.orbs) and self.orbs[enemy.target_orb_idx].health > 0:
                target_orb = self.orbs[enemy.target_orb_idx]
            else: # Retarget if orb is destroyed
                alive_orbs = [i for i, o in enumerate(self.orbs) if o.health > 0]
                if not alive_orbs: break
                enemy.target_orb_idx = random.choice(alive_orbs)
                target_orb = self.orbs[enemy.target_orb_idx]
            enemy.update(dt, target_orb.pos)

        for p in self.particles[:]:
            p.update(dt)
            if p.lifetime <= 0:
                self.particles.remove(p)

        # 3. Handle Collisions
        # Swipes vs Enemies
        for swipe in self.swipes:
            for i, enemy in enumerate(self.enemies):
                if i not in swipe.hit_enemies:
                    if self._line_circle_collision(swipe.start_pos, swipe.end_pos, enemy.pos, enemy.size):
                        swipe.hit_enemies.add(i)
                        enemy.health -= swipe.damage
                        enemy.hit_timer = 0.2
                        reward += 0.1
                        self.combo_count += 1
                        self.combo_timer = 2.0 # Reset combo timeout
                        # sound: enemy_hit.wav
                        self._create_particles(enemy.pos, 5, self.COLOR_SWIPE, 2, 0.5)

        # Enemies vs Orbs
        for enemy in self.enemies[:]:
            target_orb = self.orbs[enemy.target_orb_idx]
            if target_orb.health > 0:
                dist = np.linalg.norm(enemy.pos - target_orb.pos)
                if dist < enemy.size + target_orb.current_size:
                    damage_taken = target_orb.take_damage(20)
                    reward -= 0.1
                    self.combo_count = 0 # Break combo
                    # sound: orb_damage.wav
                    self._create_particles(target_orb.pos, 10, target_orb.color, 3, 0.8)
                    self.enemies.remove(enemy)

        # 4. Process destroyed enemies
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                reward += 1.0
                self.score += 10 + self.combo_count
                # sound: enemy_destroy.wav
                self._create_particles(enemy.pos, 20, self.COLOR_ENEMY, 4, 1.0)
                self.enemies.remove(enemy)
        
        # 5. Update Combo
        if self.combo_timer > 0:
            self.combo_timer -= dt
        else:
            self.combo_count = 0

        # 6. Wave Management
        if not self.enemies and not self.game_won:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= dt
            else:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.game_won = True
                    self.game_over = True
                else:
                    reward += 5.0
                    self._spawn_wave()
                    self.wave_transition_timer = 3.0 # Cooldown between waves

        # 7. Check Termination Conditions
        terminated = False
        truncated = False
        if sum(o.health for o in self.orbs) <= 0:
            reward = -100.0
            self.game_over = True
            terminated = True
            # sound: game_over.wav
        elif self.game_won:
            reward = 100.0
            terminated = True
            # sound: game_win.wav
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Gymnasium standard is to set both to True on truncation

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_wave(self):
        num_enemies = self.current_wave * 2 + 3
        speed = 0.5 + self.current_wave * 0.05
        for _ in range(num_enemies):
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = [random.uniform(0, self.WIDTH), -20]
            elif edge == 'bottom': pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 20]
            elif edge == 'left': pos = [-20, random.uniform(0, self.HEIGHT)]
            else: pos = [self.WIDTH + 20, random.uniform(0, self.HEIGHT)]
            
            alive_orbs = [i for i, o in enumerate(self.orbs) if o.health > 0]
            if not alive_orbs: break
            target_idx = random.choice(alive_orbs)
            self.enemies.append(self.Enemy(pos, speed, target_idx))

    def _create_particles(self, pos, count, color, size, lifetime):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(20, 100)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            p_lifetime = lifetime * random.uniform(0.7, 1.3)
            p_size = size * random.uniform(0.7, 1.3)
            self.particles.append(self.Particle(pos, vel, p_size, color, p_lifetime))

    def _line_circle_collision(self, p1, p2, c, r):
        p1 = p1 - c
        p2 = p2 - c
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dr2 = dx*dx + dy*dy
        D = p1[0]*p2[1] - p2[0]*p1[1]
        
        if dr2 == 0: return np.linalg.norm(p1) < r # Swipe is a point
        
        delta = r*r * dr2 - D*D
        if delta < 0: return False # No intersection
        
        # Check if segment is near the circle
        dot1 = np.dot(p2-p1, -p1)
        dot2 = np.dot(p1-p2, -p2)
        if dot1 < 0 or dot2 < 0:
            return np.linalg.norm(p1) < r or np.linalg.norm(p2) < r
            
        return True

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        for p in self.particles: p.draw(self.screen)
        for s in self.swipes: s.draw(self.screen)
        for orb in self.orbs:
            if orb.health > 0: orb.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        # Charge meter
        charge_w = 100
        charge_h = 10
        charge_x = self.WIDTH/2 - charge_w/2
        charge_y = self.HEIGHT - 30
        pygame.draw.rect(self.screen, (50, 50, 80), (charge_x, charge_y, charge_w, charge_h), 1)
        fill_w = charge_w * self.charge_level
        if fill_w > 0:
            pygame.draw.rect(self.screen, self.COLOR_SWIPE, (charge_x, charge_y, fill_w, charge_h))
        # Combo
        if self.combo_count > 1:
            combo_str = f"{self.combo_count}x COMBO"
            text_surf = self.font_combo.render(combo_str, True, self.COLOR_COMBO)
            t = min(1.0, self.combo_timer / 0.5) # Fade in/out
            scale = 1.0 + (1.0 - t) * 0.2 # Pop effect
            scaled_surf = pygame.transform.smoothscale_by(text_surf, scale)
            scaled_surf.set_alpha(int(255 * t))
            pos = (self.WIDTH/2 - scaled_surf.get_width()/2, self.HEIGHT/2 - scaled_surf.get_height()/2 - 50)
            self.screen.blit(scaled_surf, pos)
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU SURVIVED" if self.game_won else "ORBS DESTROYED"
            text = self.font_large.render(msg, True, self.COLOR_SWIPE)
            text_pos = (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2)
            self.screen.blit(text, text_pos)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "combo": self.combo_count,
            "orbs_health": [o.health for o in self.orbs]
        }
        
    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The following code is for human interaction and visualization.
    # It is not part of the required Gymnasium environment implementation.
    # Un-set the headless environment variable to allow for display.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame window for human play ---
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Orb Defender")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        clock.tick(30)

    env.close()
    pygame.quit()