import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:22:50.821472
# Source Brief: brief_00974.md
# Brief Index: 974
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Particle:
    def __init__(self, pos, vel, color, size, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifetime -= 1
        self.size = max(0, self.size * (self.lifetime / self.max_lifetime))

    def draw(self, surface):
        if self.lifetime > 0:
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos[0]), int(self.pos[1]),
                int(self.size), self.color
            )
            pygame.gfxdraw.aacircle(
                surface, int(self.pos[0]), int(self.pos[1]),
                int(self.size), self.color
            )

class Projectile:
    def __init__(self, pos, target_pos, element, damage, speed=8):
        self.pos = np.array(pos, dtype=float)
        self.color = GameEnv.ELEMENT_COLORS[element]
        self.element = element
        self.damage = damage
        self.radius = 4
        
        angle = math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
        self.vel = np.array([math.cos(angle), math.sin(angle)]) * speed

    def update(self):
        self.pos += self.vel

    def draw(self, surface):
        # Trail effect
        for i in range(3):
            p = self.pos - self.vel * (i * 0.4)
            alpha = 255 - i * 80
            color = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(p[0]), int(p[1]), max(1, self.radius - i*1), color)
        
        # Core projectile
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), self.radius, self.color)


class Golem:
    def __init__(self, pos, zone_idx, golem_type, upgrades):
        self.pos = np.array(pos, dtype=float)
        self.visual_pos = np.array(pos, dtype=float)
        self.zone_idx = zone_idx
        self.golem_type = golem_type
        
        self.max_health = 100
        self.health = self.max_health
        self.element = 'fire' # Default element
        self.radius = 12
        
        self.attack_range = 150 if self.golem_type == 'melee' else 250
        self.base_damage = 10 if self.golem_type == 'melee' else 7
        self.attack_cooldown = 0
        self.max_cooldown = 45 if self.golem_type == 'melee' else 35
        self.upgrades = upgrades
        self.target = None
        self.anim_angle = random.uniform(0, 2 * math.pi)

    def get_damage(self):
        return self.base_damage * self.upgrades.get(f'{self.element}_damage_bonus', 1.0)

    def update(self, enemies):
        # Smooth visual movement
        self.visual_pos += (self.pos - self.visual_pos) * 0.2
        self.anim_angle += 0.05
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Find target
        self.target = None
        min_dist = self.attack_range
        for enemy in enemies:
            dist = np.linalg.norm(self.pos - enemy.pos)
            if dist < min_dist:
                min_dist = dist
                self.target = enemy
    
    def draw(self, surface):
        x, y = int(self.visual_pos[0]), int(self.visual_pos[1])
        color = GameEnv.ELEMENT_COLORS[self.element]

        # Glow effect
        for i in range(4):
            alpha = 40 - i * 10
            radius = self.radius + i * 3
            pygame.gfxdraw.filled_circle(surface, x, y, radius, (*color, alpha))

        # Orbiting crystals
        for i in range(3):
            angle = self.anim_angle + i * (2 * math.pi / 3)
            ox = x + math.cos(angle) * (self.radius + 2)
            oy = y + math.sin(angle) * (self.radius + 2)
            pygame.gfxdraw.filled_circle(surface, int(ox), int(oy), 3, color)

        # Main body
        pygame.draw.polygon(surface, color, [
            (x, y - self.radius),
            (x - self.radius * 0.866, y + self.radius * 0.5),
            (x + self.radius * 0.866, y + self.radius * 0.5)
        ])
        
        # Health bar
        if self.health < self.max_health:
            bar_w = self.radius * 2
            bar_h = 4
            bar_x = x - self.radius
            bar_y = y - self.radius - 8
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surface, (100, 255, 100), (bar_x, bar_y, int(bar_w * health_pct), bar_h))

class Enemy:
    def __init__(self, pos, wave_number):
        self.pos = np.array(pos, dtype=float)
        self.visual_pos = np.array(pos, dtype=float)
        
        enemy_types = list(GameEnv.ENEMY_VULNERABILITIES.keys())
        self.type = random.choice(enemy_types)
        self.color = GameEnv.ENEMY_COLORS[self.type]
        self.vulnerability = GameEnv.ENEMY_VULNERABILITIES[self.type]
        
        difficulty_mod = 1 + (wave_number - 1) * 0.05
        self.max_health = 20 * difficulty_mod
        self.health = self.max_health
        self.damage = 5 * difficulty_mod
        self.speed = random.uniform(0.6, 1.0)
        self.radius = 10
        self.attack_range = 25
        self.target = None
        self.anim_pulse = 0
        
    def update(self, golems):
        self.visual_pos += (self.pos - self.visual_pos) * 0.2
        self.anim_pulse = (self.anim_pulse + 0.1) % (2 * math.pi)

        # Find target
        self.target = None
        min_dist = float('inf')
        if golems:
            for golem in golems:
                dist = np.linalg.norm(self.pos - golem.pos)
                if dist < min_dist:
                    min_dist = dist
                    self.target = golem
        
        # Movement
        if self.target and min_dist > self.attack_range:
            direction = (self.target.pos - self.pos) / min_dist
            self.pos += direction * self.speed
    
    def draw(self, surface):
        x, y = int(self.visual_pos[0]), int(self.visual_pos[1])
        pulse_size = self.radius + math.sin(self.anim_pulse) * 2

        # Simple geometric shape
        if self.type == 'purple': # Square
             pygame.draw.rect(surface, self.color, (x - pulse_size, y - pulse_size, pulse_size*2, pulse_size*2))
        elif self.type == 'orange': # Triangle
            pygame.draw.polygon(surface, self.color, [(x, y - pulse_size), (x - pulse_size, y + pulse_size), (x + pulse_size, y + pulse_size)])
        elif self.type == 'brown': # Diamond
            pygame.draw.polygon(surface, self.color, [(x, y - pulse_size), (x - pulse_size, y), (x, y + pulse_size), (x + pulse_size, y)])
        else: # Gray - Circle
            pygame.gfxdraw.filled_circle(surface, x, y, int(pulse_size), self.color)
            pygame.gfxdraw.aacircle(surface, x, y, int(pulse_size), self.color)
        
        # Health bar
        if self.health < self.max_health:
            bar_w = self.radius * 2
            bar_h = 4
            bar_x = x - self.radius
            bar_y = y - self.radius - 8
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surface, (255, 100, 100), (bar_x, bar_y, int(bar_w * health_pct), bar_h))


# --- Main Game Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend against waves of enemies by strategically placing elemental Golems. "
        "Switch your Golems' element on the fly to exploit enemy weaknesses and survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a spawn zone and press space to build a Golem. "
        "Press shift to cycle the elemental alignment of all your Golems."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 30000 # Increased for longer gameplay
    MAX_WAVES = 20
    GOLEM_COST = 50
    
    COLOR_BG = (15, 10, 25)
    COLOR_GRID = (30, 20, 50)
    COLOR_UI_TEXT = (220, 220, 255)
    
    ELEMENTS = ['fire', 'water', 'earth', 'lightning']
    ELEMENT_COLORS = {
        'fire': (255, 100, 80), 'water': (80, 150, 255),
        'earth': (100, 220, 100), 'lightning': (255, 255, 100)
    }
    ENEMY_VULNERABILITIES = {
        'purple': 'fire', 'orange': 'water', 'brown': 'earth', 'gray': 'lightning'
    }
    ENEMY_COLORS = {
        'purple': (200, 100, 255), 'orange': (255, 165, 0),
        'brown': (139, 69, 19), 'gray': (150, 150, 150)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        
        self.zones = [
            pygame.Rect(self.WIDTH/2 - 50, 50, 100, 100),    # Up
            pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 150, 100, 100), # Down
            pygame.Rect(50, self.HEIGHT/2 - 50, 100, 100),   # Left
            pygame.Rect(self.WIDTH - 150, self.HEIGHT/2 - 50, 100, 100) # Right
        ]
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.golems = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_number = 0
        self.resources = 100
        self.resource_tick = 0
        
        self.available_golem_types = ['melee']
        self.upgrades = {}
        
        self.last_space_held = False
        self.last_shift_held = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if space_pressed:
            zone_idx = movement - 1 if movement > 0 else random.randint(0, 3)
            if self._spawn_golem(zone_idx):
                pass # Successful spawn, no immediate reward

        if shift_pressed and self.golems:
            # SFX: Element_Switch.wav
            current_element_idx = self.ELEMENTS.index(self.golems[0].element)
            next_element_idx = (current_element_idx + 1) % len(self.ELEMENTS)
            next_element = self.ELEMENTS[next_element_idx]
            for golem in self.golems:
                golem.element = next_element

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- 2. Update Game Logic ---
        self.resource_tick += 1
        if self.resource_tick >= 30: # 1 resource per second at 30fps
            self.resources += 1
            self.resource_tick = 0

        for golem in self.golems:
            golem.update(self.enemies)
            if golem.target and golem.attack_cooldown == 0:
                self._spawn_projectile(golem)
                golem.attack_cooldown = golem.max_cooldown

        for enemy in self.enemies:
            enemy.update(self.golems)
            if enemy.target and np.linalg.norm(enemy.pos - enemy.target.pos) <= enemy.attack_range:
                # SFX: Golem_Hit.wav
                enemy.target.health -= enemy.damage
                reward -= 0.1
                self._create_impact_particles(enemy.target.visual_pos, (255, 50, 50))

        for p in self.projectiles: p.update()
        for p in self.particles: p.update()

        # --- 3. Handle Collisions & Events ---
        new_projectiles = []
        for proj in self.projectiles:
            hit = False
            for enemy in self.enemies:
                if np.linalg.norm(proj.pos - enemy.pos) < enemy.radius + proj.radius:
                    # SFX: Enemy_Hit.wav
                    damage = proj.damage
                    if proj.element == enemy.vulnerability:
                        damage *= 2.0 # Double damage for elemental weakness
                    enemy.health -= damage
                    reward += 0.1
                    self._create_impact_particles(enemy.visual_pos, proj.color)
                    hit = True
                    break
            if not hit and 0 < proj.pos[0] < self.WIDTH and 0 < proj.pos[1] < self.HEIGHT:
                new_projectiles.append(proj)
        self.projectiles = new_projectiles

        # --- 4. Remove dead entities and grant rewards ---
        alive_enemies = []
        for enemy in self.enemies:
            if enemy.health > 0:
                alive_enemies.append(enemy)
            else:
                # SFX: Enemy_Destroyed.wav
                reward += 1.0
                self.score += 10
                self.resources += 15
                self._create_explosion(enemy.visual_pos, enemy.color)
        self.enemies = alive_enemies

        alive_golems = []
        for golem in self.golems:
            if golem.health > 0:
                alive_golems.append(golem)
            else:
                # SFX: Golem_Destroyed.wav
                reward -= 1.0
                self._create_explosion(golem.visual_pos, GameEnv.ELEMENT_COLORS[golem.element])
        self.golems = alive_golems

        # --- 5. Check game state transitions ---
        if not self.enemies and self.wave_number <= self.MAX_WAVES:
            self._start_next_wave()
        
        terminated = False
        if self.wave_number > self.MAX_WAVES: # Victory
            reward += 100
            self.score += 1000
            terminated = True
        elif not self.golems and self.resources < self.GOLEM_COST: # Defeat
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        # --- 6. Cleanup ---
        self.particles = [p for p in self.particles if p.lifetime > 0]
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    # --- Helper Methods ---

    def _spawn_golem(self, zone_idx):
        if self.resources >= self.GOLEM_COST:
            # SFX: Golem_Spawn.wav
            self.resources -= self.GOLEM_COST
            zone_rect = self.zones[zone_idx]
            pos = (zone_rect.centerx + random.uniform(-20, 20), zone_rect.centery + random.uniform(-20, 20))
            
            golem_type = 'ranged' if 'ranged' in self.available_golem_types else 'melee'
            new_golem = Golem(pos, zone_idx, golem_type, self.upgrades)
            
            if self.golems: # Match element of existing golems
                new_golem.element = self.golems[0].element
            
            self.golems.append(new_golem)
            self._create_spawn_effect(pos)
            return True
        return False

    def _spawn_projectile(self, golem):
        # SFX: Projectile_Launch.wav
        if golem.target:
            proj = Projectile(golem.pos, golem.target.pos, golem.element, golem.get_damage())
            self.projectiles.append(proj)

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        # Check for unlocks
        if self.wave_number in [5, 10, 15] and 'ranged' not in self.available_golem_types:
            self.available_golem_types.append('ranged')
        if self.wave_number in [2, 7, 12, 17]:
            element_to_upgrade = random.choice(self.ELEMENTS)
            key = f'{element_to_upgrade}_damage_bonus'
            self.upgrades[key] = self.upgrades.get(key, 1.0) + 0.1
        
        num_enemies = 3 + self.wave_number
        for _ in range(num_enemies):
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top': pos = (random.uniform(0, self.WIDTH), -20)
            elif side == 'bottom': pos = (random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif side == 'left': pos = (-20, random.uniform(0, self.HEIGHT))
            else: pos = (self.WIDTH + 20, random.uniform(0, self.HEIGHT))
            self.enemies.append(Enemy(pos, self.wave_number))

    # --- Particle Effects ---
    def _create_impact_particles(self, pos, color):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.uniform(2, 4)
            lifetime = random.randint(10, 20)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _create_explosion(self, pos, color):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = random.uniform(2, 6)
            lifetime = random.randint(20, 40)
            self.particles.append(Particle(pos, vel, color, size, lifetime))

    def _create_spawn_effect(self, pos):
        for i in range(20):
            angle = (i / 20) * 2 * math.pi
            speed = 2
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(Particle(pos, vel, (200, 200, 255), 3, 15))

    # --- Rendering Methods ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_game(self):
        for particle in self.particles: particle.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for golem in self.golems: golem.draw(self.screen)

    def _render_ui(self):
        # Resources
        res_text = self.font_m.render(f"CRYSTALS: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(res_text, (10, 10))

        # Wave
        wave_text = self.font_m.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_s.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Current Element
        if self.golems:
            element = self.golems[0].element.upper()
            color = self.ELEMENT_COLORS[self.golems[0].element]
            element_text = self.font_m.render(f"ELEMENT: {element}", True, color)
            self.screen.blit(element_text, (self.WIDTH // 2 - element_text.get_width() // 2, 10))
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "golems": len(self.golems)}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation failed: {e}")
        
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space = 0
        shift = 0
        
        # Create a display if one doesn't exist for human play
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise AttributeError
        except (pygame.error, AttributeError):
            display_surf = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering for Human ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before restarting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()