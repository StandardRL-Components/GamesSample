
# Generated: 2025-08-28T02:25:33.434783
# Source Brief: brief_04446.md
# Brief Index: 4446

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Unit:
    def __init__(self, pos, unit_type, level):
        self.pos = pos
        self.type = unit_type
        self.level = level
        self.cooldown = 0
        self.set_stats()

    def set_stats(self):
        if self.type == "Shooter":
            self.range = 150 + (self.level - 1) * 10
            self.fire_rate = 20 - (self.level - 1)
            self.damage = 10 + (self.level - 1) * 2
        elif self.type == "Cannon":
            self.range = 200 + (self.level - 1) * 15
            self.fire_rate = 60 - (self.level - 1) * 3
            self.damage = 30 + (self.level - 1) * 5
    
    def upgrade(self):
        self.level += 1
        self.set_stats()

class Enemy:
    def __init__(self, pos, wave):
        self.pos = np.array(pos, dtype=float)
        self.max_health = 20 * (1.05 ** (wave - 1))
        self.health = self.max_health
        self.speed = 1.0 * (1.02 ** (wave - 1))
        self.radius = 8

class Projectile:
    def __init__(self, start_pos, target_enemy, damage):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.damage = damage
        self.speed = 10.0
        
        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.velocity = (direction / distance) * self.speed
        else:
            self.velocity = np.array([0, -self.speed])

class Particle:
    def __init__(self, pos, color, life, size, velocity_spread):
        self.pos = np.array(pos, dtype=float)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, velocity_spread)
        self.velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press space to cycle through actions (Summon Unit, Upgrade). Press space again to confirm."
    )

    game_description = (
        "A minimalist tower defense game. Survive 20 waves by strategically summoning units and upgrading your tower."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_TOWER = (70, 120, 180)
        self.COLOR_TOWER_GLOW = (70, 120, 180, 50)
        self.COLOR_UNIT_SHOOTER = (0, 200, 150)
        self.COLOR_UNIT_CANNON = (200, 150, 0)
        self.COLOR_ENEMY = (210, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (40, 180, 100)
        self.COLOR_HEALTH_RED = (180, 40, 40)

        # Fonts
        self.font_s = pygame.font.SysFont("Arial", 16)
        self.font_m = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_l = pygame.font.SysFont("Arial", 32, bold=True)
        
        # Game constants
        self.MAX_STEPS = 30 * 180 # 3 minutes at 30fps
        self.TOWER_POS = (self.WIDTH // 2, self.HEIGHT // 2)
        self.MAX_TOWER_HEALTH = 100
        self.MAX_WAVES = 20
        self.UNIT_ORBIT_RADIUS = 50
        
        # Initialize state variables
        self.tower_health = None
        self.current_wave = None
        self.wave_timer = None
        self.is_wave_active = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.enemies = []
        self.units = []
        self.projectiles = []
        self.particles = []
        self.tower_unit_level = 1
        self.max_units = 3

        # Action handling state
        self.action_options = ["Summon Shooter", "Summon Cannon", "Upgrade Units"]
        self.selection_index = 0
        self.is_confirming_action = False
        self.last_space_press = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.tower_health = self.MAX_TOWER_HEALTH
        self.current_wave = 0
        self.wave_timer = 90  # 3 seconds until first wave
        self.is_wave_active = False
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.enemies.clear()
        self.units.clear()
        self.projectiles.clear()
        self.particles.clear()

        self.tower_unit_level = 1
        self.max_units = 3
        
        self.selection_index = 0
        self.is_confirming_action = False
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        self.steps += 1
        reward = -0.001 # Small penalty for existing

        self._handle_input(action)
        reward += self._update_game_state()
        
        terminated = self.tower_health <= 0 or self.current_wave > self.MAX_WAVES or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.tower_health <= 0:
                reward -= 100
            elif self.current_wave > self.MAX_WAVES:
                reward += 100
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        space_held = action[1] == 1
        is_new_press = space_held and not self.last_space_press

        if is_new_press:
            if not self.is_confirming_action:
                self.is_confirming_action = True
                self.selection_index = (self.selection_index + 1) % len(self.action_options)
            else: # Confirm action
                action_name = self.action_options[self.selection_index]
                if "Summon" in action_name and len(self.units) < self.max_units:
                    unit_type = action_name.split(" ")[1]
                    self._summon_unit(unit_type)
                    # sfx: summon_unit
                elif "Upgrade" in action_name:
                    self._upgrade_units()
                    # sfx: upgrade_success
                self.is_confirming_action = False
        
        self.last_space_press = space_held

    def _summon_unit(self, unit_type):
        num_units = len(self.units)
        angle = (2 * math.pi / self.max_units) * num_units
        x = self.TOWER_POS[0] + self.UNIT_ORBIT_RADIUS * math.cos(angle)
        y = self.TOWER_POS[1] + self.UNIT_ORBIT_RADIUS * math.sin(angle)
        self.units.append(Unit((x, y), unit_type, self.tower_unit_level))
    
    def _upgrade_units(self):
        self.tower_unit_level += 1
        for unit in self.units:
            unit.upgrade()
        # Increase max units every 3 levels
        if self.tower_unit_level % 3 == 0:
            self.max_units += 1

    def _update_game_state(self):
        step_reward = 0

        # Wave management
        if not self.is_wave_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave <= self.MAX_WAVES:
                    self._spawn_wave()
                    self.is_wave_active = True
        elif len(self.enemies) == 0:
            self.is_wave_active = False
            self.wave_timer = 150 # 5 second break
            step_reward += 5 # Wave clear bonus
            self.score += 100 * self.current_wave

        # Update units
        for unit in self.units:
            unit.cooldown = max(0, unit.cooldown - 1)
            if unit.cooldown == 0:
                closest_enemy = None
                min_dist = unit.range ** 2
                for enemy in self.enemies:
                    dist_sq = (enemy.pos[0] - unit.pos[0])**2 + (enemy.pos[1] - unit.pos[1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        closest_enemy = enemy
                
                if closest_enemy:
                    self.projectiles.append(Projectile(unit.pos, closest_enemy, unit.damage))
                    unit.cooldown = unit.fire_rate
                    # sfx: unit_fire

        # Update projectiles
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p.pos += p.velocity
            if not (0 <= p.pos[0] < self.WIDTH and 0 <= p.pos[1] < self.HEIGHT):
                projectiles_to_remove.append(i)
                continue

            hit = False
            if p.target in self.enemies:
                dist_sq = (p.pos[0] - p.target.pos[0])**2 + (p.pos[1] - p.target.pos[1])**2
                if dist_sq < p.target.radius ** 2:
                    p.target.health -= p.damage
                    step_reward += 0.1 # Damage reward
                    hit = True
                    # sfx: enemy_hit
            
            if hit:
                projectiles_to_remove.append(i)
                for _ in range(5):
                    self.particles.append(Particle(p.pos, self.COLOR_PROJECTILE, 20, 3, 2))

        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]

        # Update enemies
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            if enemy.health <= 0:
                enemies_to_remove.append(i)
                step_reward += 1 # Kill reward
                self.score += 10 * self.current_wave
                # sfx: enemy_death
                for _ in range(20):
                    self.particles.append(Particle(enemy.pos, self.COLOR_ENEMY, 30, 4, 3))
                continue
            
            direction = np.array(self.TOWER_POS) - enemy.pos
            distance = np.linalg.norm(direction)
            if distance < 25: # Reached tower
                self.tower_health -= 10
                enemies_to_remove.append(i)
                # sfx: tower_damage
                for _ in range(30):
                    self.particles.append(Particle(enemy.pos, self.COLOR_TOWER, 40, 5, 4))
                continue

            enemy.pos += (direction / distance) * enemy.speed
        
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        # Update particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p.pos += p.velocity
            p.life -= 1
            if p.life <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

        return step_reward

    def _spawn_wave(self):
        num_enemies = 2 + self.current_wave
        for _ in range(num_enemies):
            side = random.randint(0, 3)
            if side == 0: # Top
                pos = (random.uniform(0, self.WIDTH), -10)
            elif side == 1: # Right
                pos = (self.WIDTH + 10, random.uniform(0, self.HEIGHT))
            elif side == 2: # Bottom
                pos = (random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            else: # Left
                pos = (-10, random.uniform(0, self.HEIGHT))
            self.enemies.append(Enemy(pos, self.current_wave))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Tower
        pygame.gfxdraw.filled_circle(self.screen, self.TOWER_POS[0], self.TOWER_POS[1], 25, self.COLOR_TOWER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, self.TOWER_POS[0], self.TOWER_POS[1], 20, self.COLOR_TOWER)
        pygame.gfxdraw.aacircle(self.screen, self.TOWER_POS[0], self.TOWER_POS[1], 20, self.COLOR_TOWER)

        # Unit ranges
        for unit in self.units:
            pygame.gfxdraw.aacircle(self.screen, int(unit.pos[0]), int(unit.pos[1]), int(unit.range), (50, 80, 100))

        # Units
        for unit in self.units:
            color = self.COLOR_UNIT_SHOOTER if unit.type == "Shooter" else self.COLOR_UNIT_CANNON
            pos_int = (int(unit.pos[0]), int(unit.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 8, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 8, color)
        
        # Projectiles
        for p in self.projectiles:
            pos_int = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, pos_int, (pos_int[0] - int(p.velocity[0]), pos_int[1] - int(p.velocity[1])), 2)

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy.pos[0]), int(enemy.pos[1]))
            pygame.gfxdraw.filled_trigon(self.screen, pos_int[0], pos_int[1] - 9, pos_int[0] - 8, pos_int[1] + 5, pos_int[0] + 8, pos_int[1] + 5, self.COLOR_ENEMY)
            pygame.gfxdraw.aatrigon(self.screen, pos_int[0], pos_int[1] - 9, pos_int[0] - 8, pos_int[1] + 5, pos_int[0] + 8, pos_int[1] + 5, self.COLOR_ENEMY)
            # Health bar
            if enemy.health < enemy.max_health:
                health_pct = enemy.health / enemy.max_health
                bar_w = 16
                bar_h = 3
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (pos_int[0] - bar_w/2, pos_int[1] - 20, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (pos_int[0] - bar_w/2, pos_int[1] - 20, bar_w * health_pct, bar_h))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size, p.size), int(p.size * (p.life / p.max_life)))
            self.screen.blit(s, (int(p.pos[0] - p.size), int(p.pos[1] - p.size)))

    def _render_ui(self):
        # Wave info
        wave_text = f"Wave: {self.current_wave}/{self.MAX_WAVES}" if self.is_wave_active else f"Next wave in {self.wave_timer/30:.1f}s"
        wave_surf = self.font_m.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(wave_surf, (10, 10))

        # Tower health
        health_rect = pygame.Rect(self.WIDTH - 160, 10, 150, 20)
        health_pct = max(0, self.tower_health / self.MAX_TOWER_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, health_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (health_rect.x, health_rect.y, health_rect.w * health_pct, health_rect.h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_rect, 1)

        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_m.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 20))
        self.screen.blit(score_surf, score_rect)

        # Action selection
        action_text = self.action_options[self.selection_index]
        if self.is_confirming_action:
            prompt = f"Confirm: {action_text}?"
        else:
            prompt = f"Next: {action_text}"
        
        prompt_surf = self.font_m.render(prompt, True, self.COLOR_TEXT)
        prompt_rect = prompt_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 50))
        self.screen.blit(prompt_surf, prompt_rect)

        # Game Over / Win
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            msg = "YOU SURVIVED" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            msg_surf = self.font_l.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "tower_health": self.tower_health,
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert self.tower_health == self.MAX_TOWER_HEALTH # Testable assertion
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        # Test wave spawn assertion
        initial_enemy_count = len(self.enemies)
        self.wave_timer = 1
        self.is_wave_active = False
        self.step(self.action_space.sample())
        assert len(self.enemies) > initial_enemy_count, "Enemy count should increase on new wave"
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Main game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Action from keyboard for manual play
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # no-op
        if keys[pygame.K_SPACE]:
            action[1] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    pygame.quit()