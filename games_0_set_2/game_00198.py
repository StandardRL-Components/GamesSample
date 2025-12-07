
# Generated: 2025-08-27T12:54:22.001653
# Source Brief: brief_00198.md
# Brief Index: 198

        
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


# Helper classes for game objects
class Tower:
    def __init__(self, pos, tower_type, stats):
        self.pos = pygame.Vector2(pos)
        self.type = tower_type
        self.range = stats['range']
        self.damage = stats['damage']
        self.fire_rate = stats['fire_rate']
        self.cooldown = 0
        self.color = stats['color']

class Enemy:
    def __init__(self, health, speed, path):
        self.path = path
        self.pos = pygame.Vector2(path[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.waypoint_index = 1
        self.distance_traveled = 0

    def move(self):
        if self.waypoint_index >= len(self.path):
            return True # Reached the end

        target = pygame.Vector2(self.path[self.waypoint_index])
        direction = (target - self.pos).normalize()
        distance_to_target = self.pos.distance_to(target)

        move_distance = min(self.speed, distance_to_target)
        self.pos += direction * move_distance
        self.distance_traveled += move_distance

        if self.pos.distance_to(target) < 1:
            self.waypoint_index += 1
        
        return False

class Projectile:
    def __init__(self, pos, target, damage, speed, color):
        self.pos = pygame.Vector2(pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.color = color

    def move(self):
        if self.target.health <= 0:
            return True # Target is already gone

        direction = (self.target.pos - self.pos).normalize()
        self.pos += direction * self.speed
        
        if self.pos.distance_to(self.target.pos) < 8: # Hit radius
            return True # Hit
        return False

class Particle:
    def __init__(self, pos, color, lifespan, size, velocity):
        self.pos = pygame.Vector2(pos)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.size = size
        self.velocity = velocity

    def update(self):
        self.pos += self.velocity
        self.lifespan -= 1
        return self.lifespan <= 0


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to select a build site. Space to build a fast, low-damage tower. Shift to build a slow, high-damage tower."
    )

    game_description = (
        "A minimalist tower defense game. Strategically place towers to defend your base against waves of enemies."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 5

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PATH = (40, 40, 55)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_ENEMY = (230, 25, 75)
        self.COLOR_TEXT = (245, 245, 245)
        self.COLOR_HEALTH_BAR = (70, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0)
        self.TOWER_STATS = {
            1: {'range': 80, 'damage': 1.5, 'fire_rate': 8, 'color': (66, 135, 245)}, # Blue
            2: {'range': 100, 'damage': 8, 'fire_rate': 30, 'color': (255, 225, 25)} # Yellow
        }
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Game world setup
        self.path = [
            (50, -20), (50, 120), (250, 120), (250, 280),
            (550, 280), (550, 100), (400, 100), (400, 420)
        ]
        self.base_pos = (400, 380)
        self.tower_zones = [
            (150, 120), (150, 200), (350, 200),
            (475, 180), (475, 350), (300, 350)
        ]
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False

        self.base_health = 100
        self.current_wave = 0
        
        self.towers = {} # map zone_idx to Tower object
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.selected_zone_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self.wave_spawning = False
        self.wave_spawn_timer = 0
        self.enemies_to_spawn = []
        self.wave_cleared_bonus_given = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        placed_tower_this_step = False

        if not self.game_over:
            # 1. Handle player input
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            placed_tower_this_step = self._handle_input(movement, space_held, shift_held)

            # 2. Update game logic
            self._update_waves()
            step_rewards = self._update_towers_and_projectiles()
            reward += step_rewards
            
            enemy_reward, game_over_penalty = self._update_enemies()
            reward += enemy_reward
            if game_over_penalty:
                reward += game_over_penalty
                self.game_over = True

            self._update_particles()
        
        # 3. Calculate final reward & termination
        if not placed_tower_this_step and not self.game_over:
            reward -= 0.01

        self.steps += 1
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over: # Game ended by time or victory
            if self.victory:
                reward += 100 # Win bonus
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Selector Movement ---
        if movement == 1: # Up
            self.selected_zone_idx = (self.selected_zone_idx - 3) % len(self.tower_zones)
        elif movement == 2: # Down
            self.selected_zone_idx = (self.selected_zone_idx + 3) % len(self.tower_zones)
        elif movement == 3: # Left
            row_start = (self.selected_zone_idx // 3) * 3
            offset = (self.selected_zone_idx - row_start - 1 + 3) % 3
            self.selected_zone_idx = row_start + offset
        elif movement == 4: # Right
            row_start = (self.selected_zone_idx // 3) * 3
            offset = (self.selected_zone_idx - row_start + 1) % 3
            self.selected_zone_idx = row_start + offset

        # --- Tower Placement ---
        placed_tower = False
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if self.selected_zone_idx not in self.towers:
            if space_pressed:
                # Place Tower Type 1
                pos = self.tower_zones[self.selected_zone_idx]
                self.towers[self.selected_zone_idx] = Tower(pos, 1, self.TOWER_STATS[1])
                placed_tower = True
                # SFX: Place Tower 1
            elif shift_pressed:
                # Place Tower Type 2
                pos = self.tower_zones[self.selected_zone_idx]
                self.towers[self.selected_zone_idx] = Tower(pos, 2, self.TOWER_STATS[2])
                placed_tower = True
                # SFX: Place Tower 2

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return placed_tower

    def _start_next_wave(self):
        if self.current_wave >= self.MAX_WAVES:
            self.victory = True
            return 0 # Game won

        # Wave completion reward
        wave_reward = 0
        if self.current_wave > 0 and not self.wave_cleared_bonus_given:
             wave_reward = 5.0
             self.wave_cleared_bonus_given = True

        self.current_wave += 1
        self.wave_spawning = True
        self.wave_spawn_timer = 0
        self.wave_cleared_bonus_given = False

        num_enemies = 3 + (self.current_wave - 1) * 2
        health = 10 + (self.current_wave - 1) * 10
        speed = 0.8 * (1 + (self.current_wave - 1) * 0.05)
        
        self.enemies_to_spawn = [Enemy(health, speed, self.path) for _ in range(num_enemies)]
        return wave_reward

    def _update_waves(self):
        if self.wave_spawning:
            self.wave_spawn_timer += 1
            if self.wave_spawn_timer > 45 and self.enemies_to_spawn:
                self.enemies.append(self.enemies_to_spawn.pop(0))
                self.wave_spawn_timer = 0
                # SFX: Enemy Spawn
            
            if not self.enemies_to_spawn:
                self.wave_spawning = False
    
    def _update_towers_and_projectiles(self):
        reward = 0
        # Update Towers
        for tower in self.towers.values():
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue
            
            # Find a target: enemy furthest along the path within range
            target = None
            max_dist = -1
            for enemy in self.enemies:
                if tower.pos.distance_to(enemy.pos) <= tower.range:
                    if enemy.distance_traveled > max_dist:
                        max_dist = enemy.distance_traveled
                        target = enemy
            
            if target:
                self.projectiles.append(Projectile(tower.pos, target, tower.damage, 8, tower.color))
                tower.cooldown = tower.fire_rate
                # SFX: Tower Fire

        # Update Projectiles
        dead_projectiles = []
        for proj in self.projectiles:
            if proj.move(): # Hit or target gone
                if proj.target.health > 0 and proj.pos.distance_to(proj.target.pos) < 8:
                    proj.target.health -= proj.damage
                    reward += 0.1 # Hit reward
                    # SFX: Enemy Hit
                    for _ in range(5): # Hit particles
                        angle = random.uniform(0, 2 * math.pi)
                        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(0.5, 2)
                        self.particles.append(Particle(proj.pos, proj.color, 15, random.randint(1, 3), vel))

                dead_projectiles.append(proj)
        self.projectiles = [p for p in self.projectiles if p not in dead_projectiles]
        return reward

    def _update_enemies(self):
        reward = 0
        game_over_penalty = 0
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                reward += 1.0 # Defeat reward
                # SFX: Enemy Death
                for _ in range(20): # Death explosion
                    angle = random.uniform(0, 2 * math.pi)
                    vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, 4)
                    self.particles.append(Particle(enemy.pos, self.COLOR_ENEMY, 30, random.randint(2, 4), vel))
                continue

            if enemy.move(): # Reached base
                self.base_health -= 10
                # SFX: Base Damage
                if self.base_health <= 0:
                    self.base_health = 0
                    game_over_penalty = -100 # Loss penalty
                continue
            
            surviving_enemies.append(enemy)
        self.enemies = surviving_enemies
        
        # Check for wave clear
        if not self.wave_spawning and not self.enemies:
            reward += self._start_next_wave()

        return reward, game_over_penalty

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _check_termination(self):
        return (
            self.base_health <= 0 or
            self.steps >= self.MAX_STEPS or
            self.victory
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path, 30)

        # Draw tower zones and selector
        for i, pos in enumerate(self.tower_zones):
            if i == self.selected_zone_idx:
                # Pulsing glow for selector
                alpha = 100 + 50 * math.sin(self.steps * 0.2)
                glow_color = (150, 150, 200, alpha)
                s = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, 30, 30, 30, glow_color)
                self.screen.blit(s, (pos[0] - 30, pos[1] - 30))
            
            if i not in self.towers:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 20, (100, 100, 120))

        # Draw base
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0]-15, self.base_pos[1]-15, 30, 30))
        pygame.gfxdraw.box(self.screen, (self.base_pos[0]-15, self.base_pos[1]-15, 30, 30), self.COLOR_BASE)

        # Draw towers
        for tower in self.towers.values():
            p1 = (tower.pos.x, tower.pos.y - 12)
            p2 = (tower.pos.x - 10, tower.pos.y + 6)
            p3 = (tower.pos.x + 10, tower.pos.y + 6)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], tower.color)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], tower.color)

        # Draw enemies
        for enemy in self.enemies:
            x, y = int(enemy.pos.x), int(enemy.pos.y)
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_ENEMY)
            # Health bar
            health_ratio = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (0,255,0), (x - 8, y - 12, 16 * health_ratio, 3))

        # Draw projectiles
        for proj in self.projectiles:
            x, y = int(proj.pos.x), int(proj.pos.y)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, proj.color)
            pygame.gfxdraw.aacircle(self.screen, x, y, 3, proj.color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.max_lifespan))
            color = (*p.color, alpha)
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.size, p.size), p.size)
            self.screen.blit(s, (p.pos.x - p.size, p.pos.y - p.size))

    def _render_ui(self):
        # Wave number
        wave_text = self.font_large.render(f"WAVE {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(score_text, score_rect)

        # Base Health
        health_text = self.font_small.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH - 120, 10))
        health_ratio = self.base_health / 100
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.WIDTH - 122, 30, 104, 14))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (self.WIDTH - 120, 32, 100 * health_ratio, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.victory else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn)
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
        assert self.base_health == 100
        assert self.current_wave == 1
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a demonstration of the environment and not part of the required solution
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keyboard to MultiDiscrete action space
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    pygame.quit()