
# Generated: 2025-08-28T06:17:41.826017
# Source Brief: brief_05848.md
# Brief Index: 5848

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game entities to keep the main environment class clean
class Tower:
    """Represents a single defensive tower."""
    def __init__(self, pos, tower_type):
        self.pos = pos
        self.type = tower_type
        self.cooldown = 0
        if tower_type == 'basic':
            self.range = 100
            self.damage = 10
            self.fire_rate = 20  # frames
            self.color = (50, 150, 255)
            self.projectile_speed = 8
        elif tower_type == 'heavy':
            self.range = 120
            self.damage = 35
            self.fire_rate = 60
            self.color = (255, 100, 0)
            self.projectile_speed = 6
        elif tower_type == 'fast':
            self.range = 80
            self.damage = 5
            self.fire_rate = 8
            self.color = (200, 50, 255)
            self.projectile_speed = 12

    def can_fire(self):
        return self.cooldown <= 0

    def fire(self):
        self.cooldown = self.fire_rate
        # sfx: tower_fire.wav

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def draw(self, surface):
        size = 12 if self.type != 'heavy' else 15
        pygame.draw.circle(surface, self.color, self.pos, size)
        pygame.draw.circle(surface, (255, 255, 255), self.pos, size, 2)
        # Firing animation with a subtle glow
        if self.fire_rate - self.cooldown < 3:
            flash_color = (255, 255, 255)
            pygame.gfxdraw.filled_circle(surface, self.pos[0], self.pos[1], size + 3, (*flash_color, 100))

class Enemy:
    """Represents a single enemy unit."""
    def __init__(self, path, speed, health=20):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.health = health
        self.max_health = health
        self.speed = speed
        self.size = 8
        self.is_alive = True

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True # Reached end of the path

        target = np.array(self.path[self.path_index + 1])
        direction = target - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        return False

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
            # sfx: enemy_destroyed.wav
        # sfx: enemy_hit.wav

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        rect = pygame.Rect(x - self.size, y - self.size, self.size * 2, self.size * 2)
        pygame.draw.rect(surface, (255, 50, 50), rect)
        pygame.draw.rect(surface, (255, 150, 150), rect, 2)

        # Draw health bar above the enemy
        health_pct = self.health / self.max_health
        bar_width = self.size * 2
        health_bar_rect = pygame.Rect(x - self.size, y - self.size - 8, bar_width, 4)
        health_fill_rect = pygame.Rect(x - self.size, y - self.size - 8, int(bar_width * health_pct), 4)
        pygame.draw.rect(surface, (50, 50, 50), health_bar_rect)
        pygame.draw.rect(surface, (50, 255, 50), health_fill_rect)

class Projectile:
    """Represents a projectile fired from a tower."""
    def __init__(self, start_pos, target_enemy, damage, speed, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.damage = damage
        self.speed = speed
        self.color = color
        self.active = True

    def move_and_check_hit(self):
        if not self.target.is_alive:
            self.active = False
            return False

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.target.size: # Hit detection
            self.target.take_damage(self.damage)
            self.active = False
            return True
        else:
            self.pos += (direction / distance) * self.speed
            return False

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 3, self.color)

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, vel, lifespan, color, size):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        alpha = max(0, min(255, int(255 * (self.lifespan / self.max_lifespan))))
        color_with_alpha = (*self.color, alpha)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.size), color_with_alpha)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to place basic towers in 4 slots. "
        "Hold space to place a heavy tower, and hold shift to place a fast-firing tower."
    )

    game_description = (
        "A minimalist tower defense game. Defend your base from 5 waves of enemies "
        "by strategically placing different types of towers on the grid."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    COLOR_BG = (10, 20, 30)
    COLOR_PATH = (30, 40, 50)
    COLOR_BASE = (50, 200, 50)
    COLOR_SLOT = (40, 60, 80)
    MAX_STEPS = 1800
    TOTAL_WAVES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.base_pos = (60, self.SCREEN_HEIGHT // 2)
        self.enemy_path = self._define_path()
        self.tower_slots = self._define_tower_slots()
        
        # Initialize state variables that will be reset
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def _define_path(self):
        # A winding path for enemies to follow
        return [
            (self.SCREEN_WIDTH + 20, self.SCREEN_HEIGHT // 2),
            (550, self.SCREEN_HEIGHT // 2),
            (500, 80),
            (300, 80),
            (250, self.SCREEN_HEIGHT // 2),
            (300, self.SCREEN_HEIGHT - 80),
            (500, self.SCREEN_HEIGHT - 80),
            (550, self.SCREEN_HEIGHT // 2),
            (self.base_pos[0] + 20, self.base_pos[1])
        ]

    def _define_tower_slots(self):
        # Pre-defined locations for tower placement
        cx, cy = 280, self.SCREEN_HEIGHT // 2
        return {
            'up': {'pos': (cx, cy - 100), 'occupied': False, 'type': 'basic'},
            'down': {'pos': (cx, cy + 100), 'occupied': False, 'type': 'basic'},
            'left': {'pos': (cx - 100, cy - 50), 'occupied': False, 'type': 'basic'},
            'right': {'pos': (cx - 100, cy + 50), 'occupied': False, 'type': 'basic'},
            'heavy': {'pos': (cx + 100, cy), 'occupied': False, 'type': 'heavy'},
            'fast': {'pos': (160, cy), 'occupied': False, 'type': 'fast'},
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.max_base_health = 100
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.next_wave_timer = 90 # 3 seconds at 30fps
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        for slot in self.tower_slots.values():
            slot['occupied'] = False
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        self.step_reward = -0.01

        if not self.game_over:
            self._handle_action(action)
            self._update_wave_state()
            self._update_towers()
            self._update_projectiles()
            self._update_enemies()
        
        self._update_particles()
        
        self.steps += 1
        reward = self.step_reward
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        slot_key = None
        # Prioritize special towers over basic ones
        if shift_held: slot_key = 'fast'
        elif space_held: slot_key = 'heavy'
        elif movement == 1: slot_key = 'up'
        elif movement == 2: slot_key = 'down'
        elif movement == 3: slot_key = 'left'
        elif movement == 4: slot_key = 'right'
        
        if slot_key and not self.tower_slots[slot_key]['occupied']:
            slot = self.tower_slots[slot_key]
            self.towers.append(Tower(slot['pos'], slot['type']))
            slot['occupied'] = True
            # sfx: place_tower.wav

    def _update_wave_state(self):
        if not self.wave_in_progress and not self.enemies:
            if self.wave_number >= self.TOTAL_WAVES:
                self.win = True
                self.game_over = True
                return

            self.next_wave_timer -= 1
            if self.next_wave_timer <= 0:
                if self.wave_number > 0:
                    self.score += 5
                    self.step_reward += 5
                self.wave_number += 1
                self._spawn_wave()
                self.wave_in_progress = True
    
    def _spawn_wave(self):
        num_enemies = 3 + (self.wave_number - 1) * 2
        enemy_speed = 1.0 + (self.wave_number - 1) * 0.1
        enemy_health = 20 + (self.wave_number - 1) * 10
        
        for i in range(num_enemies):
            # Stagger spawn by giving each enemy a slightly delayed path start
            offset_path = [(p[0] + i * 25, p[1]) for p in self.enemy_path]
            self.enemies.append(Enemy(offset_path, enemy_speed, enemy_health))

    def _update_towers(self):
        for tower in self.towers:
            tower.update()
            if tower.can_fire():
                # Find the closest enemy in range
                target = None
                min_dist = tower.range
                for enemy in self.enemies:
                    dist = np.linalg.norm(np.array(tower.pos) - enemy.pos)
                    if dist < min_dist:
                        min_dist = dist
                        target = enemy
                
                if target:
                    tower.fire()
                    self.projectiles.append(Projectile(tower.pos, target, tower.damage, tower.projectile_speed, tower.color))

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if not p.active:
                self.projectiles.remove(p)
                continue
            
            hit = p.move_and_check_hit()
            if hit:
                self.step_reward += 0.1
                self._create_explosion(p.target.pos, 5, (255, 255, 100))
                if not p.target.is_alive:
                    self.step_reward += 1
                    self.score += 1
                    self._create_explosion(p.target.pos, 15, (255, 100, 50))
                    self.enemies.remove(p.target)
            
            # Remove projectiles that go off-screen
            if not (0 < p.pos[0] < self.SCREEN_WIDTH and 0 < p.pos[1] < self.SCREEN_HEIGHT):
                p.active = False

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            reached_base = enemy.move()
            if reached_base:
                self.base_health -= 10
                self.base_health = max(0, self.base_health)
                self.enemies.remove(enemy)
                self._create_explosion(self.base_pos, 20, (255, 0, 0))
                # sfx: base_damage.wav
        
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            self.next_wave_timer = 150 # 5 seconds between waves

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 25)
            size = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, lifespan, color, size))

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        if len(self.enemy_path) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.enemy_path, 30)

        for slot in self.tower_slots.values():
            if not slot['occupied']:
                pygame.gfxdraw.filled_circle(self.screen, slot['pos'][0], slot['pos'][1], 15, (*self.COLOR_SLOT, 100))
                pygame.gfxdraw.aacircle(self.screen, slot['pos'][0], slot['pos'][1], 15, self.COLOR_SLOT)

        pygame.draw.circle(self.screen, self.COLOR_BASE, self.base_pos, 20)
        pygame.draw.circle(self.screen, (255, 255, 255), self.base_pos, 20, 3)

        for tower in self.towers: tower.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        
    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, (255, 255, 255))
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        health_pct = self.base_health / self.max_base_health
        bar_width = 150
        health_bar_rect = pygame.Rect(10, 35, bar_width, 15)
        health_fill_rect = pygame.Rect(10, 35, int(bar_width * health_pct), 15)
        pygame.draw.rect(self.screen, (50, 0, 0), health_bar_rect)
        pygame.draw.rect(self.screen, (0, 200, 0), health_fill_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), health_bar_rect, 1)

        if not self.wave_in_progress and not self.game_over and self.wave_number < self.TOTAL_WAVES:
            secs_left = math.ceil(self.next_wave_timer / 30)
            wave_announce = self.font_large.render(f"WAVE {self.wave_number + 1} IN {secs_left}", True, (255, 255, 255))
            text_rect = wave_announce.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(wave_announce, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block is for human playtesting and visualization.
    # It demonstrates how to run the environment with keyboard controls.
    # Set the SDL_VIDEODRIVER for your OS to see the window.
    import os
    # For Windows: os.environ['SDL_VIDEODRIVER'] = 'windows'
    # For Linux/Mac: os.environ['SDL_VIDEODRIVER'] = 'x11' or 'cocoa'
    # For headless: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    try:
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        env = GameEnv(render_mode="rgb_array")
    except pygame.error:
        print("Display driver not found, running headlessly. No window will be shown.")
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")

    action = np.array([0, 0, 0]) # [movement, space, shift]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        keys = pygame.key.get_pressed()
        
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()