import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Tower:
    def __init__(self, pos, fire_rate=1.0, tower_range=150, damage=5):
        self.pos = pos
        self.fire_rate = fire_rate
        self.range = tower_range
        self.damage = damage
        self.cooldown = 0
        self.target = None

    def update(self, dt, enemies, projectiles):
        self.cooldown = max(0, self.cooldown - dt)
        if self.cooldown == 0:
            self.find_target(enemies)
            if self.target:
                projectiles.append(Projectile(self.pos, self.target, self.damage))
                self.cooldown = 1.0 / self.fire_rate
                # Sound placeholder: # sfx_tower_shoot.wav

    def find_target(self, enemies):
        self.target = None
        closest_dist = self.range
        for enemy in enemies:
            if not enemy.is_alive:
                continue
            dist = math.hypot(self.pos[0] - enemy.pos[0], self.pos[1] - enemy.pos[1])
            if dist < closest_dist:
                closest_dist = dist
                self.target = enemy

    def draw(self, screen):
        # Draw tower base
        pygame.gfxdraw.box(screen, (int(self.pos[0] - 12), int(self.pos[1] - 12), 24, 24), (50, 100, 255, 150))
        pygame.gfxdraw.rectangle(screen, (int(self.pos[0] - 12), int(self.pos[1] - 12), 24, 24), (150, 200, 255))
        # Draw tower turret
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), 8, (150, 200, 255))
        pygame.gfxdraw.aacircle(screen, int(self.pos[0]), int(self.pos[1]), 8, (150, 200, 255))


class Enemy:
    def __init__(self, path, health, speed, size, value):
        self.path = path
        self.path_index = 0
        self.pos = list(self.path[self.path_index])
        self.health = health
        self.max_health = health
        self.speed = speed
        self.size = size
        self.value = value # Score value when killed
        self.is_alive = True
        self.hit_flash_timer = 0

    def update(self, dt):
        self.hit_flash_timer = max(0, self.hit_flash_timer - dt)
        if not self.is_alive:
            return

        if self.path_index >= len(self.path):
            self.is_alive = False
            return

        target_pos = self.path[self.path_index]
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed * dt:
            self.pos = list(target_pos)
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.is_alive = False # Reached the end
        else:
            self.pos[0] += (dx / dist) * self.speed * dt
            self.pos[1] += (dy / dist) * self.speed * dt

    def take_damage(self, amount):
        self.health -= amount
        self.hit_flash_timer = 0.1
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
        return self.is_alive

    def draw(self, screen):
        # Health bar
        if self.health < self.max_health:
            bar_width = self.size * 2
            bar_height = 5
            health_pct = self.health / self.max_health
            pygame.draw.rect(screen, (100, 0, 0), (int(self.pos[0] - bar_width/2), int(self.pos[1] - self.size - 10), bar_width, bar_height))
            pygame.draw.rect(screen, (0, 200, 0), (int(self.pos[0] - bar_width/2), int(self.pos[1] - self.size - 10), int(bar_width * health_pct), bar_height))

        # Body
        color = (255, 50, 50)
        if self.hit_flash_timer > 0:
            color = (255, 255, 255)
        
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), self.size, color)
        pygame.gfxdraw.aacircle(screen, int(self.pos[0]), int(self.pos[1]), self.size, color)


class Projectile:
    def __init__(self, start_pos, target, damage, speed=400):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.is_active = True
        
        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.vel = [(dx / dist) * self.speed, (dy / dist) * self.speed]
        else:
            self.vel = [0, self.speed] # Failsafe if spawned on target

    def update(self, dt):
        if not self.is_active:
            return
        
        # If target is dead, projectile fizzles
        if not self.target.is_alive:
            self.is_active = False
            return

        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # Check if out of bounds
        if not (0 < self.pos[0] < 640 and 0 < self.pos[1] < 400):
            self.is_active = False

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 255, 0), (int(self.pos[0]-2), int(self.pos[1]-2), 4, 4))


class Particle:
    def __init__(self, pos, vel, lifetime, color, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.color = color
        self.size = size

    def update(self, dt):
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.lifetime -= dt

    def draw(self, screen):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (self.size, self.size), self.size)
            screen.blit(s, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to place towers on the four grid squares around your central base."
    )

    game_description = (
        "A streamlined tower defense game. Defend your base from waves of enemies by placing towers strategically."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 18)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PATH = (40, 40, 50)
        self.COLOR_GRID = (60, 60, 70, 100)
        self.COLOR_BASE = (0, 200, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Game constants
        self.MAX_BASE_HEALTH = 100
        self.TOTAL_WAVES = 3
        
        # Define enemy path and tower grid
        self._define_path_and_grid()
        
        self.wave_definitions = {
            1: {"count": 20, "health": 10, "speed": 40, "size": 8, "spawn_interval": 1.5, "value": 1},
            2: {"count": 25, "health": 15, "speed": 50, "size": 10, "spawn_interval": 1.2, "value": 2},
            3: {"count": 30, "health": 20, "speed": 60, "size": 12, "spawn_interval": 1.0, "value": 3},
        }

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.current_wave = 0
        self.enemies_to_spawn = 0
        self.spawn_timer = 0
        self.intermission_timer = 0
        self.wave_state = "INTERMISSION"
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        # FIX: Initialize tower_slots with keys from grid_locations to prevent KeyError
        self.tower_slots = {loc: None for loc in self.grid_locations.values()}
        
        # This is a temporary state for validation, will be properly set in reset()
        self.reset()
        self.validate_implementation()

    def _define_path_and_grid(self):
        self.path = [
            (0, 100), (100, 100), (100, 300), (540, 300), (540, 100), (self.WIDTH, 100)
        ]
        self.base_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        grid_offset = 80
        self.grid_locations = {
            1: (self.base_pos[0], self.base_pos[1] - grid_offset), # Up
            2: (self.base_pos[0], self.base_pos[1] + grid_offset), # Down
            3: (self.base_pos[0] - grid_offset, self.base_pos[1]), # Left
            4: (self.base_pos[0] + grid_offset, self.base_pos[1]), # Right
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.MAX_BASE_HEALTH
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.tower_slots = {loc: None for loc in self.grid_locations.values()}
        
        self.intermission_timer = 3.0 # Time before first wave
        self.wave_state = "INTERMISSION"
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return # Game won
        
        wave_def = self.wave_definitions[self.current_wave]
        self.enemies_to_spawn = wave_def["count"]
        self.spawn_timer = 0
        self.wave_state = "ACTIVE"
        # Sound placeholder: # sfx_wave_start.wav

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        dt = self.clock.tick(self.FPS) / 1000.0
        reward = -0.001 # Small penalty for time passing

        # Handle action
        place_action = action[0]
        if place_action in self.grid_locations:
            reward += self._place_tower(place_action)

        # Update game state
        reward += self._update_game_logic(dt)
        
        self.score += reward
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _place_tower(self, location_key):
        pos = self.grid_locations[location_key]
        if self.tower_slots[pos] is None:
            new_tower = Tower(pos)
            self.towers.append(new_tower)
            self.tower_slots[pos] = new_tower
            # Sound placeholder: # sfx_tower_place.wav
            return 0.1 # Small reward for placing a tower
        return -0.1 # Penalty for trying to place on occupied slot

    def _update_game_logic(self, dt):
        reward = 0
        
        # Wave management
        if self.wave_state == "INTERMISSION":
            self.intermission_timer -= dt
            if self.intermission_timer <= 0:
                self._start_next_wave()
        elif self.wave_state == "ACTIVE":
            if self.enemies_to_spawn > 0:
                self.spawn_timer -= dt
                if self.spawn_timer <= 0:
                    self._spawn_enemy()
                    wave_def = self.wave_definitions[self.current_wave]
                    self.spawn_timer = wave_def["spawn_interval"]
            elif len(self.enemies) == 0:
                # Wave cleared
                reward += 5.0
                self.wave_state = "INTERMISSION"
                self.intermission_timer = 5.0 # Time between waves
                if self.current_wave == self.TOTAL_WAVES:
                    reward += 50.0 # Game won
                    self.game_over = True

        # Update entities
        for tower in self.towers:
            tower.update(dt, self.enemies, self.projectiles)
            
        for p in self.projectiles[:]:
            p.update(dt)
            if not p.is_active:
                self.projectiles.remove(p)
                continue
            
            dist = math.hypot(p.pos[0] - p.target.pos[0], p.pos[1] - p.target.pos[1])
            if dist < p.target.size:
                if p.target.is_alive:
                    reward += 0.1 # Reward for hitting
                    if not p.target.take_damage(p.damage): # if target is killed
                        reward += p.target.value
                        self._create_explosion(p.target.pos, 20)
                        # Sound placeholder: # sfx_enemy_explode.wav
                    # Sound placeholder: # sfx_enemy_hit.wav
                self.projectiles.remove(p)

        for enemy in self.enemies[:]:
            enemy.update(dt)
            if not enemy.is_alive:
                if enemy.path_index >= len(enemy.path): # Reached base
                    self.base_health -= enemy.max_health # Damage proportional to enemy health
                    self.base_health = max(0, self.base_health)
                    self._create_explosion(enemy.pos, 10, (255,100,0))
                    # Sound placeholder: # sfx_base_damage.wav
                self.enemies.remove(enemy)

        for particle in self.particles[:]:
            particle.update(dt)
            if particle.lifetime <= 0:
                self.particles.remove(particle)
                
        return reward

    def _spawn_enemy(self):
        wave_def = self.wave_definitions[self.current_wave]
        new_enemy = Enemy(
            self.path,
            wave_def["health"],
            wave_def["speed"],
            wave_def["size"],
            wave_def["value"]
        )
        self.enemies.append(new_enemy)
        self.enemies_to_spawn -= 1
        # Sound placeholder: # sfx_enemy_spawn.wav

    def _create_explosion(self, position, num_particles, color=(255, 150, 0)):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 100)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.uniform(0.3, 0.8)
            size = self.np_random.integers(2, 5)
            self.particles.append(Particle(position, vel, lifetime, color, size))

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 40)
        
        # Draw tower placement grid
        for pos in self.grid_locations.values():
            if self.tower_slots.get(pos) is None:
                pygame.gfxdraw.box(self.screen, (int(pos[0]-15), int(pos[1]-15), 30, 30), self.COLOR_GRID)

        # Draw base
        base_size = 40
        base_rect = pygame.Rect(self.base_pos[0] - base_size/2, self.base_pos[1] - base_size/2, base_size, base_size)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, (200, 255, 220), base_rect, 2)

        # Draw entities
        for tower in self.towers:
            tower.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)
        for proj in self.projectiles:
            proj.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)

    def _render_ui(self):
        # Score
        score_text = self.font_s.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}"
        if self.wave_state == "INTERMISSION" and not self.game_over:
            if self.current_wave < self.TOTAL_WAVES:
                wave_str = f"NEXT WAVE IN {int(self.intermission_timer)+1}"
        wave_text = self.font_s.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Base Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_pct = self.base_health / self.MAX_BASE_HEALTH
        pygame.draw.rect(self.screen, (100, 0, 0), ((self.WIDTH - health_bar_width) / 2, 10, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BASE, ((self.WIDTH - health_bar_width) / 2, 10, int(health_bar_width * health_pct), health_bar_height))
        health_text = self.font_s.render(f"{int(self.base_health)}/{self.MAX_BASE_HEALTH}", True, (0,0,0))
        self.screen.blit(health_text, ((self.WIDTH - health_text.get_width()) / 2, 12))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.base_health <= 0:
                msg = "GAME OVER"
                color = (255, 50, 50)
            else:
                msg = "VICTORY!"
                color = (50, 255, 50)
                
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, ((self.WIDTH - end_text.get_width()) / 2, (self.HEIGHT - end_text.get_height()) / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    # It's not part of the Gymnasium interface but is useful for testing
    import sys
    
    # Re-enable video driver for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy screen for display if running locally
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    total_reward = 0
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait a bit before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()
    sys.exit()