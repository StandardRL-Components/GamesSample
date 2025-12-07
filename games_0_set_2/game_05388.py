
# Generated: 2025-08-28T04:51:39.563045
# Source Brief: brief_05388.md
# Brief Index: 5388

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim the turret. Press space to fire your weapon. "
        "Defend the city from descending alien ships."
    )

    game_description = (
        "Defend your city from a relentless alien invasion. Aim your turret and "
        "shoot down enemy ships before they reach the city skyline and cause damage."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    CITY_HEIGHT = 50
    MAX_CITY_HEALTH = 100

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_STAR = (100, 100, 120)
    COLOR_CITY = (50, 50, 60)
    COLOR_CITY_DAMAGED = (90, 60, 60)
    COLOR_TURRET_BASE = (120, 120, 130)
    COLOR_TURRET_BARREL = (200, 200, 210)
    COLOR_PROJECTILE = (100, 255, 100)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_GREEN = (80, 220, 80)
    COLOR_HEALTH_YELLOW = (220, 220, 80)
    COLOR_HEALTH_RED = (220, 80, 80)

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

        self.np_random = None # Will be initialized in reset
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turret_angle = 0.0
        self.fire_cooldown = 0
        self.city_health = 0
        self.city_damage_flash = 0
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.enemy_spawn_rate = 0.0
        self.enemy_speed_multiplier = 0.0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        # Turret
        self.turret_angle = -math.pi / 2
        self.fire_cooldown = 0
        
        # City
        self.city_health = self.MAX_CITY_HEALTH
        self.city_damage_flash = 0
        
        # Entities
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Difficulty
        self.enemy_spawn_rate = 0.03
        self.enemy_speed_multiplier = 1.0
        
        # Background
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT - self.CITY_HEIGHT), self.np_random.integers(1, 4))
            for _ in range(100)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            
            reward += 0.01  # Small survival reward per step
            
            step_reward, new_particles = self._handle_collisions()
            reward += step_reward
            self.particles.extend(new_particles)

            self._update_difficulty()
            
            self.steps += 1
            terminated = self.city_health <= 0 or self.steps >= self.MAX_STEPS
            if terminated:
                self.game_over = True
                if self.city_health > 0 and self.steps >= self.MAX_STEPS:
                    reward += 100  # Win bonus
                    self.win_condition = True
                else:
                    reward -= 100  # Loss penalty
        
        self.clock.tick(30) # Maintain 30 FPS for smooth visuals
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Aim turret
        if movement == 3:  # Left
            self.turret_angle -= 0.08
        elif movement == 4:  # Right
            self.turret_angle += 0.08
        
        # Clamp angle to prevent aiming downwards
        self.turret_angle = np.clip(self.turret_angle, -math.pi + 0.1, -0.1)

        # Fire projectile
        if space_held and self.fire_cooldown <= 0:
            self.fire_cooldown = 8 # 8 frames cooldown
            barrel_length = 30
            turret_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - self.CITY_HEIGHT)
            
            proj_start_x = turret_pos[0] + math.cos(self.turret_angle) * barrel_length
            proj_start_y = turret_pos[1] + math.sin(self.turret_angle) * barrel_length
            
            proj_speed = 15
            proj_vel_x = math.cos(self.turret_angle) * proj_speed
            proj_vel_y = math.sin(self.turret_angle) * proj_speed
            
            self.projectiles.append({
                "pos": pygame.Vector2(proj_start_x, proj_start_y),
                "vel": pygame.Vector2(proj_vel_x, proj_vel_y),
                "rect": pygame.Rect(proj_start_x-2, proj_start_y-4, 4, 8)
            })
            # sfx: fire_laser

    def _update_game_state(self):
        # Cooldowns
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.city_damage_flash > 0:
            self.city_damage_flash -= 1

        # Update projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["rect"].center = p["pos"]
            if not self.screen.get_rect().colliderect(p["rect"]):
                self.projectiles.remove(p)
        
        # Spawn enemies
        if self.np_random.random() < self.enemy_spawn_rate:
            self._spawn_enemy()
            
        # Update enemies
        for e in self.enemies[:]:
            # Movement patterns
            if e["type"] == "linear":
                e["pos"].y += e["speed"]
            elif e["type"] == "sine":
                e["pos"].y += e["speed"]
                e["pos"].x = e["origin_x"] + math.sin(e["pos"].y * 0.02) * 50
            elif e["type"] == "diagonal":
                e["pos"].x += e["diag_dir"] * e["speed"] * 0.5
                e["pos"].y += e["speed"]

            e["rect"].center = e["pos"]

            # Check for city impact
            if e["pos"].y > self.SCREEN_HEIGHT - self.CITY_HEIGHT:
                self.city_health = max(0, self.city_health - 10)
                self.city_damage_flash = 15 # Flash for 15 frames
                self.enemies.remove(e)
                self.score = max(0, self.score - 5)
                # sfx: city_hit
                self._create_explosion(e["pos"], 15, (255, 150, 50))
            elif e["pos"].y > self.SCREEN_HEIGHT + 20: # Off-screen cleanup
                 self.enemies.remove(e)

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["radius"] -= 0.5
            if p["radius"] <= 0:
                self.particles.remove(p)

    def _spawn_enemy(self):
        enemy_type = self.np_random.choice(["linear", "sine", "diagonal"])
        speed = self.np_random.uniform(1.0, 2.5) * self.enemy_speed_multiplier
        speed = min(speed, 10.0) # Cap max speed
        
        x_pos = self.np_random.integers(20, self.SCREEN_WIDTH - 20)
        
        enemy = {
            "pos": pygame.Vector2(x_pos, -20),
            "speed": speed,
            "type": enemy_type,
            "rect": pygame.Rect(x_pos - 10, -20, 20, 20),
            "origin_x": x_pos,
            "diag_dir": self.np_random.choice([-1, 1])
        }
        self.enemies.append(enemy)

    def _handle_collisions(self):
        reward = 0
        new_particles = []
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if p["rect"].colliderect(e["rect"]):
                    if p in self.projectiles: self.projectiles.remove(p)
                    if e in self.enemies: self.enemies.remove(e)
                    
                    self.score += 10
                    reward += 1
                    new_particles.extend(self._create_explosion(e["pos"], 20, (255, 180, 50)))
                    # sfx: explosion
                    break # Projectile is gone, move to next one
        return reward, new_particles

    def _create_explosion(self, pos, num_particles, color):
        particles = []
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "radius": self.np_random.integers(6, 12),
                "color": color
            })
        return particles

    def _update_difficulty(self):
        # Increase spawn rate by 0.001 every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_spawn_rate += 0.0003
            self.enemy_spawn_rate = min(self.enemy_spawn_rate, 0.2) # Cap spawn rate
        
        # Increase enemy speed every 500 steps
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_speed_multiplier += 0.1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))
            
        # City
        city_color = self.COLOR_CITY_DAMAGED if self.city_damage_flash > 0 and self.steps % 4 < 2 else self.COLOR_CITY
        pygame.draw.rect(self.screen, city_color, (0, self.SCREEN_HEIGHT - self.CITY_HEIGHT, self.SCREEN_WIDTH, self.CITY_HEIGHT))
        
        # Turret
        turret_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - self.CITY_HEIGHT)
        pygame.gfxdraw.filled_circle(self.screen, turret_pos[0], turret_pos[1], 15, self.COLOR_TURRET_BASE)
        pygame.gfxdraw.aacircle(self.screen, turret_pos[0], turret_pos[1], 15, self.COLOR_TURRET_BASE)
        
        barrel_end_x = turret_pos[0] + math.cos(self.turret_angle) * 30
        barrel_end_y = turret_pos[1] + math.sin(self.turret_angle) * 30
        pygame.draw.line(self.screen, self.COLOR_TURRET_BARREL, turret_pos, (barrel_end_x, barrel_end_y), 6)
        
        # Projectiles
        for p in self.projectiles:
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, p["pos"] - p["vel"] * 0.5, p["pos"] + p["vel"] * 0.5, 3)
            
        # Enemies
        for e in self.enemies:
            pygame.gfxdraw.filled_polygon(self.screen, [
                (e["rect"].centerx, e["rect"].top),
                (e["rect"].right, e["rect"].bottom),
                (e["rect"].left, e["rect"].bottom)
            ], self.COLOR_ENEMY)
            
        # Particles
        for p in self.particles:
            color = (
                max(0, p["color"][0] * (p["radius"] / 10)),
                max(0, p["color"][1] * (p["radius"] / 10)),
                max(0, p["color"][2] * (p["radius"] / 10)),
            )
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Health Bar
        health_pct = self.city_health / self.MAX_CITY_HEALTH
        bar_color = self.COLOR_HEALTH_RED
        if health_pct > 0.6:
            bar_color = self.COLOR_HEALTH_GREEN
        elif health_pct > 0.3:
            bar_color = self.COLOR_HEALTH_YELLOW
            
        bar_width = 200
        bar_height = 20
        health_width = int(bar_width * health_pct)
        
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_width, bar_height))
        if health_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 10, health_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, bar_width, bar_height), 2)
        
        # Game Over / Win Message
        if self.game_over:
            if self.win_condition:
                msg = "CITY SAVED"
                color = self.COLOR_HEALTH_GREEN
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "city_health": self.city_health,
            "enemies_on_screen": len(self.enemies),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        assert self.score == 0
        assert self.city_health == self.MAX_CITY_HEALTH
        assert self.city_health <= 100
        assert self.score >= 0

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("City Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("CITY DEFENDER")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # Action defaults
        movement = 0
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

    env.close()