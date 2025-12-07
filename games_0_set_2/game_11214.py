import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:47:40.862836
# Source Brief: brief_01214.md
# Brief Index: 1214
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, size, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        alpha = int(255 * (self.life / self.lifespan))
        if alpha > 0:
            color_with_alpha = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.size), color_with_alpha)

class Enemy:
    def __init__(self, pos, speed, health, radius):
        self.pos = np.array(pos, dtype=float)
        self.speed = speed
        self.health = health
        self.max_health = health
        self.radius = radius
        self.color = (255, 50, 50)
        self.slow_timer = 0
        self.damage_timer = 0
        self.pulse_offset = random.uniform(0, 2 * math.pi)

    def update(self, base_pos):
        if self.slow_timer > 0:
            self.slow_timer -= 1
            current_speed = self.speed * 0.4
        else:
            current_speed = self.speed

        if self.damage_timer > 0:
            self.damage_timer -= 1
            self.health -= 0.1 # Damage over time

        direction = base_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 1:
            direction /= dist
        self.pos += direction * current_speed

        return self.health > 0

    def draw(self, surface, steps):
        # Pulsing effect for "breathing"
        pulse = math.sin(steps * 0.1 + self.pulse_offset) * 2
        radius = int(self.radius + pulse)
        
        # Glow effect
        glow_radius = int(radius * 1.5)
        glow_alpha = 60
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), glow_radius, self.color + (glow_alpha,))
        
        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius, self.color)

class Shrine:
    SHRINE_TYPES = {
        "repel": {"color": (0, 150, 255), "shape": "triangle", "max_size": 15},
        "slow": {"color": (255, 200, 0), "shape": "square", "max_size": 12},
        "damage": {"color": (255, 100, 0), "shape": "hexagon", "max_size": 10},
    }

    def __init__(self, pos, type):
        self.pos = np.array(pos, dtype=float)
        self.type = type
        self.stats = self.SHRINE_TYPES[type]
        self.color = self.stats["color"]
        self.max_size = self.stats["max_size"]
        self.size = self.max_size
        self.lifespan = 30 * 15 # 15 seconds at 30 FPS
        self.life = self.lifespan
        self.is_activating = False
        self.activation_radius = 0
        self.max_activation_radius = 120
        self.activation_speed = 8

    def update(self):
        self.life -= 1
        self.size = self.max_size * (self.life / self.lifespan)
        
        if self.is_activating:
            self.activation_radius += self.activation_speed
            if self.activation_radius >= self.max_activation_radius:
                self.is_activating = False
                self.activation_radius = 0

        return self.life > 0 and self.size > 1

    def draw(self, surface):
        # Glow
        glow_radius = int(self.size * 2.5)
        glow_alpha = int(100 * (self.life / self.lifespan))
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), glow_radius, self.color + (glow_alpha,))

        # Shape
        if self.stats["shape"] == "triangle":
            p1 = (self.pos[0], self.pos[1] - self.size)
            p2 = (self.pos[0] - self.size * 0.866, self.pos[1] + self.size * 0.5)
            p3 = (self.pos[0] + self.size * 0.866, self.pos[1] + self.size * 0.5)
            points = [p1, p2, p3]
            pygame.gfxdraw.filled_polygon(surface, [ (int(p[0]), int(p[1])) for p in points], self.color)
            pygame.gfxdraw.aapolygon(surface, [ (int(p[0]), int(p[1])) for p in points], self.color)
        elif self.stats["shape"] == "square":
            rect = pygame.Rect(self.pos[0] - self.size, self.pos[1] - self.size, self.size * 2, self.size * 2)
            pygame.draw.rect(surface, self.color, rect, border_radius=3)
        elif self.stats["shape"] == "hexagon":
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                x = self.pos[0] + self.size * math.cos(angle) * 1.5
                y = self.pos[1] + self.size * math.sin(angle) * 1.5
                points.append((int(x), int(y)))
            pygame.gfxdraw.filled_polygon(surface, points, self.color)
            pygame.gfxdraw.aapolygon(surface, points, self.color)

        # Activation Pulse
        if self.is_activating:
            radius = int(self.activation_radius)
            alpha = int(200 * (1 - (radius / self.max_activation_radius)))
            if alpha > 0 and radius > 0:
                # Gradient effect
                for i in range(radius, max(0, radius-15), -2):
                    grad_alpha = int(alpha * (1 - (radius - i) / 15))
                    pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), i, self.color + (grad_alpha,))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your base from waves of incoming enemies by strategically placing magical shrines with different effects."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the reticle. Press space to deploy a shrine and shift to cycle between shrine types."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_HILL_1 = (25, 40, 55)
        self.COLOR_HILL_2 = (35, 55, 75)
        self.COLOR_UI = (220, 220, 240)
        self.COLOR_PLAYER_BASE = (0, 255, 150)
        self.COLOR_RETICLE = (255, 255, 255)
        self.COLOR_PORTAL = (100, 0, 150)

        # Game parameters
        self.MAX_STEPS = 3000
        self.TOTAL_WAVES = 20
        self.STARTING_HEALTH = 100
        self.PLAYER_BASE_POS = np.array([self.WIDTH / 2, self.HEIGHT - 20])
        self.PLAYER_BASE_RADIUS = 30
        self.RETICLE_SPEED = 10
        self.DEPLOY_COOLDOWN_FRAMES = 15
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = 0
        self.wave_number = 0
        self.enemies = []
        self.shrines = []
        self.particles = []
        self.reticle_pos = [0, 0]
        self.shrine_types_unlocked = []
        self.current_shrine_type_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.time_to_next_wave = 0
        self.deploy_cooldown = 0
        self.reward_this_step = 0
        self.spawn_portals = []
        
        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() is for debugging and not needed in the final version.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_health = self.STARTING_HEALTH
        self.wave_number = 0
        self.enemies = []
        self.shrines = []
        self.particles = []
        self.reticle_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        self.shrine_types_unlocked = ["repel"]
        self.current_shrine_type_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.time_to_next_wave = 3 * 30 # 3 seconds
        self.deploy_cooldown = 0
        self.reward_this_step = 0
        
        if not self.spawn_portals:
            self.spawn_portals = [
                (self.WIDTH * 0.1, 50),
                (self.WIDTH * 0.5, 50),
                (self.WIDTH * 0.9, 50)
            ]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = -0.001 # Small penalty for existing
        
        self._handle_input(action)
        self._update_game_state()
        
        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            if self.win:
                reward += 50
            elif self.player_health <= 0:
                reward -= 100
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # Reticle Movement
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED
        
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.HEIGHT - 50) # Prevent placing in UI area

        # Deploy Shrine (Space on press)
        if space_pressed and not self.prev_space_held and self.deploy_cooldown <= 0:
            shrine_type = self.shrine_types_unlocked[self.current_shrine_type_idx]
            self.shrines.append(Shrine(self.reticle_pos, shrine_type))
            self.deploy_cooldown = self.DEPLOY_COOLDOWN_FRAMES
            # SFX: Deploy_Shrine.wav
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append(Particle(self.reticle_pos, vel, random.randint(2, 4), (255,255,255), 20))


        # Cycle Shrine Type (Shift on press)
        if shift_pressed and not self.prev_shift_held:
            self.current_shrine_type_idx = (self.current_shrine_type_idx + 1) % len(self.shrine_types_unlocked)
            # SFX: Cycle_Weapon.wav

        self.prev_space_held = space_pressed
        self.prev_shift_held = shift_pressed

    def _update_game_state(self):
        if self.deploy_cooldown > 0:
            self.deploy_cooldown -= 1

        # Wave management
        if not self.enemies and self.wave_number <= self.TOTAL_WAVES:
            if self.time_to_next_wave > 0:
                self.time_to_next_wave -= 1
            else:
                if self.wave_number > 0:
                    self.reward_this_step += 1.0 # Wave clear bonus
                    self.score += 100 * self.wave_number
                self._spawn_wave()

        # Update shrines
        active_shrines = []
        for shrine in self.shrines:
            if shrine.update():
                active_shrines.append(shrine)
        self.shrines = active_shrines

        # Update enemies
        active_enemies = []
        for enemy in self.enemies:
            if enemy.update(self.PLAYER_BASE_POS):
                # Check collision with player base
                if np.linalg.norm(enemy.pos - self.PLAYER_BASE_POS) < self.PLAYER_BASE_RADIUS + enemy.radius:
                    self.player_health -= 25
                    self.reward_this_step -= 5.0
                    self.score -= 50
                    # SFX: Base_Hit.wav
                    for _ in range(30):
                        angle = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(2, 5)
                        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                        self.particles.append(Particle(enemy.pos, vel, random.randint(3, 6), (255, 80, 80), 40))
                else:
                    active_enemies.append(enemy)
            else: # Enemy health <= 0
                self.reward_this_step += 0.5
                self.score += 25
                # SFX: Enemy_Destroyed.wav
                for _ in range(15):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(1, 4)
                    vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                    self.particles.append(Particle(enemy.pos, vel, random.randint(2, 4), enemy.color, 30))

        self.enemies = active_enemies
        
        # Shrine-Enemy interaction (on keypress, not continuous)
        if self.deploy_cooldown == self.DEPLOY_COOLDOWN_FRAMES - 1: # Hacky way to check for shrine deploy this frame
            shrine = self.shrines[-1]
            shrine.is_activating = True
            # SFX: Activate_Shrine.wav
            for _ in range(40):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                self.particles.append(Particle(shrine.pos, vel, random.randint(2, 5), shrine.color, 50))


        for shrine in self.shrines:
            if shrine.is_activating and shrine.activation_radius > 0:
                for enemy in self.enemies:
                    dist_vec = enemy.pos - shrine.pos
                    dist = np.linalg.norm(dist_vec)
                    if dist < shrine.activation_radius + enemy.radius and dist > 0:
                        if shrine.type == "repel":
                            repel_force = 15 * (1 - dist / shrine.max_activation_radius)
                            enemy.pos += (dist_vec / dist) * repel_force
                            self.reward_this_step += 0.1
                        elif shrine.type == "slow":
                            enemy.slow_timer = 150 # 5 seconds
                            self.reward_this_step += 0.1
                        elif shrine.type == "damage":
                            enemy.damage_timer = 150 # 5 seconds DoT
                            self.reward_this_step += 0.1

        # Update particles
        self.particles = [p for p in self.particles if p.update()]

    def _spawn_wave(self):
        self.wave_number += 1
        if self.wave_number > self.TOTAL_WAVES:
            self.win = True
            return

        # Unlock new shrines
        if self.wave_number == 5 and "slow" not in self.shrine_types_unlocked:
            self.shrine_types_unlocked.append("slow")
            self.reward_this_step += 5.0
        if self.wave_number == 10 and "damage" not in self.shrine_types_unlocked:
            self.shrine_types_unlocked.append("damage")
            self.reward_this_step += 5.0

        num_enemies = 2 + self.wave_number
        enemy_speed = 0.5 + self.wave_number * 0.05
        enemy_health = 50 + self.wave_number * 5
        
        for _ in range(num_enemies):
            pos = random.choice(self.spawn_portals)
            offset = (random.uniform(-20, 20), random.uniform(-20, 20))
            self.enemies.append(Enemy((pos[0]+offset[0], pos[1]+offset[1]), enemy_speed, enemy_health, 8))
        
        self.time_to_next_wave = 5 * 30 # 5 seconds between waves

    def _calculate_reward(self):
        reward = self.reward_this_step
        self.reward_this_step = 0
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.win:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background hills
        pygame.draw.polygon(self.screen, self.COLOR_HILL_1, [(0, 400), (0, 150), (350, 400)])
        pygame.draw.polygon(self.screen, self.COLOR_HILL_2, [(640, 400), (640, 100), (150, 400)])

        # Player Base
        pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_BASE_POS[0]), int(self.PLAYER_BASE_POS[1]), self.PLAYER_BASE_RADIUS, self.COLOR_PLAYER_BASE + (20,))
        pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_BASE_POS[0]), int(self.PLAYER_BASE_POS[1]), self.PLAYER_BASE_RADIUS - 5, self.COLOR_PLAYER_BASE + (40,))
        pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_BASE_POS[0]), int(self.PLAYER_BASE_POS[1]), self.PLAYER_BASE_RADIUS, self.COLOR_PLAYER_BASE)

        # Spawn Portals
        for pos in self.spawn_portals:
            pulse = math.sin(self.steps * 0.05) * 5
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(20 + pulse), self.COLOR_PORTAL + (100,))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, (0,0,0))
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(20 + pulse), self.COLOR_PORTAL)

        for p in self.particles: p.draw(self.screen)
        for s in self.shrines: s.draw(self.screen)
        for e in self.enemies: e.draw(self.screen, self.steps)

        # Trajectory Line
        start_pos = (int(self.PLAYER_BASE_POS[0]), int(self.PLAYER_BASE_POS[1] - self.PLAYER_BASE_RADIUS))
        end_pos = (int(self.reticle_pos[0]), int(self.reticle_pos[1]))
        dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            num_dashes = int(dist / 10)
            for i in range(num_dashes):
                if i % 2 == 0:
                    p1 = (start_pos[0] + dx * (i / num_dashes), start_pos[1] + dy * (i / num_dashes))
                    p2 = (start_pos[0] + dx * ((i + 1) / num_dashes), start_pos[1] + dy * ((i + 1) / num_dashes))
                    pygame.draw.line(self.screen, self.COLOR_RETICLE, p1, p2, 1)

        # Reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        pygame.gfxdraw.aacircle(self.screen, rx, ry, 12, self.COLOR_RETICLE)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - 8, ry), (rx + 8, ry))
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - 8), (rx, ry + 8))
        
    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.STARTING_HEALTH)
        health_bar_width = 200
        health_bar_fill = int(health_ratio * health_bar_width)
        pygame.draw.rect(self.screen, (255, 0, 0), (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (10, 10, health_bar_fill, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI, (10, 10, health_bar_width, 20), 2)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, 10))

        # Wave Counter
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Shrine Type Indicator
        current_type = self.shrine_types_unlocked[self.current_shrine_type_idx]
        type_stats = Shrine.SHRINE_TYPES[current_type]
        type_text = self.font_small.render(f"Shrine: {current_type.upper()}", True, type_stats["color"])
        self.screen.blit(type_text, (self.WIDTH / 2 - type_text.get_width() / 2, self.HEIGHT - 30))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 100) if self.win else (255, 50, 50)
            end_text = self.font_large.render(message, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the autograder.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible display driver
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Shrine Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Display final frame for a moment before reset
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            
            obs, info = env.reset() # Auto-reset for continuous play
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()