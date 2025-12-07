import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:34:33.058934
# Source Brief: brief_02251.md
# Brief Index: 2251
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, life, color, size_range):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(size_range[0], size_range[1])

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        self.vel[1] += 0.05 # a little gravity

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)), special_flags=pygame.BLEND_RGBA_ADD)

class Projectile:
    def __init__(self, pos, angle, p_type, speed):
        self.pos = list(pos)
        self.angle = angle
        self.type = p_type
        self.speed = speed
        self.vel = [math.cos(self.angle) * self.speed, math.sin(self.angle) * self.speed]
        self.trail = []

    def update(self):
        self.trail.append(list(self.pos))
        if len(self.trail) > 5:
            self.trail.pop(0)
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

class Enemy:
    def __init__(self, pos, health, pattern_type, speed):
        self.pos = list(pos)
        self.health = health
        self.max_health = health
        self.pattern_type = pattern_type
        self.speed = speed
        self.size = 12
        self.glitch_timer = 0
        self.glitch_offset = [0, 0]
        self.pattern_param = random.uniform(0, 2 * math.pi) # Initial phase for sine wave

    def update(self, player_pos):
        if self.pattern_type == "sine":
            self.pos[0] += self.speed
            self.pos[1] += math.sin(self.pos[0] / 30 + self.pattern_param) * 2
        elif self.pattern_type == "diagonal":
            self.pos[0] += self.speed
            self.pos[1] += self.speed
        elif self.pattern_type == "homing":
            angle_to_player = math.atan2(player_pos[1] - self.pos[1], player_pos[0] - self.pos[0])
            self.pos[0] += math.cos(angle_to_player) * self.speed * 0.7
            self.pos[1] += math.sin(angle_to_player) * self.speed * 0.7

        self.glitch_timer -= 1
        if self.glitch_timer <= 0:
            self.glitch_timer = random.randint(5, 15)
            self.glitch_offset = [random.randint(-3, 3), random.randint(-3, 3)]
        elif self.glitch_timer > 2:
            self.glitch_offset = [0, 0]

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

class PowerUp:
    def __init__(self, pos, p_type):
        self.pos = list(pos)
        self.type = p_type
        self.size = 10
        self.bob_angle = 0

    def update(self):
        self.bob_angle += 0.1
        self.pos[1] += math.sin(self.bob_angle) * 0.2

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Survive until dawn in this top-down arcade shooter. Rotate your ship to fend off waves of glitchy enemies and collect power-ups."
    )
    user_guide = (
        "Controls: ←/↓ to rotate counter-clockwise, →/↑ to rotate clockwise. "
        "Press space to fire and shift to switch weapons."
    )
    auto_advance = True

    unlocked_weapons = {"default": "SINGLE SHOT"}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_CABINET = (10, 10, 15)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_SHIELD = (100, 150, 255)
        self.COLOR_ENEMY = (255, 50, 100)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_AMMO = (255, 180, 0)
        self.COLOR_SHIELD_PU = (0, 150, 255)
        self.COLOR_RAPID_PU = (255, 0, 255)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_RED = (255, 50, 50)
        self.COLOR_GREEN = (50, 255, 50)

        # Game constants
        self.PLAYER_POS = (self.WIDTH // 2, self.HEIGHT // 2)
        self.PLAYER_ROTATION_SPEED = 0.1
        self.MAX_STEPS = 1000
        
        # Initialize state variables
        # self.reset() # reset is called by the environment runner

        # Run validation check
        # self.validate_implementation() # Commented out for production
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_angle = -math.pi / 2
        self.player_health = 100
        self.player_max_health = 100
        self.player_ammo = 30
        self.player_max_ammo = 30
        
        self.player_shield_timer = 0
        self.player_rapid_fire_timer = 0
        self.weapon_switch_cooldown = 0
        
        self.fire_cooldown = 0
        self.prev_shift_held = False
        
        self.current_weapon_idx = 0
        self.available_weapons = list(self.unlocked_weapons.keys())

        self.enemies = []
        self.projectiles = []
        self.powerups = []
        self.particles = []

        self.base_spawn_rate = 50
        self.enemy_spawn_timer = self.base_spawn_rate
        self.base_enemy_health = 3
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Rotation
        if movement in [1, 4]: # Up, Right -> Clockwise
            self.player_angle += self.PLAYER_ROTATION_SPEED
        elif movement in [2, 3]: # Down, Left -> Counter-Clockwise
            self.player_angle -= self.PLAYER_ROTATION_SPEED

        # Weapon Switch
        if shift_held and not self.prev_shift_held and self.weapon_switch_cooldown <= 0:
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.available_weapons)
            self.weapon_switch_cooldown = 10 # 1/3 second cooldown
            # Sound: weapon_switch.wav
        
        # Firing
        fire_rate = 10
        if self.player_rapid_fire_timer > 0:
            fire_rate = 3
        
        if space_held and self.fire_cooldown <= 0 and self.player_ammo > 0:
            self.fire_cooldown = fire_rate
            self.player_ammo -= 1
            
            current_weapon = self.available_weapons[self.current_weapon_idx]
            self._spawn_projectiles(current_weapon)
            self._create_particles(self.PLAYER_POS, 5, (255, 255, 200), 2, 0.5) # Muzzle flash
            # Sound: shoot.wav

        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self.steps += 1
        reward += 0.01 # Survival reward

        # Cooldowns
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.player_shield_timer > 0: self.player_shield_timer -= 1
        if self.player_rapid_fire_timer > 0: self.player_rapid_fire_timer -= 1
        if self.weapon_switch_cooldown > 0: self.weapon_switch_cooldown -= 1
        
        # Update Projectiles
        for p in self.projectiles[:]:
            p.update()
            if not (0 < p.pos[0] < self.WIDTH and 0 < p.pos[1] < self.HEIGHT):
                self.projectiles.remove(p)

        # Update Enemies
        for e in self.enemies[:]:
            e.update(self.PLAYER_POS)
            # Player collision
            if math.hypot(e.pos[0] - self.PLAYER_POS[0], e.pos[1] - self.PLAYER_POS[1]) < e.size + 15:
                if self.player_shield_timer <= 0:
                    self.player_health -= 10
                    self._create_particles(self.PLAYER_POS, 20, self.COLOR_RED, 3, 2)
                    # Sound: player_hit.wav
                if e in self.enemies: self.enemies.remove(e)
                self._create_particles(e.pos, 30, self.COLOR_ENEMY, 4, 3) # Enemy explodes on player
                # Sound: enemy_explosion.wav

        # Update Particles
        for p in self.particles[:]:
            p.update()
            if p.life <= 0: self.particles.remove(p)
        
        # Update Powerups
        for pu in self.powerups[:]:
            pu.update()
            if math.hypot(pu.pos[0] - self.PLAYER_POS[0], pu.pos[1] - self.PLAYER_POS[1]) < pu.size + 15:
                self._apply_powerup(pu.type)
                reward += 2
                self.powerups.remove(pu)
                self._create_particles(pu.pos, 30, (100, 255, 100), 4, 2)
                # Sound: powerup_collect.wav

        # --- Collision Detection (Projectile vs Enemy) ---
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if math.hypot(p.pos[0] - e.pos[0], p.pos[1] - e.pos[1]) < e.size:
                    reward += 1
                    is_dead = e.take_damage(1)
                    self._create_particles(p.pos, 10, self.COLOR_PROJECTILE, 2, 1) # Hit spark
                    # Sound: enemy_hit.wav
                    if p in self.projectiles and p.type != "pierce":
                        self.projectiles.remove(p)
                    
                    if is_dead:
                        reward += 5
                        self.score += 10
                        if e in self.enemies: self.enemies.remove(e)
                        self._create_particles(e.pos, 50, self.COLOR_ENEMY, 5, 4) # Explosion
                        # Sound: enemy_explosion.wav
                        
                        # Chance to drop powerup
                        if random.random() < 0.2:
                            ptype = random.choice(["ammo", "shield", "rapid"])
                            self.powerups.append(PowerUp(e.pos, ptype))
                    break

        # --- Spawning ---
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            spawn_rate_scaling = 1 - (self.steps / self.MAX_STEPS) * 0.8
            self.enemy_spawn_timer = int(self.base_spawn_rate * spawn_rate_scaling)
            
            enemy_health = self.base_enemy_health + (self.steps // 200)
            self._spawn_enemy(enemy_health)

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.player_health <= 0:
            reward = -100
            terminated = True
            self.game_over = True
            # Sound: game_over.wav
        
        if self.steps >= self.MAX_STEPS:
            reward = 100
            terminated = True # Or truncated, depending on game design. Let's use terminated for a win condition.
            self.game_over = True
            self.win = True
            # Unlock new weapons
            if "spread" not in self.unlocked_weapons:
                self.unlocked_weapons["spread"] = "SPREAD SHOT"
            elif "pierce" not in self.unlocked_weapons:
                self.unlocked_weapons["pierce"] = "PIERCE SHOT"
            # Sound: win.wav

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_projectiles(self, weapon_type):
        if weapon_type == "default":
            self.projectiles.append(Projectile(self.PLAYER_POS, self.player_angle, "default", 10))
        elif weapon_type == "spread":
            for i in range(-1, 2):
                angle = self.player_angle + i * 0.2
                self.projectiles.append(Projectile(self.PLAYER_POS, angle, "default", 8))
        elif weapon_type == "pierce":
            self.projectiles.append(Projectile(self.PLAYER_POS, self.player_angle, "pierce", 12))

    def _spawn_enemy(self, health):
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top": pos = [random.uniform(0, self.WIDTH), -20]
        elif side == "bottom": pos = [random.uniform(0, self.WIDTH), self.HEIGHT + 20]
        elif side == "left": pos = [-20, random.uniform(0, self.HEIGHT)]
        else: pos = [self.WIDTH + 20, random.uniform(0, self.HEIGHT)]
        
        pattern = random.choice(["sine", "diagonal", "homing"])
        speed = random.uniform(1.0, 2.0) + (self.steps / self.MAX_STEPS) * 2.0
        
        # Ensure diagonal enemies move towards screen
        if pattern == "diagonal":
            if pos[0] > self.WIDTH / 2: speed = -abs(speed)
            else: speed = abs(speed)
        elif pattern == "sine":
            if pos[0] > self.WIDTH / 2: speed = -abs(speed)
            else: speed = abs(speed)

        self.enemies.append(Enemy(pos, health, pattern, speed))

    def _apply_powerup(self, ptype):
        if ptype == "ammo":
            self.player_ammo = min(self.player_max_ammo, self.player_ammo + 20)
        elif ptype == "shield":
            self.player_shield_timer = 300 # 10 seconds
        elif ptype == "rapid":
            self.player_rapid_fire_timer = 240 # 8 seconds

    def _create_particles(self, pos, count, color, speed_max, size_max):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, life, color, (1, size_max)))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "weapon": self.available_weapons[self.current_weapon_idx]
        }
    
    def _render_background(self):
        # Cabinet sides
        pygame.draw.rect(self.screen, self.COLOR_CABINET, (0, 0, 40, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_CABINET, (self.WIDTH - 40, 0, 40, self.HEIGHT))
        # Grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_game(self):
        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Powerups
        for pu in self.powerups:
            color = self.COLOR_AMMO
            if pu.type == "shield": color = self.COLOR_SHIELD_PU
            elif pu.type == "rapid": color = self.COLOR_RAPID_PU
            
            pulse = (math.sin(pu.bob_angle * 2) + 1) / 2 * 4
            size = pu.size + pulse
            
            pygame.draw.circle(self.screen, tuple(c*0.5 for c in color), (int(pu.pos[0]), int(pu.pos[1])), int(size+2))
            pygame.draw.circle(self.screen, color, (int(pu.pos[0]), int(pu.pos[1])), int(size))
            
            letter = pu.type[0].upper()
            text = self.font_small.render(letter, True, self.COLOR_BG)
            self.screen.blit(text, (int(pu.pos[0] - text.get_width()/2), int(pu.pos[1] - text.get_height()/2)))

        # Enemies
        for e in self.enemies:
            pos = (int(e.pos[0] + e.glitch_offset[0]), int(e.pos[1] + e.glitch_offset[1]))
            # Glitch effect
            if e.glitch_timer < 3:
                color = random.choice([(255,0,255), (0,255,255)])
            else:
                color = self.COLOR_ENEMY
            
            pygame.draw.rect(self.screen, color, (pos[0] - e.size//2, pos[1] - e.size//2, e.size, e.size))
            # Health bar
            if e.health < e.max_health:
                health_pct = e.health / e.max_health
                pygame.draw.rect(self.screen, self.COLOR_RED, (pos[0] - e.size//2, pos[1] - e.size//2 - 5, e.size, 3))
                pygame.draw.rect(self.screen, self.COLOR_GREEN, (pos[0] - e.size//2, pos[1] - e.size//2 - 5, int(e.size * health_pct), 3))

        # Player
        player_points = [
            (20, 0), (-10, -10), (-5, 0), (-10, 10)
        ]
        rotated_points = []
        for x, y in player_points:
            rx = self.PLAYER_POS[0] + x * math.cos(self.player_angle) - y * math.sin(self.player_angle)
            ry = self.PLAYER_POS[1] + x * math.sin(self.player_angle) + y * math.cos(self.player_angle)
            rotated_points.append((int(rx), int(ry)))
        
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, rotated_points)
        pygame.draw.aalines(self.screen, tuple(min(255, c+50) for c in self.COLOR_PLAYER), True, rotated_points)
        
        # Shield effect
        if self.player_shield_timer > 0:
            alpha = 50 + (self.player_shield_timer % 15) * 4
            pygame.gfxdraw.filled_circle(self.screen, int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1]), 25, self.COLOR_PLAYER_SHIELD + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, int(self.PLAYER_POS[0]), int(self.PLAYER_POS[1]), 25, self.COLOR_PLAYER_SHIELD)

        # Projectiles
        for p in self.projectiles:
            # Trail
            for i, pos in enumerate(p.trail):
                alpha = int(255 * (i / len(p.trail)))
                color = (self.COLOR_PROJECTILE[0], self.COLOR_PROJECTILE[1], self.COLOR_PROJECTILE[2], alpha)
                pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 2, 1)
            # Head
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p.pos[0]), int(p.pos[1])), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (50, 10))

        # Steps
        steps_text = self.font_small.render(f"DAWN IN: {self.MAX_STEPS - self.steps}", True, self.COLOR_WHITE)
        self.screen.blit(steps_text, (self.WIDTH - 50 - steps_text.get_width(), 10))

        # Health Bar
        health_pct = max(0, self.player_health / self.player_max_health)
        pygame.draw.rect(self.screen, self.COLOR_RED, (50, self.HEIGHT - 30, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_GREEN, (50, self.HEIGHT - 30, int(200 * health_pct), 20))
        
        # Ammo Bar
        ammo_pct = max(0, self.player_ammo / self.player_max_ammo)
        pygame.draw.rect(self.screen, (80, 60, 0), (self.WIDTH - 250, self.HEIGHT - 30, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_AMMO, (self.WIDTH - 250, self.HEIGHT - 30, int(200 * ammo_pct), 20))

        # Weapon display
        weapon_name = self.unlocked_weapons[self.available_weapons[self.current_weapon_idx]]
        weapon_text = self.font_small.render(weapon_name, True, self.COLOR_WHITE)
        self.screen.blit(weapon_text, (self.WIDTH/2 - weapon_text.get_width()/2, self.HEIGHT - 30))

        # Game Over / Win Text
        if self.game_over:
            if self.win:
                text = self.font_large.render("DAWN ARRIVES", True, self.COLOR_GREEN)
            else:
                text = self.font_large.render("GAME OVER", True, self.COLOR_RED)
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2 - 50))
    
    def close(self):
        pygame.quit()

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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For this to work, you must comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # And install the full pygame library: pip install pygame
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # To run with a display, pygame.display.set_mode must be called
    # and the dummy video driver must be disabled.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Arcade Defender")
        has_display = True
    except pygame.error:
        print("No display available. Running headlessly.")
        has_display = False

    clock = pygame.time.Clock()

    movement = 0
    space_held = 0
    shift_held = 0
    
    total_reward = 0

    running = True
    while running:
        if has_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            
            # Map keyboard to MultiDiscrete action space
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            else: movement = 0

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
        else: # If no display, just sample actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            if has_display:
                pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display
        if has_display:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()