import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:13:57.688311
# Source Brief: brief_00941.md
# Brief Index: 941
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Tempo Tower: A rhythm-action tower defense game.
    Defend the tower from descending enemies by firing colored projectiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your tower from descending waves of enemies by firing colored projectiles. "
        "Manage your energy and switch projectile types to counter different threats."
    )
    user_guide = (
        "Controls: ↑/↓ arrows to aim the cannon. Press space to fire. "
        "Press shift to cycle between projectile types."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    TOWER_BASE_Y = 380
    TOWER_WIDTH = 80
    TOWER_HEIGHT = 60
    CANNON_Y_OFFSET = -20
    MAX_STEPS = 5000
    MAX_WAVES = 50

    # Colors
    COLOR_BG = (15, 19, 25)
    COLOR_GRID = (30, 35, 45)
    COLOR_TOWER = (100, 110, 120)
    COLOR_TOWER_DMG = (255, 80, 80)
    
    # Projectile States
    STATE_BLUE = 0  # Fast, single target
    STATE_GREEN = 1 # Slower, higher damage
    STATE_YELLOW = 2 # Splash damage (unlockable)

    PROJECTILE_COLORS = {
        STATE_BLUE: (0, 150, 255),
        STATE_GREEN: (80, 255, 80),
        STATE_YELLOW: (255, 220, 50),
    }
    PROJECTILE_PROPS = {
        STATE_BLUE: {'speed': 12, 'damage': 1, 'cost': 1, 'cooldown': 5, 'radius': 5},
        STATE_GREEN: {'speed': 8, 'damage': 2, 'cost': 2, 'cooldown': 8, 'radius': 6},
        STATE_YELLOW: {'speed': 7, 'damage': 1.5, 'cost': 5, 'cooldown': 15, 'radius': 7, 'splash_radius': 60},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_health = 0
        self.max_tower_health = 100
        self.resources = 0
        self.max_resources = 100
        self.projectile_state = 0
        self.aim_angle = 0.0
        self.wave_number = 0
        self.enemy_spawn_timer = 0
        self.fire_cooldown = 0
        self.shift_cooldown = 0
        self.tower_damage_flash = 0
        self.upgrades = {}
        
        self.projectiles = []
        self.enemies = []
        self.particles = []

        # Initialize state by calling reset
        # self.reset() # This is typically called by the training loop, not in __init__
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.tower_health = self.max_tower_health
        self.resources = 50
        self.projectile_state = self.STATE_BLUE
        self.aim_angle = 0.0  # Straight up
        
        self.wave_number = 1
        self.enemy_spawn_timer = 120 # Delay before first wave
        
        self.fire_cooldown = 0
        self.shift_cooldown = 0
        self.tower_damage_flash = 0
        
        self.upgrades = {"damage_multiplier": 1.0, "splash_unlocked": False}

        self.projectiles = []
        self.enemies = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # --- Action Handling ---
            self._handle_actions(action)
            
            # --- Game Logic Updates ---
            self._update_cooldowns()
            self._update_resources()
            
            # --- Entity Updates ---
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
            
            # --- Wave Management ---
            self._manage_waves()
        
        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.tower_health <= 0:
                reward -= 100 # Loss penalty
            elif self.wave_number > self.MAX_WAVES:
                reward += 100 # Victory bonus
                self.score += 1000 # Score bonus for winning

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    # ==========================================================================
    # --- Action and State Update Methods ---
    # ==========================================================================

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Aiming: Up/Down adjust angle
        angle_speed = 2.5
        if movement == 1: # Up
            self.aim_angle = max(-80, self.aim_angle - angle_speed)
        elif movement == 2: # Down
            self.aim_angle = min(80, self.aim_angle + angle_speed)

        # Firing
        if space_held and self.fire_cooldown == 0:
            self._fire_projectile()

        # State Switching
        if shift_held and self.shift_cooldown == 0:
            self._cycle_projectile_state()
            self.shift_cooldown = 15 # Prevent rapid cycling

    def _fire_projectile(self):
        props = self.PROJECTILE_PROPS[self.projectile_state]
        if self.resources >= props['cost']:
            self.resources -= props['cost']
            self.fire_cooldown = props['cooldown']
            
            angle_rad = math.radians(self.aim_angle - 90)
            start_pos = pygame.Vector2(self.WIDTH / 2, self.TOWER_BASE_Y + self.CANNON_Y_OFFSET)
            velocity = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * props['speed']
            
            self.projectiles.append({
                "pos": start_pos,
                "vel": velocity,
                "state": self.projectile_state,
                "radius": props['radius']
            })
            # sfx: shoot_laser_01.wav, shoot_laser_02.wav, etc.

    def _cycle_projectile_state(self):
        self.projectile_state = (self.projectile_state + 1) % 3
        # If splash is not unlocked, skip yellow
        if self.projectile_state == self.STATE_YELLOW and not self.upgrades["splash_unlocked"]:
            self.projectile_state = self.STATE_BLUE
        # sfx: state_switch.wav

    def _update_cooldowns(self):
        self.fire_cooldown = max(0, self.fire_cooldown - 1)
        self.shift_cooldown = max(0, self.shift_cooldown - 1)
        self.tower_damage_flash = max(0, self.tower_damage_flash - 1)

    def _update_resources(self):
        # Regenerate 1 resource every 20 steps
        if self.steps % 20 == 0:
            self.resources = min(self.max_resources, self.resources + 1)

    def _update_projectiles(self):
        reward = 0
        # Iterate backwards to allow removal
        for i in range(len(self.projectiles) - 1, -1, -1):
            proj = self.projectiles[i]
            proj["pos"] += proj["vel"]
            
            # Check for collision with enemies
            hit = False
            for enemy in self.enemies:
                if proj["pos"].distance_to(enemy["pos"]) < proj["radius"] + enemy["radius"]:
                    reward += self._handle_projectile_hit(proj, enemy)
                    hit = True
                    break # Projectile can only hit one enemy directly
            
            # Remove projectile if it hit or is off-screen
            if hit or proj["pos"].y < 0 or proj["pos"].x < 0 or proj["pos"].x > self.WIDTH:
                self.projectiles.pop(i)
        return reward

    def _handle_projectile_hit(self, proj, enemy):
        reward = 0.1 # Reward for hitting
        props = self.PROJECTILE_PROPS[proj['state']]
        damage = props['damage'] * self.upgrades['damage_multiplier']
        
        # Handle direct hit
        enemy['health'] -= damage
        self._create_particles(proj['pos'], self.PROJECTILE_COLORS[proj['state']], 10, 3)
        # sfx: hit_01.wav
        
        # Handle splash damage
        if proj['state'] == self.STATE_YELLOW and self.upgrades['splash_unlocked']:
            splash_radius = props['splash_radius']
            self._create_particles(proj['pos'], self.PROJECTILE_COLORS[proj['state']], 30, 5, 1.5, splash_radius/20)
            for other_enemy in self.enemies:
                if other_enemy is not enemy and proj['pos'].distance_to(other_enemy['pos']) < splash_radius:
                    other_enemy['health'] -= damage * 0.5 # Splash does half damage
                    reward += 0.05 # Smaller reward for splash hit
        
        return reward

    def _update_enemies(self):
        reward = 0
        for i in range(len(self.enemies) - 1, -1, -1):
            enemy = self.enemies[i]
            enemy["pos"].y += enemy["speed"]
            
            # Wobble effect
            enemy["pos"].x = enemy["origin_x"] + math.sin(self.steps * 0.1 + enemy["phase"]) * 10

            # Check for destruction
            if enemy["health"] <= 0:
                reward += 1.0
                self.score += 10 * self.wave_number
                self._create_particles(enemy["pos"], (255, 100, 100), 25, 5)
                self.enemies.pop(i)
                # sfx: enemy_explode.wav
                continue

            # Check for reaching the tower
            if enemy["pos"].y > self.TOWER_BASE_Y:
                self.tower_health -= 10
                self.tower_damage_flash = 15
                reward -= 10.0
                self.enemies.pop(i)
                # sfx: tower_hit.wav
                continue
        return reward

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.pop(i)

    def _manage_waves(self):
        # Check for unlocks
        if self.wave_number >= 10 and not self.upgrades["splash_unlocked"]:
            self.upgrades["splash_unlocked"] = True
            # Visual/Audio cue for unlock
        if self.wave_number >= 25 and self.upgrades["damage_multiplier"] == 1.0:
            self.upgrades["damage_multiplier"] = 1.5
            # Visual/Audio cue for unlock

        # If all enemies are gone and spawn timer is up, start next wave
        if not self.enemies and self.enemy_spawn_timer <= 0:
            self.wave_number += 1
            if self.wave_number > self.MAX_WAVES:
                return # Game won
            
            # Spawn new wave
            num_enemies = 5 + self.wave_number
            enemy_speed = 1.0 + (self.wave_number // 5) * 0.1
            enemy_health = 1.0 + (self.wave_number // 10) * 1.0
            
            for _ in range(num_enemies):
                self.enemies.append({
                    "pos": pygame.Vector2(random.uniform(50, self.WIDTH - 50), random.uniform(-150, -50)),
                    "origin_x": random.uniform(50, self.WIDTH - 50),
                    "phase": random.uniform(0, 2 * math.pi),
                    "speed": random.uniform(enemy_speed * 0.8, enemy_speed * 1.2),
                    "health": enemy_health,
                    "max_health": enemy_health,
                    "radius": 12,
                })
            self.enemy_spawn_timer = 240 # Cooldown between waves
        
        self.enemy_spawn_timer = max(0, self.enemy_spawn_timer - 1)

    def _check_termination(self):
        return (
            self.tower_health <= 0
            or self.wave_number > self.MAX_WAVES
        )

    # ==========================================================================
    # --- Rendering Methods ---
    # ==========================================================================

    def _get_observation(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_grid()

        # Game Elements
        self._render_tower()
        self._render_aim_assist()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()

        # UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_tower(self):
        # Base
        base_rect = pygame.Rect(
            self.WIDTH / 2 - self.TOWER_WIDTH / 2, 
            self.TOWER_BASE_Y, 
            self.TOWER_WIDTH, 
            self.TOWER_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_TOWER, base_rect, border_radius=3)
        
        # Cannon
        cannon_pos = (self.WIDTH / 2, self.TOWER_BASE_Y + self.CANNON_Y_OFFSET)
        angle_rad = math.radians(self.aim_angle - 90)
        cannon_end = (
            cannon_pos[0] + 30 * math.cos(angle_rad),
            cannon_pos[1] + 30 * math.sin(angle_rad)
        )
        pygame.draw.line(self.screen, self.COLOR_TOWER, cannon_pos, cannon_end, 10)
        pygame.gfxdraw.aacircle(self.screen, int(cannon_pos[0]), int(cannon_pos[1]), 10, self.COLOR_TOWER)
        
        # Damage flash
        if self.tower_damage_flash > 0:
            flash_surface = pygame.Surface((self.TOWER_WIDTH, self.TOWER_HEIGHT), pygame.SRCALPHA)
            alpha = int(150 * (self.tower_damage_flash / 15))
            flash_surface.fill((*self.COLOR_TOWER_DMG, alpha))
            self.screen.blit(flash_surface, base_rect.topleft)

    def _render_aim_assist(self):
        angle_rad = math.radians(self.aim_angle - 90)
        start_pos = (self.WIDTH / 2, self.TOWER_BASE_Y + self.CANNON_Y_OFFSET)
        for i in range(10, 200, 20):
            p1 = (start_pos[0] + i * math.cos(angle_rad), start_pos[1] + i * math.sin(angle_rad))
            p2 = (start_pos[0] + (i + 10) * math.cos(angle_rad), start_pos[1] + (i + 10) * math.sin(angle_rad))
            if p2[1] < start_pos[1]: # Only draw upwards
                pygame.draw.line(self.screen, (255, 255, 255, 50), p1, p2, 1)

    def _render_projectiles(self):
        for p in self.projectiles:
            color = self.PROJECTILE_COLORS[p['state']]
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'] + 2, (*color, 50))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p['radius'], color)

    def _render_enemies(self):
        for e in self.enemies:
            pos_int = (int(e['pos'].x), int(e['pos'].y))
            color = (220, 50, 80)
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], e['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], e['radius'], color)
            
            # Health bar
            if e['health'] < e['max_health']:
                health_pct = max(0, e['health'] / e['max_health'])
                bar_width = e['radius'] * 2
                bar_height = 4
                bar_x = pos_int[0] - e['radius']
                bar_y = pos_int[1] - e['radius'] - 8
                pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (80, 255, 80), (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_particles(self):
        for p in self.particles:
            life_pct = p['life'] / p['max_life']
            color = (*p['color'], int(255 * life_pct))
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'] * life_pct)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.tower_health / self.max_tower_health)
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, (255, 0, 0), (10, 10, 200 * health_pct, 20))
        health_text = self.font_small.render(f"TOWER: {int(self.tower_health)}/{self.max_tower_health}", True, (255, 255, 255))
        self.screen.blit(health_text, (15, 12))

        # Resources
        resource_text = self.font_large.render(f"ENERGY: {int(self.resources)}", True, (255, 220, 50))
        self.screen.blit(resource_text, (self.WIDTH - resource_text.get_width() - 10, 10))

        # Score and Wave
        score_text = self.font_large.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WIDTH/2 - score_text.get_width()/2, 10))
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, (200, 200, 200))
        self.screen.blit(wave_text, (self.WIDTH/2 - wave_text.get_width()/2, 40))

        # Projectile State Indicator
        indicator_y = self.HEIGHT - 25
        colors = [self.PROJECTILE_COLORS[self.STATE_BLUE], self.PROJECTILE_COLORS[self.STATE_GREEN]]
        if self.upgrades["splash_unlocked"]:
            colors.append(self.PROJECTILE_COLORS[self.STATE_YELLOW])
        
        num_states = len(colors)
        total_width = num_states * 30 - 10
        start_x = self.WIDTH/2 - total_width/2
        
        for i, color in enumerate(colors):
            x = start_x + i * 30
            if i == self.projectile_state:
                pygame.gfxdraw.filled_circle(self.screen, int(x), indicator_y, 12, (*color, 80))
                pygame.gfxdraw.filled_circle(self.screen, int(x), indicator_y, 10, color)
                pygame.gfxdraw.aacircle(self.screen, int(x), indicator_y, 10, color)
            else:
                pygame.gfxdraw.filled_circle(self.screen, int(x), indicator_y, 8, (*color, 100))
                pygame.gfxdraw.aacircle(self.screen, int(x), indicator_y, 8, (*color, 150))


    # ==========================================================================
    # --- Helper Methods ---
    # ==========================================================================

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "tower_health": self.tower_health,
            "resources": self.resources,
        }

    def _create_particles(self, pos, color, count, max_speed, life_mult=1.0, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(15, 30) * life_mult,
                "max_life": 30 * life_mult,
                "color": color,
                "radius": random.randint(2, 4)
            })

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Controls ---
    # Up/Down Arrow: Adjust aim
    # Space: Fire
    # Shift: Switch projectile type
    
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tempo Tower")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Input to Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Limit to 30 FPS for smooth play

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()