import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:08:22.127638
# Source Brief: brief_01505.md
# Brief Index: 1505
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
        "Defend your base with a rotating turret. Destroy waves of incoming enemies and "
        "charge up your shield for temporary invincibility."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to rotate the turret. Press space to fire the laser."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    MAX_STEPS = 10000
    TOTAL_WAVES = 10

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_BASE = (50, 50, 80)
    COLOR_TURRET = (150, 160, 180)
    COLOR_ENEMY = (255, 50, 100)
    COLOR_LASER = (100, 200, 255)
    COLOR_CHARGE_BAR_BG = (40, 40, 60)
    COLOR_CHARGE_BAR_FG = (0, 150, 255)
    COLOR_SHIELD = (255, 220, 50)
    COLOR_HEALTH_HIGH = (80, 220, 80)
    COLOR_HEALTH_MED = (220, 220, 80)
    COLOR_HEALTH_LOW = (220, 80, 80)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (20, 20, 30)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 60)
            self.font_medium = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.turret_health = 0.0
        self.turret_charge = 0.0
        self.turret_angle = 0.0
        self.turret_pos = (0,0)
        self.turret_rotation_speed = 0.0
        self.current_wave = 0
        self.enemies_per_wave = 0
        self.enemies_to_spawn_in_wave = 0
        self.enemies = []
        self._spawn_timer = 0
        self._spawn_interval = 0
        self.laser = None
        self.particles = []
        self.invincibility_timer = 0
        self.invincibility_duration = 0
        self.charge_reward_75_given = False
        self.stars = []

        # self.reset() is called by the wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Turret
        self.turret_health = 100.0
        self.turret_charge = 0.0
        self.turret_angle = -90.0 # Pointing up
        self.turret_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30)
        self.turret_rotation_speed = 2.0 # degrees per step

        # Waves & Enemies
        self.current_wave = 1
        self.enemies_per_wave = 50
        self.enemies_to_spawn_in_wave = self.enemies_per_wave
        self.enemies = []
        self._spawn_timer = 0
        self._spawn_interval = self.TARGET_FPS // 3 # Spawn 3 enemies per second

        # Laser
        self.laser = None 

        # Effects
        self.particles = []
        self.invincibility_timer = 0
        self.invincibility_duration = 3 * self.TARGET_FPS # 3 seconds

        # Reward flags
        self.charge_reward_75_given = False
        
        # Background stars
        self.stars = [
            (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT), self.np_random.integers(1, 3))
            for _ in range(100)
        ]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Update Game Logic ---
        self._handle_input(movement, space_pressed)
        
        laser_hit_count = self._update_laser()
        if laser_hit_count > 0:
            reward += laser_hit_count * 0.1
            self.turret_charge = min(100.0, self.turret_charge + laser_hit_count * 4.0)
            # // Sound: Charge gain

        damage_taken = self._update_enemies()
        if damage_taken > 0 and self.invincibility_timer <= 0:
            self.turret_health -= damage_taken
            self._create_base_hit_effect()
            # // Sound: Base hit

        self._update_particles()
        self._update_spawner()
        self._update_charge_and_invincibility()
        
        # --- Reward Calculation ---
        if self.turret_charge >= 75 and not self.charge_reward_75_given:
            reward += 1
            self.charge_reward_75_given = True

        if self.enemies_to_spawn_in_wave <= 0 and not self.enemies:
            reward += 5
            # // Sound: Wave clear
            self.current_wave += 1
            if self.current_wave > self.TOTAL_WAVES:
                self.win = True
            else:
                self.enemies_to_spawn_in_wave = self.enemies_per_wave
                self.charge_reward_75_given = False

        self.score += reward
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win:
                reward += 100
            elif self.turret_health <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Helpers ---
    def _handle_input(self, movement, space_pressed):
        if movement == 3: # Left
            self.turret_angle -= self.turret_rotation_speed
        elif movement == 4: # Right
            self.turret_angle += self.turret_rotation_speed
        self.turret_angle = max(-180, min(0, self.turret_angle))

        if space_pressed and not self.laser:
            self._fire_laser()

    def _fire_laser(self):
        # // Sound: Laser fire
        rad_angle = math.radians(self.turret_angle)
        start_pos = (
            self.turret_pos[0] + 35 * math.cos(rad_angle),
            self.turret_pos[1] + 35 * math.sin(rad_angle)
        )
        end_pos = (
            self.turret_pos[0] + 1000 * math.cos(rad_angle),
            self.turret_pos[1] + 1000 * math.sin(rad_angle)
        )
        self.laser = {
            "start": start_pos,
            "end": end_pos,
            "timer": 3, # frames
            "rad_angle": rad_angle
        }
        if self.turret_charge < 100:
            self.turret_charge = 0
            self.charge_reward_75_given = False

    def _update_laser(self):
        if not self.laser:
            return 0
        
        self.laser["timer"] -= 1
        
        hit_count = 0
        hit_enemies = []
        
        p1 = pygame.Vector2(self.laser["start"])
        laser_dir = pygame.Vector2(math.cos(self.laser["rad_angle"]), math.sin(self.laser["rad_angle"]))
        
        for enemy in self.enemies:
            p_enemy = pygame.Vector2(enemy["pos"])
            v_enemy_rel = p_enemy - p1
            
            proj_len = v_enemy_rel.dot(laser_dir)
            if proj_len > 0: # Enemy is in front of the turret
                dist_sq = v_enemy_rel.length_squared() - proj_len**2
                if dist_sq < enemy["radius"]**2:
                    hit_count += 1
                    hit_enemies.append(enemy)
                    self._create_explosion(enemy["pos"], self.COLOR_ENEMY)

        if hit_enemies:
            # // Sound: Enemy explosion
            self.enemies = [e for e in self.enemies if e not in hit_enemies]

        if self.laser["timer"] <= 0:
            self.laser = None
            
        return hit_count

    def _update_enemies(self):
        damage_taken = 0
        base_pos = pygame.Vector2(self.turret_pos)
        base_radius_sq = 40**2
        
        for enemy in self.enemies[:]:
            direction = (pygame.Vector2(self.turret_pos) - pygame.Vector2(enemy["pos"])).normalize()
            enemy["pos"] = (enemy["pos"][0] + direction.x * enemy["speed"],
                            enemy["pos"][1] + direction.y * enemy["speed"])

            if pygame.Vector2(enemy["pos"]).distance_squared_to(base_pos) < base_radius_sq:
                if self.invincibility_timer > 0:
                    self._create_explosion(enemy["pos"], self.COLOR_SHIELD)
                    # // Sound: Shield deflect
                else:
                    damage_taken += 10
                self.enemies.remove(enemy)
        
        return damage_taken

    def _update_particles(self):
        for p in self.particles[:]:
            p["timer"] -= 1
            if p["timer"] <= 0:
                self.particles.remove(p)
            else:
                p["pos"] = (p["pos"][0] + p["vel"][0], p["pos"][1] + p["vel"][1])
                p["vel"] = (p["vel"][0] * 0.98, p["vel"][1] * 0.98) # friction
                p["radius"] *= p["decay"]

    def _update_spawner(self):
        if self.enemies_to_spawn_in_wave > 0:
            self._spawn_timer += 1
            if self._spawn_timer >= self._spawn_interval:
                self._spawn_timer = 0
                self._spawn_enemy()
                self.enemies_to_spawn_in_wave -= 1

    def _spawn_enemy(self):
        wave_speed_bonus = (self.current_wave - 1) * 0.1
        speed = self.np_random.uniform(1.0 + wave_speed_bonus, 2.0 + wave_speed_bonus)
        
        self.enemies.append({
            "pos": (self.np_random.uniform(20, self.SCREEN_WIDTH - 20), -20),
            "speed": speed,
            "radius": 10
        })

    def _update_charge_and_invincibility(self):
        if self.turret_charge >= 100 and self.invincibility_timer <= 0:
            self.invincibility_timer = self.invincibility_duration
            # // Sound: Shield activate
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
            if self.invincibility_timer <= 0:
                # // Sound: Shield deactivate
                self.turret_charge = 0 # Deplete charge after invincibility ends
                self.charge_reward_75_given = False

    def _check_termination(self):
        return self.turret_health <= 0 or self.win

    # --- Effect Creators ---
    def _create_explosion(self, pos, color):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.uniform(3, 6),
                "color": color,
                "timer": self.np_random.integers(15, 30),
                "decay": 0.95
            })

    def _create_base_hit_effect(self):
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(math.pi, 2 * math.pi) # Upwards
            speed = self.np_random.uniform(2, 6)
            self.particles.append({
                "pos": list(self.turret_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "radius": self.np_random.uniform(2, 5),
                "color": self.COLOR_HEALTH_LOW,
                "timer": self.np_random.integers(20, 40),
                "decay": 0.94
            })

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, (100, 100, 120), (x, y, size, size))

    def _render_game(self):
        self._render_particles()
        if self.invincibility_timer > 0:
            self._render_shield()
        self._render_turret()
        self._render_enemies()
        if self.laser:
            self._render_laser()
        
    def _render_turret(self):
        # Base
        pygame.gfxdraw.filled_circle(self.screen, int(self.turret_pos[0]), int(self.turret_pos[1]), 30, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, int(self.turret_pos[0]), int(self.turret_pos[1]), 30, self.COLOR_BASE)

        # Charge indicator
        charge_angle = int(360 * (self.turret_charge / 100.0))
        self._render_arc(self.screen, self.COLOR_CHARGE_BAR_BG, self.turret_pos, 35, 0, 360, 5)
        if charge_angle > 0:
            self._render_arc(self.screen, self.COLOR_CHARGE_BAR_FG, self.turret_pos, 35, -90, -90 + charge_angle, 5)

        # Gun
        rad_angle = math.radians(self.turret_angle)
        gun_length = 40
        gun_width = 10
        p1 = (self.turret_pos[0] + gun_width/2 * math.cos(rad_angle + math.pi/2), self.turret_pos[1] + gun_width/2 * math.sin(rad_angle + math.pi/2))
        p2 = (self.turret_pos[0] + gun_width/2 * math.cos(rad_angle - math.pi/2), self.turret_pos[1] + gun_width/2 * math.sin(rad_angle - math.pi/2))
        p3 = (p2[0] + gun_length * math.cos(rad_angle), p2[1] + gun_length * math.sin(rad_angle))
        p4 = (p1[0] + gun_length * math.cos(rad_angle), p1[1] + gun_length * math.sin(rad_angle))
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_TURRET)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_TURRET)

    def _render_shield(self):
        progress = self.invincibility_timer / self.invincibility_duration
        alpha = 100 + 100 * math.sin(self.steps * 0.3) * progress
        radius = 45 + 5 * math.sin(self.steps * 0.3)
        
        shield_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(shield_surf, int(radius), int(radius), int(radius), (*self.COLOR_SHIELD, int(alpha/2)))
        pygame.gfxdraw.aacircle(shield_surf, int(radius), int(radius), int(radius), (*self.COLOR_SHIELD, int(alpha)))
        self.screen.blit(shield_surf, (self.turret_pos[0] - radius, self.turret_pos[1] - radius))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            radius = int(enemy["radius"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)

    def _render_laser(self):
        start = (int(self.laser["start"][0]), int(self.laser["start"][1]))
        end = (int(self.laser["end"][0]), int(self.laser["end"][1]))
        alpha = 255 * (self.laser["timer"] / 3.0)
        
        # Glow effect
        pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha/4)), start, end, 15)
        pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha/2)), start, end, 7)
        pygame.draw.line(self.screen, (220, 240, 255, alpha), start, end, 3)

    def _render_particles(self):
        for p in self.particles:
            if p["radius"] > 1:
                alpha = 255 * (p["timer"] / p.get("initial_timer", p["timer"]))
                color = (*p["color"], int(alpha))
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), color)

    def _render_ui(self):
        # Health bar
        health_color = self.COLOR_HEALTH_HIGH
        if self.turret_health < 60: health_color = self.COLOR_HEALTH_MED
        if self.turret_health < 30: health_color = self.COLOR_HEALTH_LOW
        bar_width = max(0, int(200 * (self.turret_health / 100.0)))
        pygame.draw.rect(self.screen, (40,40,40), (20, self.SCREEN_HEIGHT - 30, 200, 20))
        pygame.draw.rect(self.screen, health_color, (20, self.SCREEN_HEIGHT - 30, bar_width, 20))
        self._render_text_shadow(f"HEALTH", self.font_small, (225, self.SCREEN_HEIGHT - 21), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, x_offset=125)

        # Wave counter
        self._render_text_shadow(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", self.font_medium, (20, 10), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            self._render_text_shadow(msg, self.font_large, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 50), self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

    def _render_text_shadow(self, text, font, pos, color, shadow_color, center=False, x_offset=0):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2 + x_offset, text_rect.y + 2))
        self.screen.blit(text_surf, (text_rect.x + x_offset, text_rect.y))
        
    def _render_arc(self, surf, color, center, radius, start_angle, end_angle, width):
        for i in range(width):
            pygame.gfxdraw.arc(surf, int(center[0]), int(center[1]), radius - i, start_angle, end_angle, color)

    # --- Gymnasium Interface ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.turret_health,
            "charge": self.turret_charge,
            "wave": self.current_wave,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium interface
    # It will not be executed by the tests, but is useful for debugging.
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use Arrow Keys to rotate, Space to fire
    os.environ.setdefault("SDL_VIDEODRIVER", "x11") # or "windows", "macOS"
    pygame.display.set_caption("Neon Turret Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    truncated = False
    
    while running:
        if terminated or truncated:
            # Wait for a moment on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            truncated = False

        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
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
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.TARGET_FPS)
        
    env.close()