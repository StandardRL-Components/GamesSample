import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:06:17.219711
# Source Brief: brief_01485.md
# Brief Index: 1485
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A tower defense game where the player deploys timed echoes from two portals
    to protect a central tower from waves of enemies.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your central tower from waves of enemies by deploying timed sonic echoes from two portals."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to rotate the aimer. "
        "Press space to fire an echo from the left portal and shift to fire from the right portal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_TOWER = (0, 255, 128)
    COLOR_TOWER_GLOW = (0, 255, 128, 50)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 60)
    COLOR_PORTAL = (255, 255, 0)
    COLOR_PORTAL_GLOW = (255, 255, 0, 80)
    COLOR_ECHO = (0, 192, 255)
    COLOR_AIMER = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (100, 40, 40)
    
    # Game Parameters
    TOWER_POS = pygame.math.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    TOWER_RADIUS = 20
    TOWER_MAX_HEALTH = 1000

    PORTAL_1_POS = pygame.math.Vector2(SCREEN_WIDTH * 0.25, SCREEN_HEIGHT // 2)
    PORTAL_2_POS = pygame.math.Vector2(SCREEN_WIDTH * 0.75, SCREEN_HEIGHT // 2)
    PORTAL_RADIUS = 12
    PORTAL_COOLDOWN = 30  # in frames

    AIMER_ORBIT_RADIUS = 40
    AIMER_RADIUS = 5
    AIMER_TURN_SPEED = 0.1  # radians per frame

    ECHO_MAX_RADIUS = 100
    ECHO_EXPAND_SPEED = 2.5
    ECHO_DAMAGE = 50

    ENEMY_RADIUS = 8
    ENEMY_BASE_HEALTH = 100
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPAWN_MARGIN = 50
    ENEMY_DAMAGE = 100

    WAVE_BASE_SIZE = 3
    WAVE_COOLDOWN_FRAMES = 120 # 4 seconds

    PARTICLE_LIFETIME = 20
    PARTICLE_SPEED = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_health = 0
        self.enemies = []
        self.echoes = []
        self.particles = []
        self.current_wave = 0
        self.wave_timer = 0
        self.enemies_to_spawn = 0
        self.base_enemy_speed = 0
        self.aimer_angle = 0
        self.portal_1_cooldown = 0
        self.portal_2_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tower_health = self.TOWER_MAX_HEALTH
        self.enemies = []
        self.echoes = []
        self.particles = []
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_COOLDOWN_FRAMES // 2
        self.enemies_to_spawn = 0
        self.base_enemy_speed = self.ENEMY_BASE_SPEED

        self.aimer_angle = 0.0
        self.portal_1_cooldown = 0
        self.portal_2_cooldown = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # --- UPDATE GAME LOGIC ---
        self._handle_input(action)
        self._update_portals()
        reward += self._update_echoes()
        reward += self._update_enemies()
        reward += self._update_wave_system()
        self._update_difficulty()
        self._update_particles()
        
        # --- CHECK TERMINATION ---
        terminated = False
        if self.tower_health <= 0:
            self.game_over = True
            terminated = True
        
        if self.current_wave > self.MAX_WAVES and not self.enemies:
            reward += 100  # Victory bonus
            self.game_over = True
            terminated = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Aimer Movement ---
        if movement == 1: self.aimer_angle -= self.AIMER_TURN_SPEED  # Up -> Counter-clockwise
        if movement == 2: self.aimer_angle += self.AIMER_TURN_SPEED  # Down -> Clockwise
        if movement == 3: self.aimer_angle -= self.AIMER_TURN_SPEED  # Left -> Counter-clockwise
        if movement == 4: self.aimer_angle += self.AIMER_TURN_SPEED  # Right -> Clockwise
        self.aimer_angle %= (2 * math.pi)

        # --- Firing Echoes (on button press) ---
        if space_held and not self.last_space_held and self.portal_1_cooldown == 0:
            self._fire_echo(1)
        if shift_held and not self.last_shift_held and self.portal_2_cooldown == 0:
            self._fire_echo(2)
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
    
    def _fire_echo(self, portal_index):
        # sfx: ECHO_FIRE
        if portal_index == 1:
            pos = self.PORTAL_1_POS
            self.portal_1_cooldown = self.PORTAL_COOLDOWN
        else:
            pos = self.PORTAL_2_POS
            self.portal_2_cooldown = self.PORTAL_COOLDOWN

        aim_offset = pygame.math.Vector2()
        aim_offset.from_polar((self.AIMER_ORBIT_RADIUS, math.degrees(self.aimer_angle)))
        
        self.echoes.append({
            "pos": pos + aim_offset,
            "radius": 0,
            "max_radius": self.ECHO_MAX_RADIUS,
            "hit_enemies": set()
        })

    def _update_portals(self):
        self.portal_1_cooldown = max(0, self.portal_1_cooldown - 1)
        self.portal_2_cooldown = max(0, self.portal_2_cooldown - 1)

    def _update_echoes(self):
        reward = 0
        new_enemies = self.enemies[:]
        enemies_removed_indices = set()

        for echo in self.echoes[:]:
            echo["radius"] += self.ECHO_EXPAND_SPEED
            if echo["radius"] > echo["max_radius"]:
                self.echoes.remove(echo)
                continue
            
            # Check for collisions with enemies
            for i, enemy in enumerate(self.enemies):
                if i not in echo["hit_enemies"]:
                    distance = echo["pos"].distance_to(enemy["pos"])
                    if distance < echo["radius"] + self.ENEMY_RADIUS:
                        # sfx: ENEMY_HIT
                        enemy["health"] -= self.ECHO_DAMAGE
                        echo["hit_enemies"].add(i)
                        reward += 0.1
                        self._create_particles(enemy["pos"], self.COLOR_ECHO, 10)
                        
                        if enemy["health"] <= 0 and i not in enemies_removed_indices:
                            # sfx: ENEMY_DESTROYED
                            reward += 1.0
                            self.score += 10
                            self._create_particles(enemy["pos"], self.COLOR_ENEMY, 20)
                            enemies_removed_indices.add(i)

        if enemies_removed_indices:
            self.enemies = [enemy for i, enemy in enumerate(self.enemies) if i not in enemies_removed_indices]
            # Adjust hit indices for subsequent echoes in the same frame is complex,
            # so we simplify by just removing dead enemies at the end of the echo update.
            # This can cause a dead enemy to be hit by multiple echoes in one frame, which is acceptable.
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy["pos"] += enemy["vel"]
            if enemy["pos"].distance_to(self.TOWER_POS) < self.TOWER_RADIUS + self.ENEMY_RADIUS:
                # sfx: TOWER_DAMAGE
                self.tower_health -= self.ENEMY_DAMAGE
                reward -= 5.0
                self.enemies.remove(enemy)
                self._create_particles(self.TOWER_POS, self.COLOR_TOWER, 30)
        return reward

    def _update_wave_system(self):
        reward = 0
        # If wave is over, start cooldown for the next one
        if not self.enemies and self.enemies_to_spawn == 0 and self.current_wave > 0 and self.current_wave <= self.MAX_WAVES:
            if self.wave_timer <= 0: # Only give reward once per wave clear
                reward += 50
                self.score += 100
                self.wave_timer = self.WAVE_COOLDOWN_FRAMES
        
        # If ready to start a new wave
        if self.wave_timer == 0 and self.enemies_to_spawn == 0:
            self.current_wave += 1
            if self.current_wave <= self.MAX_WAVES:
                self._spawn_wave()
        
        self.wave_timer = max(0, self.wave_timer - 1)
        
        # Spawn enemies if there are any in the queue
        if self.enemies_to_spawn > 0 and self.steps % 10 == 0: # Spawn one every 10 frames
            self._spawn_enemy()
            self.enemies_to_spawn -= 1
            
        return reward
    
    def _spawn_wave(self):
        wave_size = self.WAVE_BASE_SIZE + (self.current_wave -1) // 2
        self.enemies_to_spawn = wave_size

    def _spawn_enemy(self):
        side = self.np_random.integers(4)
        if side == 0: # Top
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SPAWN_MARGIN)
        elif side == 1: # Right
            pos = pygame.math.Vector2(self.SCREEN_WIDTH + self.ENEMY_SPAWN_MARGIN, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        elif side == 2: # Bottom
            pos = pygame.math.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_SPAWN_MARGIN)
        else: # Left
            pos = pygame.math.Vector2(-self.ENEMY_SPAWN_MARGIN, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        direction = (self.TOWER_POS - pos).normalize()
        speed = self.base_enemy_speed + self.np_random.uniform(-0.2, 0.2)
        vel = direction * speed

        self.enemies.append({
            "pos": pos,
            "vel": vel,
            "health": self.ENEMY_BASE_HEALTH,
        })
        
    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 500 == 0: # Slower difficulty progression
            self.base_enemy_speed = min(2.5, self.base_enemy_speed + 0.05)
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, self.PARTICLE_SPEED)
            vel = pygame.math.Vector2()
            vel.from_polar((speed, math.degrees(angle)))
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": self.np_random.integers(self.PARTICLE_LIFETIME // 2, self.PARTICLE_LIFETIME),
                "max_lifetime": self.PARTICLE_LIFETIME,
                "color": color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # friction
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "health": self.tower_health}
    
    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_portals_and_aimer()
        self._render_tower()
        for echo in self.echoes: self._render_echo(echo)
        for enemy in self.enemies: self._render_enemy(enemy)
        for particle in self.particles: self._render_particle(particle)
        self._render_ui()

    def _render_tower(self):
        # Glow
        self._render_antialiased_circle(self.TOWER_POS, self.TOWER_RADIUS + 10, self.COLOR_TOWER_GLOW)
        # Core
        pygame.gfxdraw.aacircle(self.screen, int(self.TOWER_POS.x), int(self.TOWER_POS.y), self.TOWER_RADIUS, self.COLOR_TOWER)
        pygame.gfxdraw.filled_circle(self.screen, int(self.TOWER_POS.x), int(self.TOWER_POS.y), self.TOWER_RADIUS, self.COLOR_TOWER)

    def _render_portals_and_aimer(self):
        portal_positions = [self.PORTAL_1_POS, self.PORTAL_2_POS]
        for pos in portal_positions:
            self._render_antialiased_circle(pos, self.PORTAL_RADIUS + 5, self.COLOR_PORTAL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.PORTAL_RADIUS, self.COLOR_PORTAL)
        
        # Aimer
        aim_offset = pygame.math.Vector2()
        aim_offset.from_polar((self.AIMER_ORBIT_RADIUS, math.degrees(self.aimer_angle)))
        
        # Draw aimer around both portals to show potential
        for pos in portal_positions:
            aimer_pos = pos + aim_offset
            pygame.gfxdraw.aacircle(self.screen, int(aimer_pos.x), int(aimer_pos.y), self.AIMER_RADIUS, self.COLOR_AIMER)
            
    def _render_echo(self, echo):
        alpha = int(255 * (1 - (echo["radius"] / echo["max_radius"])))
        color = (*self.COLOR_ECHO, max(0, min(255, alpha // 2))) # Fading color
        self._render_antialiased_circle(echo["pos"], int(echo["radius"]), color)

    def _render_enemy(self, enemy):
        pos_int = (int(enemy["pos"].x), int(enemy["pos"].y))
        # Glow
        self._render_antialiased_circle(enemy["pos"], self.ENEMY_RADIUS + 4, self.COLOR_ENEMY_GLOW)
        # Core
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
        
    def _render_particle(self, p):
        alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
        color = (*p["color"], max(0, min(255, alpha)))
        size = int(3 * (p["lifetime"] / p["max_lifetime"]))
        if size > 0:
            rect = pygame.Rect(int(p["pos"].x - size/2), int(p["pos"].y - size/2), size, size)
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.tower_health / self.TOWER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        
        # Wave Text
        wave_text = self.font_large.render(f"WAVE {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 5))
        
        # Score Text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 35))

        # Portal Cooldowns
        self._render_cooldown_icon(self.PORTAL_1_POS.x, self.portal_1_cooldown, self.PORTAL_COOLDOWN, "SPACE")
        self._render_cooldown_icon(self.PORTAL_2_POS.x, self.portal_2_cooldown, self.PORTAL_COOLDOWN, "SHIFT")

    def _render_cooldown_icon(self, x_center, current_cd, max_cd, key_text):
        y_pos = self.SCREEN_HEIGHT - 40
        size = 30
        rect = pygame.Rect(x_center - size/2, y_pos, size, size)
        
        # Background
        pygame.draw.rect(self.screen, (50, 50, 70), rect, border_radius=4)
        
        # Cooldown Pie
        if current_cd > 0:
            angle = -360 * (current_cd / max_cd)
            pie_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.arc(pie_surf, (0,0,0,128), pie_surf.get_rect(), math.radians(90), math.radians(90 + angle), width=size//2)
            self.screen.blit(pie_surf, rect.topleft)

        # Border and Text
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, rect, width=1, border_radius=4)
        text_surf = self.font_small.render(key_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (rect.centerx - text_surf.get_width()//2, rect.centery - text_surf.get_height()//2))

    def _render_antialiased_circle(self, pos, radius, color):
        """Renders a smooth circle with alpha transparency."""
        if radius <= 0: return
        temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pos_int = (int(pos.x), int(pos.y))
        pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
        self.screen.blit(temp_surf, (pos_int[0] - radius, pos_int[1] - radius))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # --- Manual Play Script ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Echo Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()