import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:42:23.453390
# Source Brief: brief_00575.md
# Brief Index: 575
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
        "In a cyberpunk arena, tag key areas with graffiti projectiles while dodging enemies and manipulating gravity."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim. Press space to shoot and tag areas. Press shift to flip gravity."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_POS = (WIDTH // 2, HEIGHT // 2)
    PLAYER_RADIUS = 12
    PLAYER_HEALTH_MAX = 100
    
    ENEMY_SPEED_INITIAL = 1.0
    ENEMY_SPEED_INCREASE = 0.05
    ENEMY_SPAWN_RATE_INITIAL = 120 # every 120 steps
    ENEMY_RADIUS = 10
    
    PROJ_SPEED = 8.0
    PROJ_RADIUS = 5
    GRAVITY = 0.3
    
    FIRE_COOLDOWN = 10 # steps
    GRAVITY_FLIP_COOLDOWN = 60 # steps
    MAX_STEPS = 2000

    # --- COLORS (Cyberpunk Theme) ---
    COLOR_BG = (26, 26, 46) # #1a1a2e
    COLOR_GRID = (46, 46, 66)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_ENEMY = (255, 71, 87) # #ff4757
    COLOR_ENEMY_GLOW = (128, 35, 43)
    COLOR_PROJECTILE = (72, 219, 251) # #48dbfb
    COLOR_AREA_UNTAGGED = (254, 202, 87) # #feca57
    COLOR_AREA_TAGGED = (29, 209, 161) # #1dd1a1
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (46, 204, 113)
    COLOR_HEALTH_BAR_BG = (50, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = 0
        self.aim_direction = (0, -1)
        self.gravity_direction = 1
        self.fire_cooldown_timer = 0
        self.gravity_flip_timer = 0
        self.enemy_spawn_timer = 0
        self.current_enemy_speed = 0
        
        self.projectiles = []
        self.enemies = []
        self.key_areas = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.PLAYER_HEALTH_MAX
        self.aim_direction = (0, -1) # Default aim up
        self.gravity_direction = 1 # 1 for down, -1 for up
        
        self.fire_cooldown_timer = 0
        self.gravity_flip_timer = 0
        self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE_INITIAL
        self.current_enemy_speed = self.ENEMY_SPEED_INITIAL
        
        self.projectiles = []
        self.enemies = []
        self.particles = []
        
        self._setup_key_areas()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _setup_key_areas(self):
        self.key_areas = [
            {"rect": pygame.Rect(50, 50, 40, 40), "tagged": False, "pulse": 0},
            {"rect": pygame.Rect(self.WIDTH - 90, 50, 40, 40), "tagged": False, "pulse": 0},
            {"rect": pygame.Rect(50, self.HEIGHT - 90, 40, 40), "tagged": False, "pulse": 0},
            {"rect": pygame.Rect(self.WIDTH - 90, self.HEIGHT - 90, 40, 40), "tagged": False, "pulse": 0},
            {"rect": pygame.Rect(self.WIDTH // 2 - 20, 20, 40, 40), "tagged": False, "pulse": 0},
        ]
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # --- 2. Update Game Logic ---
        self.steps += 1
        self.fire_cooldown_timer = max(0, self.fire_cooldown_timer - 1)
        self.gravity_flip_timer = max(0, self.gravity_flip_timer - 1)

        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        self._update_key_areas()
        self._spawn_enemies()
        self._update_difficulty()

        # --- 3. Calculate Rewards & Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not truncated:
            if self.player_health <= 0:
                reward -= 50.0 # Death penalty
            elif all(area["tagged"] for area in self.key_areas):
                reward += 50.0 # Victory bonus
        
        self.score += reward
        
        obs = self._get_observation()
        info = self._get_info()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )

    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.aim_direction = (0, -1)  # Up
        elif movement == 2: self.aim_direction = (0, 1)   # Down
        elif movement == 3: self.aim_direction = (-1, 0)  # Left
        elif movement == 4: self.aim_direction = (1, 0)   # Right
        
        if space_held and self.fire_cooldown_timer == 0:
            # SFX: Player Shoot
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
            px, py = self.PLAYER_POS
            vx = self.aim_direction[0] * self.PROJ_SPEED
            vy = self.aim_direction[1] * self.PROJ_SPEED
            self.projectiles.append({"pos": [px, py], "vel": [vx, vy]})
            self._create_particles(self.PLAYER_POS, 5, self.COLOR_PROJECTILE)

        if shift_held and self.gravity_flip_timer == 0:
            # SFX: Gravity Flip
            self.gravity_flip_timer = self.GRAVITY_FLIP_COOLDOWN
            self.gravity_direction *= -1
            self._create_particles(self.PLAYER_POS, 20, (128, 0, 128))

    def _update_projectiles(self):
        step_reward = 0
        for proj in self.projectiles[:]:
            proj["vel"][1] += self.GRAVITY * self.gravity_direction
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]

            # Collision with enemies
            proj_rect = pygame.Rect(proj["pos"][0] - self.PROJ_RADIUS, proj["pos"][1] - self.PROJ_RADIUS, self.PROJ_RADIUS*2, self.PROJ_RADIUS*2)
            for enemy in self.enemies[:]:
                if proj_rect.colliderect(enemy["rect"]):
                    # SFX: Enemy Hit
                    step_reward += 10.0 # Reward for hitting an enemy
                    self._create_particles(proj["pos"], 10, self.COLOR_ENEMY)
                    self.enemies.remove(enemy)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    break # Projectile is destroyed
            
            # Collision with key areas
            for area in self.key_areas:
                if not area["tagged"] and area["rect"].collidepoint(proj["pos"]):
                    # SFX: Tag Success
                    area["tagged"] = True
                    area["pulse"] = 1.0
                    step_reward += 25.0 # Reward for tagging an area
                    self._create_particles(proj["pos"], 20, self.COLOR_AREA_TAGGED)
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    break
            
            # Boundary check
            if not (0 < proj["pos"][0] < self.WIDTH and 0 < proj["pos"][1] < self.HEIGHT):
                if proj in self.projectiles:
                    self.projectiles.remove(proj)
        
        return step_reward

    def _update_enemies(self):
        step_reward = 0
        player_rect = pygame.Rect(self.PLAYER_POS[0] - self.PLAYER_RADIUS, self.PLAYER_POS[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
        for enemy in self.enemies[:]:
            enemy["rect"].move_ip(enemy["vel"])
            
            if enemy["rect"].left < 0 or enemy["rect"].right > self.WIDTH:
                enemy["vel"][0] *= -1
            if enemy["rect"].top < 0 or enemy["rect"].bottom > self.HEIGHT:
                enemy["vel"][1] *= -1

            if enemy["rect"].colliderect(player_rect):
                # SFX: Player Damage
                self.player_health -= 10
                step_reward -= 5.0
                self._create_particles(self.PLAYER_POS, 30, self.COLOR_PLAYER)
                self.enemies.remove(enemy)
        return step_reward
        
    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemy_spawn_timer = self.ENEMY_SPAWN_RATE_INITIAL
            
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = [self.np_random.uniform(0, self.WIDTH), -self.ENEMY_RADIUS]
                vel = [self.np_random.uniform(-1, 1), 1]
            elif side == 1: # Bottom
                pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_RADIUS]
                vel = [self.np_random.uniform(-1, 1), -1]
            elif side == 2: # Left
                pos = [-self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
                vel = [1, self.np_random.uniform(-1, 1)]
            else: # Right
                pos = [self.WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
                vel = [-1, self.np_random.uniform(-1, 1)]

            # Normalize velocity and apply speed
            mag = math.sqrt(vel[0]**2 + vel[1]**2)
            vel = [v / mag * self.current_enemy_speed for v in vel]
            
            self.enemies.append({
                "rect": pygame.Rect(pos[0], pos[1], self.ENEMY_RADIUS*2, self.ENEMY_RADIUS*2),
                "vel": vel
            })

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.current_enemy_speed += self.ENEMY_SPEED_INCREASE
        if self.steps > 0 and self.steps % 500 == 0:
            self.ENEMY_SPAWN_RATE_INITIAL = max(30, self.ENEMY_SPAWN_RATE_INITIAL - 10)

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if all(area["tagged"] for area in self.key_areas):
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True # This condition will now be handled by truncated
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "areas_tagged": sum(1 for area in self.key_areas if area["tagged"]),
        }
        
    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        # Key Areas
        for area in self.key_areas:
            color = self.COLOR_AREA_TAGGED if area["tagged"] else self.COLOR_AREA_UNTAGGED
            if area["pulse"] > 0:
                p = area["pulse"]
                color = tuple(min(255, int(c + (255 - c) * p)) for c in color)
                pygame.draw.rect(self.screen, color, area["rect"].inflate(p*10, p*10), 2)
            pygame.draw.rect(self.screen, color, area["rect"], 0, border_radius=5)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), area["rect"], 2, border_radius=5)

        # Player
        self._draw_glowing_circle(self.PLAYER_POS, self.PLAYER_RADIUS, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Aiming Reticle
        aim_end_pos = (self.PLAYER_POS[0] + self.aim_direction[0] * 30, 
                       self.PLAYER_POS[1] + self.aim_direction[1] * 30)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, self.PLAYER_POS, aim_end_pos, 2)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            self._draw_glowing_circle(pos, self.PROJ_RADIUS, self.COLOR_PROJECTILE, self.COLOR_PROJECTILE)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["rect"].centerx), int(enemy["rect"].centery))
            self._draw_glowing_circle(pos, self.ENEMY_RADIUS, self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)

    def _draw_glowing_circle(self, pos, radius, color, glow_color):
        pos = (int(pos[0]), int(pos[1]))
        # Draw multiple layers for glow effect
        for i in range(4, 0, -1):
            alpha = 60 - i * 15
            glow_surf = pygame.Surface((radius * 2 * 2, radius * 2 * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*glow_color, alpha), (radius*2, radius*2), radius + i * 2)
            self.screen.blit(glow_surf, (pos[0] - radius*2, pos[1] - radius*2))
        
        # Draw main circle using anti-aliased drawing
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_effects(self):
        # Particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                if p in self.particles: self.particles.remove(p)
            else:
                alpha = int(255 * (p["lifespan"] / p["initial_lifespan"]))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
                self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])))

    def _update_particles(self):
        # This logic was duplicated in _render_effects, so we'll just keep it there
        # to avoid double-decrementing lifespan.
        pass
    
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos), "vel": vel, "lifespan": lifespan, 
                "initial_lifespan": lifespan, "color": color, "size": self.np_random.integers(1, 4)
            })
            
    def _update_key_areas(self):
        for area in self.key_areas:
            if area["pulse"] > 0:
                area["pulse"] = max(0, area["pulse"] - 0.05)


    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 40))

        # Tagged Areas
        tagged_count = sum(1 for area in self.key_areas if area["tagged"])
        total_areas = len(self.key_areas)
        tag_text = self.font_small.render(f"TAGS: {tagged_count}/{total_areas}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tag_text, (self.WIDTH - tag_text.get_width() - 10, 10))

        # Gravity Indicator
        arrow_points = []
        if self.gravity_direction == 1: # Down
            arrow_points = [(self.WIDTH - 20, 40), (self.WIDTH - 30, 40), (self.WIDTH - 25, 50)]
        else: # Up
            arrow_points = [(self.WIDTH - 20, 50), (self.WIDTH - 30, 50), (self.WIDTH - 25, 40)]
        
        color = (128, 0, 128)
        if self.gravity_flip_timer > self.GRAVITY_FLIP_COOLDOWN - 5: # Flash on use
             color = (255, 100, 255)
        elif self.gravity_flip_timer > 0: # Dim while on cooldown
            color = (80, 0, 80)
        
        pygame.draw.polygon(self.screen, color, arrow_points)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need to set a real video driver to see the window
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Cyberpunk Graffiti Tagger")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()