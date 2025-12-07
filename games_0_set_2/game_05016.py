import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to jump. ↑ for up, ← for left, → for right. "
        "Hold Shift for a longer jump. Press Space to fire your weapon. Use ↓ to drop through platforms."
    )

    game_description = (
        "Navigate a hopping spaceship upwards through perilous platforms, "
        "dodging and shooting enemies to reach the top."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_THRUSTER = (255, 180, 50)
        self.COLOR_PLATFORM = (100, 150, 255)
        self.COLOR_PLATFORM_TOP = (150, 200, 255)
        self.COLOR_GOAL_PLATFORM = (255, 223, 0)
        self.COLOR_BULLET = (255, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0)
        self.ENEMY_COLORS = {
            1: (255, 80, 80),   # Horizontal
            2: (255, 165, 0),  # Vertical
            3: (200, 80, 255),  # Diagonal
            4: (255, 255, 0),  # Circular
            5: (0, 255, 255),   # Teleport
        }

        # Physics & Gameplay
        self.GRAVITY = 0.4
        self.JUMP_VELOCITY = -9.0
        self.MOVE_VELOCITY = 4.0
        self.LONG_JUMP_MOD = 1.8
        self.FRICTION = 0.95
        self.BULLET_SPEED = 12
        self.FIRE_COOLDOWN_FRAMES = 8
        self.NUM_PLATFORMS = 60
        self.PLATFORM_MIN_Y_SPACING = 70
        self.PLATFORM_MAX_Y_SPACING = 100

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # RNG
        self.np_random = None

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.platforms = []
        self.enemies = []
        self.bullets = []
        self.particles = []
        self.stars = []
        self.camera_y = 0
        self.highest_platform_idx = 0
        self.fire_cooldown = 0
        self.enemy_spawn_rate = 0.01
        self.enemy_speed_mod = 1.0

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_platforms()
        
        start_platform = self.platforms[0]
        self.player = {
            "rect": pygame.Rect(start_platform.centerx - 10, start_platform.top - 20, 20, 20),
            "vx": 0,
            "vy": 0,
            "on_platform": True,
            "can_jump": True,
        }

        self.enemies = []
        self.bullets = []
        self.particles = []
        self._generate_stars(200)
        
        self.camera_y = 0
        self.highest_platform_idx = 0
        self.fire_cooldown = 0
        self.enemy_spawn_rate = 0.01
        self.enemy_speed_mod = 1.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Survival reward

        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_bullets()
        self._update_enemies()
        self._update_particles()
        
        reward += self._check_collisions()
        self._update_camera()
        self._spawn_enemies()
        self._update_difficulty()
        
        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if truncated and not self.game_over:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Fire weapon
        if space_held and self.fire_cooldown == 0:
            bullet_rect = pygame.Rect(self.player["rect"].centerx - 2, self.player["rect"].top, 4, 10)
            self.bullets.append(bullet_rect)
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
            self._create_particles(self.player["rect"].midbottom, 5, self.COLOR_PLAYER_THRUSTER, 1, 3, 0, 360, -2, 2)

        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        # Jumping logic
        if self.player["can_jump"] and movement in [1, 2, 3, 4]:
            self.player["can_jump"] = False
            self.player["on_platform"] = False
            self.player["vy"] = self.JUMP_VELOCITY
            
            self._create_particles(self.player["rect"].midbottom, 15, self.COLOR_PLATFORM_TOP, 2, 5, 180, 360, 2, 5)

            jump_mod = self.LONG_JUMP_MOD if shift_held else 1.0
            
            if movement == 1: # Up
                self.player["vx"] = 0
            elif movement == 3: # Left
                self.player["vx"] = -self.MOVE_VELOCITY * jump_mod
            elif movement == 4: # Right
                self.player["vx"] = self.MOVE_VELOCITY * jump_mod
            elif movement == 2: # Down (drop)
                self.player["vy"] = 2.0 # just a small push to detach
                self.player["rect"].y += 2

    def _update_player(self):
        # Apply gravity if not on a platform
        if not self.player["on_platform"]:
            self.player["vy"] += self.GRAVITY

        # Apply friction
        self.player["vx"] *= self.FRICTION

        # Update position
        self.player["rect"].x += int(self.player["vx"])
        self.player["rect"].y += int(self.player["vy"])

        # Screen bounds
        if self.player["rect"].left < 0:
            self.player["rect"].left = 0
            self.player["vx"] = 0
        if self.player["rect"].right > self.WIDTH:
            self.player["rect"].right = self.WIDTH
            self.player["vx"] = 0

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet.y -= self.BULLET_SPEED
            if bullet.bottom < self.camera_y:
                self.bullets.remove(bullet)

    def _update_enemies(self):
        for enemy in self.enemies:
            speed = enemy["speed"] * self.enemy_speed_mod
            if enemy["type"] == 1: # Horizontal
                enemy["rect"].x += speed * enemy["dir"]
                if enemy["rect"].left < 0 or enemy["rect"].right > self.WIDTH:
                    enemy["dir"] *= -1
            elif enemy["type"] == 2: # Vertical
                enemy["rect"].y += speed * enemy["dir"]
                if enemy["rect"].top < enemy["bounds"][0] or enemy["rect"].bottom > enemy["bounds"][1]:
                    enemy["dir"] *= -1
            elif enemy["type"] == 3: # Diagonal
                enemy["rect"].x += speed * enemy["dir_x"]
                enemy["rect"].y += speed * enemy["dir_y"]
                if enemy["rect"].left < 0 or enemy["rect"].right > self.WIDTH:
                    enemy["dir_x"] *= -1
                if enemy["rect"].top < self.camera_y or enemy["rect"].bottom > self.camera_y + self.HEIGHT + 100:
                    enemy["dir_y"] *= -1
            elif enemy["type"] == 4: # Circular
                enemy["angle"] += speed * 0.05
                enemy["rect"].centerx = enemy["center"][0] + math.cos(enemy["angle"]) * enemy["radius"]
                enemy["rect"].centery = enemy["center"][1] + math.sin(enemy["angle"]) * enemy["radius"]
            elif enemy["type"] == 5: # Teleport
                enemy["timer"] -= 1
                if enemy["timer"] <= 0:
                    enemy["timer"] = enemy["interval"]
                    enemy["rect"].center = (
                        self.np_random.integers(50, self.WIDTH - 50),
                        self.camera_y + self.np_random.integers(50, self.HEIGHT - 50)
                    )
                    enemy["pre_teleport_timer"] = 10
                if enemy["pre_teleport_timer"] > 0:
                    enemy["pre_teleport_timer"] -=1
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_camera(self):
        # Camera only scrolls up, keeping player in the lower 2/3 of the screen
        target_cam_y = self.player["rect"].centery - self.HEIGHT * 0.6
        if target_cam_y < self.camera_y:
            self.camera_y = (self.camera_y * 19 + target_cam_y) / 20 # Smooth follow
        # Never scroll down past the start
        if self.camera_y > 0:
            self.camera_y = 0

    def _update_difficulty(self):
        # Increase enemy speed
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_speed_mod += 0.05
        # Increase enemy spawn rate
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_spawn_rate *= 1.01

    def _check_collisions(self):
        reward = 0

        # Player -> Platforms
        self.player["on_platform"] = False
        if self.player["vy"] >= 0:
            for i, plat in enumerate(self.platforms):
                if self.player["rect"].colliderect(plat) and \
                   self.player["rect"].bottom <= plat.top + self.player["vy"] + 1:
                    
                    self.player["rect"].bottom = plat.top
                    self.player["vy"] = 0
                    self.player["on_platform"] = True
                    if not self.player["can_jump"]:
                        self._create_particles(self.player["rect"].midbottom, 10, self.COLOR_PLATFORM_TOP, 1, 3, 180, 360, 1, 3)
                        self.player["can_jump"] = True

                    if i > self.highest_platform_idx:
                        reward += 5 * (i - self.highest_platform_idx)
                        self.score += 10 * (i - self.highest_platform_idx)
                        self.highest_platform_idx = i

                    if i == self.NUM_PLATFORMS - 1: # Reached Goal
                        reward = 100
                        self.score += 1000
                        self.game_over = True
                    break
        
        # Player -> Enemies
        for enemy in self.enemies:
            if self.player["rect"].colliderect(enemy["rect"]):
                reward = -100
                self.game_over = True
                self._create_particles(self.player["rect"].center, 50, self.COLOR_PLAYER, 2, 8, 0, 360, 2, 8)
                break
        
        # Bullets -> Enemies
        for bullet in self.bullets[:]:
            for enemy in self.enemies[:]:
                if bullet.colliderect(enemy["rect"]):
                    self.bullets.remove(bullet)
                    self.enemies.remove(enemy)
                    reward += 10
                    self.score += 50
                    self._create_particles(enemy["rect"].center, 30, self.ENEMY_COLORS[enemy["type"]], 1, 5, 0, 360, 1, 6)
                    break
            else:
                continue
            break

        # Player -> Fall off screen
        if self.player["rect"].top > self.camera_y + self.HEIGHT:
            reward = -100
            self.game_over = True
            
        return reward

    def _spawn_enemies(self):
        if self.np_random.random() < self.enemy_spawn_rate and len(self.enemies) < 15:
            enemy_type = self.np_random.integers(1, 6)
            x = self.np_random.integers(20, self.WIDTH - 20)
            y = self.camera_y - 50
            
            if enemy_type == 1: # Horizontal
                rect = pygame.Rect(x, y, 25, 25)
                self.enemies.append({"rect": rect, "type": 1, "speed": 2, "dir": self.np_random.choice([-1, 1])})
            elif enemy_type == 2: # Vertical
                rect = pygame.Rect(x, y, 30, 15)
                bounds = (y - 50, y + 50)
                self.enemies.append({"rect": rect, "type": 2, "speed": 1.5, "dir": 1, "bounds": bounds})
            elif enemy_type == 3: # Diagonal
                rect = pygame.Rect(x, y, 20, 20)
                self.enemies.append({"rect": rect, "type": 3, "speed": 1.8, "dir_x": self.np_random.choice([-1, 1]), "dir_y": 1})
            elif enemy_type == 4: # Circular
                radius = self.np_random.integers(30, 71)
                rect = pygame.Rect(x, y, 22, 22)
                self.enemies.append({"rect": rect, "type": 4, "speed": 1.5, "center": (x, y + radius), "radius": radius, "angle": 0})
            elif enemy_type == 5: # Teleport
                interval = self.np_random.integers(60, 121)
                rect = pygame.Rect(x, y, 25, 25)
                self.enemies.append({"rect": rect, "type": 5, "speed": 0, "interval": interval, "timer": interval, "pre_teleport_timer": 0})

    def _generate_platforms(self):
        self.platforms = []
        plat_y = self.HEIGHT - 40
        # Start platform
        self.platforms.append(pygame.Rect(0, plat_y, self.WIDTH, 40))

        for i in range(1, self.NUM_PLATFORMS):
            plat_y -= self.np_random.integers(self.PLATFORM_MIN_Y_SPACING, self.PLATFORM_MAX_Y_SPACING + 1)
            width = self.np_random.integers(80, 151)
            x = self.np_random.integers(0, self.WIDTH - width + 1)
            
            prev_plat = self.platforms[-1]
            max_reach = self.MOVE_VELOCITY * self.LONG_JUMP_MOD * (abs(self.JUMP_VELOCITY) / self.GRAVITY) * 0.8
            x = max(x, prev_plat.centerx - max_reach)
            x = min(x, prev_plat.centerx + max_reach - width)
            x = max(0, min(self.WIDTH - width, x))

            self.platforms.append(pygame.Rect(int(x), int(plat_y), int(width), 20))
        
        goal_y = self.platforms[-1].y - 100
        self.platforms[-1] = pygame.Rect(0, goal_y, self.WIDTH, 40)

    def _generate_stars(self, n):
        self.stars = []
        for _ in range(n):
            self.stars.append({
                "x": self.np_random.integers(0, self.WIDTH + 1),
                "y": self.np_random.integers(0, self.HEIGHT + 1),
                "speed": self.np_random.uniform(0.1, 0.5),
                "size": self.np_random.integers(1, 3)
            })

    def _create_particles(self, pos, count, color, min_size, max_size, min_angle, max_angle, min_speed, max_speed):
        for _ in range(count):
            angle = math.radians(self.np_random.uniform(min_angle, max_angle))
            speed = self.np_random.uniform(min_speed, max_speed)
            self.particles.append({
                "x": pos[0], "y": pos[1],
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(10, 26),
                "color": color,
                "size": self.np_random.integers(min_size, max_size + 1)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.highest_platform_idx}

    def render(self):
        return self._get_observation()

    def _render_game(self):
        cam_y = int(self.camera_y)

        # Stars (parallax)
        for star in self.stars:
            star_y = (star["y"] - cam_y * star["speed"]) % self.HEIGHT
            pygame.draw.circle(self.screen, (200, 200, 200), (star["x"], star_y), star["size"])

        # Platforms
        for i, plat in enumerate(self.platforms):
            color = self.COLOR_GOAL_PLATFORM if i == self.NUM_PLATFORMS - 1 else self.COLOR_PLATFORM
            top_color = self.COLOR_GOAL_PLATFORM if i == self.NUM_PLATFORMS - 1 else self.COLOR_PLATFORM_TOP
            
            p_rect = plat.move(0, -cam_y)
            if p_rect.bottom < 0 or p_rect.top > self.HEIGHT:
                continue

            pygame.draw.rect(self.screen, color, p_rect)
            pygame.draw.rect(self.screen, top_color, (p_rect.x, p_rect.y, p_rect.width, 4))
            pygame.gfxdraw.rectangle(self.screen, p_rect, (0,0,0,50))


        # Bullets
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, bullet.move(0, -cam_y))

        # Enemies
        for enemy in self.enemies:
            e_rect = enemy["rect"].move(0, -cam_y)
            color = self.ENEMY_COLORS[enemy["type"]]
            if enemy["type"] == 5 and enemy["pre_teleport_timer"] > 0:
                alpha = int(255 * (enemy["pre_teleport_timer"] / 10))
                s = pygame.Surface(e_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (*color, alpha), (0, 0, *e_rect.size))
                self.screen.blit(s, e_rect.topleft)
            else:
                pygame.draw.rect(self.screen, color, e_rect)
                pygame.gfxdraw.rectangle(self.screen, e_rect, (0,0,0,80))

        # Player
        if not self.game_over:
            p_rect = self.player["rect"].move(0, -cam_y)
            # Thruster particles when jumping
            if self.player["vy"] < 0:
                self._create_particles(p_rect.midbottom, 2, self.COLOR_PLAYER_THRUSTER, 1, 3, 70, 110, 2, 4)
            
            # Main player shape
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect)
            # Glow effect
            glow_rect = p_rect.inflate(8, 8)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_PLAYER, 30), (0,0,*glow_rect.size), border_radius=4)
            self.screen.blit(s, glow_rect.topleft)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 25.0))
            color = (*p["color"], alpha)
            pos = (int(p["x"]), int(p["y"] - cam_y))
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (pos[0] - p["size"], pos[1] - p["size"]))


    def _render_ui(self):
        score_text = f"SCORE: {self.score}"
        height_text = f"HEIGHT: {self.highest_platform_idx}/{self.NUM_PLATFORMS - 1}"

        self._draw_text(score_text, (10, 10), self.font_large)
        self._draw_text(height_text, (self.WIDTH - 10, 10), self.font_large, align="right")
        
        if self.game_over:
            msg = "GOAL REACHED!" if self.highest_platform_idx == self.NUM_PLATFORMS - 1 else "GAME OVER"
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, align="center")

    def _draw_text(self, text, pos, font, color=None, shadow_color=None, align="left"):
        if color is None: color = self.COLOR_TEXT
        if shadow_color is None: shadow_color = self.COLOR_SHADOW
        
        shadow_surf = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)

        if align == "left":
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            text_pos = pos
        elif align == "right":
            shadow_pos = (pos[0] - shadow_surf.get_width() + 2, pos[1] + 2)
            text_pos = (pos[0] - text_surf.get_width(), pos[1])
        elif align == "center":
            shadow_pos = (pos[0] - shadow_surf.get_width() // 2 + 2, pos[1] + 2)
            text_pos = (pos[0] - text_surf.get_width() // 2, pos[1])
        
        self.screen.blit(shadow_surf, shadow_pos)
        self.screen.blit(text_surf, text_pos)

    def close(self):
        pygame.quit()