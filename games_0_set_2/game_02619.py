
# Generated: 2025-08-28T05:24:33.034050
# Source Brief: brief_02619.md
# Brief Index: 2619

        
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
        "Controls: ←→ to move, Shift to jump. ↑↓ to aim. Space to fire your weapon."
    )

    game_description = (
        "Control a robot in a side-scrolling action game to defeat five increasingly difficult bosses."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        self.WORLD_WIDTH = self.W * 2

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_FLOOR = (40, 30, 80)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_DMG = (255, 255, 255)
        self.COLOR_BOSS = (255, 80, 80)
        self.COLOR_BOSS_DMG = (255, 255, 255)
        self.COLOR_PLAYER_PROJ = (255, 255, 100)
        self.COLOR_ENEMY_PROJ = (255, 150, 50)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH = (50, 200, 50)
        self.COLOR_HEALTH_BG = (100, 20, 20)

        # Game constants
        self.GRAVITY = 0.5
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = 10
        self.PLAYER_SIZE = pygame.Vector2(30, 40)
        self.SHOOT_COOLDOWN = 10 # frames
        self.PLAYER_PROJ_SPEED = 12
        self.MAX_STEPS = 5000
        self.FLOOR_HEIGHT = self.H - 40

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.reward_this_step = 0
        self.robot_pos = pygame.Vector2(0, 0)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_health = 0
        self.robot_aim_state = 0
        self.is_grounded = False
        self.shoot_cooldown_timer = 0
        self.damage_timer = 0
        self.camera_x = 0.0
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.boss_level = 0
        self.boss = {}
        self.parallax_layers = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        
        # Player state
        self.robot_pos = pygame.Vector2(100, self.FLOOR_HEIGHT - self.PLAYER_SIZE.y)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_health = 100
        self.robot_aim_state = 0  # -1 down, 0 straight, 1 up
        self.is_grounded = True
        self.shoot_cooldown_timer = 0
        self.damage_timer = 0

        # World state
        self.camera_x = 0.0
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        # Boss state
        self.boss_level = 0
        self._spawn_boss()
        
        self.parallax_layers = self._create_parallax_layers()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_player()
        self._update_boss()
        self._update_projectiles()
        self._update_particles()
        self._update_camera()
        self._handle_collisions()

        terminated = self._check_termination()
        reward = self.reward_this_step
        if terminated:
            if self.victory:
                reward += 100
            else:
                reward -= 100
        
        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        if movement == 3: self.robot_vel.x = -self.PLAYER_SPEED
        elif movement == 4: self.robot_vel.x = self.PLAYER_SPEED
        else: self.robot_vel.x = 0
            
        # Aiming
        if movement == 1: self.robot_aim_state = 1
        elif movement == 2: self.robot_aim_state = -1
        elif movement not in [1, 2]: self.robot_aim_state = 0
                
        # Jumping
        if shift_held and self.is_grounded:
            self.robot_vel.y = -self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump
            for _ in range(15):
                self.particles.append(self._create_particle(
                    self.robot_pos + pygame.Vector2(self.PLAYER_SIZE.x / 2, self.PLAYER_SIZE.y),
                    (random.uniform(-1, 1), random.uniform(0, 2)), 
                    random.choice([self.COLOR_FLOOR, (60,50,100)]), 10, 20
                ))

        # Shooting
        if space_held and self.shoot_cooldown_timer <= 0:
            self._fire_player_projectile()
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            # sfx: shoot

    def _update_player(self):
        # Timers
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.damage_timer > 0: self.damage_timer -= 1

        # Physics
        self.robot_vel.y += self.GRAVITY
        self.robot_pos += self.robot_vel
        
        # Ground collision
        if self.robot_pos.y + self.PLAYER_SIZE.y >= self.FLOOR_HEIGHT:
            self.robot_pos.y = self.FLOOR_HEIGHT - self.PLAYER_SIZE.y
            self.robot_vel.y = 0
            self.is_grounded = True
        else:
            self.is_grounded = False

        # World bounds
        self.robot_pos.x = max(0, min(self.robot_pos.x, self.WORLD_WIDTH - self.PLAYER_SIZE.x))
        self.robot_pos.y = max(0, self.robot_pos.y)

    def _update_boss(self):
        if not self.boss: return
        
        # Movement (simple sine wave)
        self.boss["pos"].y = self.H / 2 - 50 + math.sin(self.steps * 0.02) * (self.H / 2 - 100)
        self.boss["pos"].x = self.WORLD_WIDTH - 200 + math.sin(self.steps * 0.01) * 50

        # Timers
        if self.boss["attack_cooldown"] > 0: self.boss["attack_cooldown"] -= 1
        if self.boss["damage_timer"] > 0: self.boss["damage_timer"] -= 1

        # Attack
        if self.boss["attack_cooldown"] <= 0:
            self.boss["attack_cooldown"] = self.boss["attack_rate"]
            self._fire_boss_projectile()
            # sfx: boss_shoot

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if self._move_projectile(p)]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if self._move_projectile(p)]

    def _move_projectile(self, p):
        p["pos"] += p["vel"]
        return 0 < p["pos"].x < self.WORLD_WIDTH and 0 < p["pos"].y < self.H

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _update_camera(self):
        target_cam_x = self.robot_pos.x - self.W / 3
        self.camera_x += (target_cam_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.W))

    def _handle_collisions(self):
        if not self.boss: return
        
        boss_rect = pygame.Rect(self.boss["pos"], self.boss["size"])
        
        # Player projectiles vs Boss
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj["pos"], (10, 10))
            if boss_rect.colliderect(proj_rect):
                self.player_projectiles.remove(proj)
                self.boss["health"] -= 10
                self.boss["damage_timer"] = 5
                self.reward_this_step += 1
                self.score += 10
                self._create_explosion(proj["pos"], 10, self.COLOR_PLAYER_PROJ)
                # sfx: hit_confirm
                if self.boss["health"] <= 0:
                    self._on_boss_defeat()
                break
        
        # Enemy projectiles vs Player
        player_rect = pygame.Rect(self.robot_pos, self.PLAYER_SIZE)
        for proj in self.enemy_projectiles[:]:
            proj_rect = pygame.Rect(proj["pos"], (12, 12))
            if player_rect.colliderect(proj_rect):
                self.enemy_projectiles.remove(proj)
                self.robot_health -= 10
                self.damage_timer = 10
                self.reward_this_step -= 0.1
                self._create_explosion(proj["pos"], 10, self.COLOR_ENEMY_PROJ)
                # sfx: player_hit
                break

    def _on_boss_defeat(self):
        self.reward_this_step += 50
        self.score += 500
        # sfx: boss_explosion
        self._create_explosion(self.boss["pos"] + self.boss["size"] / 2, 100, self.COLOR_BOSS)
        self.boss_level += 1
        if self.boss_level >= 5:
            self.victory = True
            self.boss = None
        else:
            self._spawn_boss()

    def _check_termination(self):
        if self.robot_health <= 0:
            self.game_over = True
            self._create_explosion(self.robot_pos + self.PLAYER_SIZE / 2, 50, self.COLOR_PLAYER)
        elif self.victory:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_parallax_bg()
        self._render_floor()
        if self.boss: self._render_boss()
        self._render_projectiles()
        self._render_particles()
        if self.robot_health > 0: self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_boss(self):
        base_health = 200
        base_attack_rate = 120
        health = base_health + self.boss_level * 100
        attack_rate = max(30, int(base_attack_rate * (1 - self.boss_level * 0.20)))
        
        self.boss = {
            "pos": pygame.Vector2(self.WORLD_WIDTH - 200, self.H / 2),
            "size": pygame.Vector2(80, 120),
            "health": health,
            "max_health": health,
            "attack_cooldown": attack_rate,
            "attack_rate": attack_rate,
            "damage_timer": 0,
        }

    def _fire_player_projectile(self):
        angle_rad = 0
        if self.robot_aim_state == 1: angle_rad = -math.pi / 6 # 30 deg up
        if self.robot_aim_state == -1: angle_rad = math.pi / 6 # 30 deg down
        
        vel = pygame.Vector2(self.PLAYER_PROJ_SPEED, 0).rotate_rad(angle_rad)
        start_pos = self.robot_pos + pygame.Vector2(self.PLAYER_SIZE.x, self.PLAYER_SIZE.y / 2)
        self.player_projectiles.append({"pos": start_pos, "vel": vel})

        # Muzzle flash
        for _ in range(5):
            self.particles.append(self._create_particle(
                start_pos, 
                (vel.x * 0.2 + random.uniform(-2, 2), vel.y * 0.2 + random.uniform(-2, 2)), 
                self.COLOR_PLAYER_PROJ, 5, 10
            ))

    def _fire_boss_projectile(self):
        # Aim at player's current position
        dir_to_player = (self.robot_pos - self.boss["pos"]).normalize()
        start_pos = self.boss["pos"] + self.boss["size"] / 2
        
        # Simple shot
        self.enemy_projectiles.append({
            "pos": start_pos.copy(),
            "vel": dir_to_player * 6
        })
        
        # Add spread shot for higher levels
        if self.boss_level > 1:
            self.enemy_projectiles.append({
                "pos": start_pos.copy(),
                "vel": dir_to_player.rotate(20) * 5
            })
        if self.boss_level > 2:
            self.enemy_projectiles.append({
                "pos": start_pos.copy(),
                "vel": dir_to_player.rotate(-20) * 5
            })

    def _create_particle(self, pos, vel_base, color, life_min, life_max):
        return {
            "pos": pos.copy(),
            "vel": pygame.Vector2(vel_base[0], vel_base[1]),
            "color": color,
            "life": random.randint(life_min, life_max)
        }

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 6)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(self._create_particle(pos, vel, color, 20, 40))

    def _create_parallax_layers(self):
        layers = []
        for i in range(4):
            speed = 0.1 + i * 0.15
            stars = []
            for _ in range(50 - i * 10):
                stars.append((
                    random.randint(0, self.WORLD_WIDTH),
                    random.randint(0, self.FLOOR_HEIGHT)
                ))
            layers.append({"stars": stars, "speed": speed, "size": i + 1})
        return layers

    def _render_parallax_bg(self):
        for layer in self.parallax_layers:
            color = tuple(max(0, c - 150) for c in self.COLOR_FLOOR)
            for x, y in layer["stars"]:
                screen_x = (x - self.camera_x * layer["speed"]) % self.W
                pygame.draw.circle(self.screen, color, (int(screen_x), int(y)), layer["size"])

    def _render_floor(self):
        floor_rect = pygame.Rect(0, self.FLOOR_HEIGHT, self.W, self.H - self.FLOOR_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FLOOR, floor_rect)

    def _render_player(self):
        screen_pos = self.robot_pos - pygame.Vector2(self.camera_x, 0)
        player_rect = pygame.Rect(screen_pos, self.PLAYER_SIZE)
        
        color = self.COLOR_PLAYER_DMG if self.damage_timer > 0 else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)
        
        # Jetpack/Aim indicator
        if not self.is_grounded:
            for _ in range(3):
                self.particles.append(self._create_particle(
                    self.robot_pos + pygame.Vector2(self.PLAYER_SIZE.x / 2, self.PLAYER_SIZE.y),
                    (random.uniform(-0.5, 0.5), random.uniform(1, 3)),
                    (255,180,50), 5, 15
                ))

    def _render_boss(self):
        screen_pos = self.boss["pos"] - pygame.Vector2(self.camera_x, 0)
        boss_rect = pygame.Rect(screen_pos, self.boss["size"])
        
        color = self.COLOR_BOSS_DMG if self.boss["damage_timer"] > 0 else self.COLOR_BOSS
        pygame.draw.rect(self.screen, color, boss_rect, border_radius=8)

        # Boss Health Bar
        if screen_pos.x < self.W and screen_pos.x + self.boss["size"].x > 0:
            bar_w = 100
            bar_h = 10
            bar_x = screen_pos.x + (self.boss["size"].x - bar_w) / 2
            bar_y = screen_pos.y - 20
            
            health_ratio = max(0, self.boss["health"] / self.boss["max_health"])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for p in self.player_projectiles:
            screen_pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 5, self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 5, self.COLOR_PLAYER_PROJ)
        for p in self.enemy_projectiles:
            screen_pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), 6, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y), 6, self.COLOR_ENEMY_PROJ)

    def _render_particles(self):
        for p in self.particles:
            screen_pos = p["pos"] - pygame.Vector2(self.camera_x, 0)
            alpha = int(255 * (p["life"] / p["life_max"])) if "life_max" in p and p["life_max"] > 0 else 255
            color = p["color"]
            size = max(1, int(5 * (p["life"] / 20)))
            pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y), size, color)

    def _render_ui(self):
        # Player Health Bar
        bar_w, bar_h = 200, 20
        health_ratio = max(0, self.robot_health / 100)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, bar_w * health_ratio, bar_h))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 10, 10))

        # Boss Counter
        boss_text = self.font_small.render(f"BOSS: {self.boss_level + 1} / 5", True, self.COLOR_UI_TEXT)
        if self.victory:
            boss_text = self.font_small.render(f"ALL BOSSES DEFEATED", True, self.COLOR_UI_TEXT)
        self.screen.blit(boss_text, (self.W / 2 - boss_text.get_width() / 2, 10))
        
        # Game Over / Victory Message
        if self.game_over:
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_PLAYER if self.victory else self.COLOR_BOSS
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.W / 2 - end_text.get_width() / 2, self.H / 2 - end_text.get_height() / 2))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")