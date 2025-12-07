
# Generated: 2025-08-28T01:44:05.036318
# Source Brief: brief_04212.md
# Brief Index: 4212

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire your laser. Dodge enemy fire and clear the arena!"
    )

    game_description = (
        "Pilot a laser-wielding robot in a top-down neon arena, blasting through waves of enemies to achieve total robotic domination."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500 # Extended for better gameplay feel

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER_HEALTHY = (0, 255, 128)
    COLOR_PLAYER_DAMAGED = (255, 50, 50)
    COLOR_ENEMY = (0, 150, 255)
    COLOR_PLAYER_LASER = (255, 255, 100)
    COLOR_ENEMY_LASER = (255, 0, 150)
    COLOR_EXPLOSION_1 = (255, 200, 0)
    COLOR_EXPLOSION_2 = (255, 100, 0)
    COLOR_WHITE = (240, 240, 240)
    COLOR_UI_BG = (50, 50, 70, 180)
    COLOR_UI_AMMO = (255, 220, 0)
    COLOR_UI_AMMO_EMPTY = (60, 60, 60)
    
    # Player
    PLAYER_SIZE = 20
    PLAYER_SPEED = 5
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 3
    PLAYER_AMMO_RECHARGE_TIME = 1.0 # seconds
    PLAYER_FIRE_COOLDOWN = 0.2 # seconds

    # Enemy
    TOTAL_ENEMIES = 20
    ENEMY_SIZE = 18
    ENEMY_HEALTH = 25
    ENEMY_SPEED_MIN = 0.5
    ENEMY_SPEED_MAX = 1.5
    ENEMY_FIRE_RATE = 2.0 # seconds

    # Projectiles
    LASER_SPEED = 15

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
        
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.player = None
        self.enemies = []
        self.player_lasers = []
        self.enemy_lasers = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {
            "rect": pygame.Rect(self.SCREEN_WIDTH // 2 - self.PLAYER_SIZE // 2, 
                               self.SCREEN_HEIGHT // 2 - self.PLAYER_SIZE // 2, 
                               self.PLAYER_SIZE, self.PLAYER_SIZE),
            "health": self.PLAYER_MAX_HEALTH,
            "ammo": self.PLAYER_MAX_AMMO,
            "ammo_recharge_timer": 0,
            "fire_cooldown_timer": 0,
        }
        
        self.enemies = []
        for _ in range(self.TOTAL_ENEMIES):
            spawn_side = self.np_random.integers(4)
            if spawn_side == 0: # top
                x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_SIZE
            elif spawn_side == 1: # bottom
                x, y = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT
            elif spawn_side == 2: # left
                x, y = -self.ENEMY_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT)
            else: # right
                x, y = self.SCREEN_WIDTH, self.np_random.uniform(0, self.SCREEN_HEIGHT)

            self.enemies.append({
                "pos": pygame.Vector2(x, y),
                "health": self.ENEMY_HEALTH,
                "fire_timer": self.np_random.uniform(0, self.ENEMY_FIRE_RATE),
                # Circular motion params
                "center": pygame.Vector2(self.np_random.uniform(self.ENEMY_SIZE, self.SCREEN_WIDTH - self.ENEMY_SIZE),
                                         self.np_random.uniform(self.ENEMY_SIZE, self.SCREEN_HEIGHT - self.ENEMY_SIZE)),
                "radius": self.np_random.uniform(50, 150),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "angular_velocity": self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
            })
            
        self.player_lasers = []
        self.enemy_lasers = []
        self.particles = []
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        # --- Update Timers ---
        delta_time = 1 / self.FPS
        self.player["fire_cooldown_timer"] = max(0, self.player["fire_cooldown_timer"] - delta_time)
        if self.player["ammo"] < self.PLAYER_MAX_AMMO:
            self.player["ammo_recharge_timer"] += delta_time
            if self.player["ammo_recharge_timer"] >= self.PLAYER_AMMO_RECHARGE_TIME:
                self.player["ammo"] = min(self.PLAYER_MAX_AMMO, self.player["ammo"] + 1)
                self.player["ammo_recharge_timer"] = 0

        # --- Handle Player Actions ---
        self._handle_player_movement(movement)
        
        # Fire on key press (transition from not held to held)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed and self.player["ammo"] > 0 and self.player["fire_cooldown_timer"] == 0:
            self._fire_player_laser()
        self.prev_space_held = space_held

        # --- Update Game Objects ---
        self._update_enemies(delta_time)
        self._update_projectiles()
        self._update_particles(delta_time)

        # --- Collision Detection & Logic ---
        reward += self._handle_collisions()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player["health"] <= 0:
            reward -= 100
            terminated = True
            self._create_explosion(self.player["rect"].center, 50, 40)
            # sfx: player_explosion
        elif not self.enemies:
            reward += 100
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement):
        player_rect = self.player["rect"]
        if movement == 1: # Up
            player_rect.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            player_rect.y += self.PLAYER_SPEED
        elif movement == 3: # Left
            player_rect.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            player_rect.x += self.PLAYER_SPEED
        
        if movement != 0:
            self._create_trail_particle(player_rect.center)

        player_rect.clamp_ip(self.screen.get_rect())

    def _fire_player_laser(self):
        self.player["ammo"] -= 1
        self.player["fire_cooldown_timer"] = self.PLAYER_FIRE_COOLDOWN
        
        # Lasers originate from the front of the robot, requires last move direction
        # For simplicity with MultiDiscrete, we fire from the center
        start_pos = pygame.Vector2(self.player["rect"].center)
        # Aim at a fixed point far away for now; can be adapted
        target_pos = pygame.Vector2(self.player["rect"].centerx, 0)
        direction = (target_pos - start_pos).normalize()
        
        self.player_lasers.append({"pos": start_pos, "dir": direction})
        self._create_muzzle_flash(start_pos)
        # sfx: player_laser_fire

    def _update_enemies(self, delta_time):
        for enemy in self.enemies:
            # Update position (circular motion)
            enemy["angle"] += enemy["angular_velocity"]
            target_pos = pygame.Vector2(
                enemy["center"].x + math.cos(enemy["angle"]) * enemy["radius"],
                enemy["center"].y + math.sin(enemy["angle"]) * enemy["radius"]
            )
            direction_to_target = target_pos - enemy["pos"]
            if direction_to_target.length() > 1:
                enemy["pos"] += direction_to_target.normalize() * self.np_random.uniform(self.ENEMY_SPEED_MIN, self.ENEMY_SPEED_MAX)

            # Update fire timer
            enemy["fire_timer"] -= delta_time
            if enemy["fire_timer"] <= 0:
                enemy["fire_timer"] = self.ENEMY_FIRE_RATE + self.np_random.uniform(-0.5, 0.5)
                player_pos = pygame.Vector2(self.player["rect"].center)
                direction = (player_pos - enemy["pos"]).normalize()
                self.enemy_lasers.append({"pos": pygame.Vector2(enemy["pos"]), "dir": direction})
                # sfx: enemy_laser_fire

    def _update_projectiles(self):
        screen_rect = self.screen.get_rect()
        self.player_lasers = [p for p in self.player_lasers if screen_rect.collidepoint(p["pos"])]
        for laser in self.player_lasers:
            laser["pos"] += laser["dir"] * self.LASER_SPEED
            
        self.enemy_lasers = [p for p in self.enemy_lasers if screen_rect.collidepoint(p["pos"])]
        for laser in self.enemy_lasers:
            laser["pos"] += laser["dir"] * self.LASER_SPEED

    def _handle_collisions(self):
        reward = 0
        
        # Player lasers vs Enemies
        for laser in self.player_lasers[:]:
            for enemy in self.enemies[:]:
                enemy_rect = pygame.Rect(enemy["pos"].x - self.ENEMY_SIZE/2, enemy["pos"].y - self.ENEMY_SIZE/2, self.ENEMY_SIZE, self.ENEMY_SIZE)
                if enemy_rect.collidepoint(laser["pos"]):
                    enemy["health"] -= 10
                    reward += 0.1
                    self.player_lasers.remove(laser)
                    self._create_hit_spark(laser["pos"], self.COLOR_PLAYER_LASER)
                    # sfx: enemy_hit
                    if enemy["health"] <= 0:
                        reward += 10
                        self.score += 100
                        self.enemies.remove(enemy)
                        self._create_explosion(enemy["pos"], 30, 20)
                        # sfx: enemy_explosion
                    break
        
        # Enemy lasers vs Player
        player_rect = self.player["rect"]
        for laser in self.enemy_lasers[:]:
            if player_rect.collidepoint(laser["pos"]):
                self.player["health"] -= 5
                self.player["health"] = max(0, self.player["health"])
                reward -= 0.1
                self.enemy_lasers.remove(laser)
                self._create_hit_spark(laser["pos"], self.COLOR_ENEMY_LASER)
                # sfx: player_hit
                break
        
        # Enemies vs Player
        for enemy in self.enemies:
            enemy_rect = pygame.Rect(enemy["pos"].x - self.ENEMY_SIZE/2, enemy["pos"].y - self.ENEMY_SIZE/2, self.ENEMY_SIZE, self.ENEMY_SIZE)
            if player_rect.colliderect(enemy_rect):
                self.player["health"] -= 0.5 # Collision damage per frame
                self.player["health"] = max(0, self.player["health"])
                reward -= 0.02
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (drawn first)
        for p in self.particles:
            p_type = p.get("type", "explosion")
            if p_type == "explosion":
                alpha = int(255 * (p["life"] / p["max_life"]))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*p["color1"], alpha))
                if p["radius"] > 5:
                     pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"] * 0.6), (*p["color2"], alpha))
            elif p_type == "spark":
                 pygame.draw.line(self.screen, p["color"], p["pos"], p["pos"] + p["vel"] * 3, 2)
            elif p_type == "trail":
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*self.COLOR_PLAYER_HEALTHY, p["alpha"]))
        
        # Render enemies
        for enemy in self.enemies:
            pygame.gfxdraw.aacircle(self.screen, int(enemy["pos"].x), int(enemy["pos"].y), self.ENEMY_SIZE // 2, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, int(enemy["pos"].x), int(enemy["pos"].y), self.ENEMY_SIZE // 2, self.COLOR_ENEMY)

        # Render player lasers
        for laser in self.player_lasers:
            end_pos = laser["pos"] + laser["dir"] * 20
            pygame.draw.line(self.screen, self.COLOR_PLAYER_LASER, laser["pos"], end_pos, 3)
            
        # Render enemy lasers
        for laser in self.enemy_lasers:
            end_pos = laser["pos"] + laser["dir"] * 15
            pygame.draw.line(self.screen, self.COLOR_ENEMY_LASER, laser["pos"], end_pos, 2)

        # Render player
        if self.player["health"] > 0:
            health_ratio = self.player["health"] / self.PLAYER_MAX_HEALTH
            player_color = (
                int(self.COLOR_PLAYER_DAMAGED[0] * (1 - health_ratio) + self.COLOR_PLAYER_HEALTHY[0] * health_ratio),
                int(self.COLOR_PLAYER_DAMAGED[1] * (1 - health_ratio) + self.COLOR_PLAYER_HEALTHY[1] * health_ratio),
                int(self.COLOR_PLAYER_DAMAGED[2] * (1 - health_ratio) + self.COLOR_PLAYER_HEALTHY[2] * health_ratio)
            )
            pygame.draw.rect(self.screen, player_color, self.player["rect"])
            pygame.draw.rect(self.screen, self.COLOR_WHITE, self.player["rect"], 1)

    def _render_ui(self):
        # Health bar
        health_bar_width = 150
        health_bar_height = 15
        health_ratio = self.player["health"] / self.PLAYER_MAX_HEALTH
        current_health_width = int(health_bar_width * health_ratio)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_DAMAGED, (10, 10, health_bar_width, health_bar_height))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_HEALTHY, (10, 10, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (10, 10, health_bar_width, health_bar_height), 1)

        # Ammo display
        ammo_icon_size = 10
        ammo_spacing = 5
        for i in range(self.PLAYER_MAX_AMMO):
            color = self.COLOR_UI_AMMO if i < self.player["ammo"] else self.COLOR_UI_AMMO_EMPTY
            pygame.draw.rect(self.screen, color, (10 + i * (ammo_icon_size + ammo_spacing), 30, ammo_icon_size, ammo_icon_size))

        # Score
        score_text = self.font_score.render(f"{self.score:06d}", True, self.COLOR_WHITE)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 20))
        self.screen.blit(score_text, score_rect)

        # Enemy count
        enemy_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_ENEMY)
        enemy_rect = enemy_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(enemy_text, enemy_rect)
        
        if self.game_over:
            win = not self.enemies
            msg = "MISSION COMPLETE" if win else "SYSTEM FAILURE"
            color = self.COLOR_PLAYER_HEALTHY if win else self.COLOR_PLAYER_DAMAGED
            end_text = self.font_score.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "enemies_left": len(self.enemies)
        }

    # --- Particle Effects ---
    def _create_explosion(self, pos, max_radius, num_sparks):
        self.particles.append({
            "type": "explosion", "pos": pos, "radius": 0, "max_radius": max_radius,
            "life": 1.0, "max_life": 1.0, "color1": self.COLOR_EXPLOSION_1, "color2": self.COLOR_EXPLOSION_2
        })
        for _ in range(num_sparks):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "type": "spark", "pos": pygame.Vector2(pos), "vel": vel,
                "life": self.np_random.uniform(0.3, 0.8), "color": self.COLOR_WHITE
            })

    def _create_hit_spark(self, pos, color):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "type": "spark", "pos": pygame.Vector2(pos), "vel": vel,
                "life": self.np_random.uniform(0.2, 0.4), "color": color
            })

    def _create_muzzle_flash(self, pos):
        self.particles.append({
            "type": "explosion", "pos": pos, "radius": 0, "max_radius": 10,
            "life": 0.1, "max_life": 0.1, "color1": self.COLOR_WHITE, "color2": self.COLOR_PLAYER_LASER
        })
    
    def _create_trail_particle(self, pos):
        if self.np_random.random() < 0.5: # Don't spawn every frame
            self.particles.append({
                "type": "trail", "pos": pygame.Vector2(pos), "radius": self.PLAYER_SIZE / 4,
                "life": 0.3, "max_life": 0.3, "alpha": 100
            })

    def _update_particles(self, delta_time):
        for p in self.particles[:]:
            p["life"] -= delta_time
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            p_type = p.get("type", "explosion")
            if p_type == "explosion":
                p["radius"] = p["max_radius"] * (1 - (p["life"] / p["max_life"]))
            elif p_type == "spark":
                p["pos"] += p["vel"]
                p["vel"] *= 0.95 # friction
            elif p_type == "trail":
                p["radius"] *= 0.9
                p["alpha"] = int(100 * (p["life"] / p["max_life"]))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless

    env = GameEnv()
    obs, info = env.reset()
    
    # Test for a few steps with random actions
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Episode finished. Final Score: {info['score']}")
            obs, info = env.reset()
    
    env.close()
    print("Environment test run completed.")