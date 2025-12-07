
# Generated: 2025-08-27T19:39:37.763882
# Source Brief: brief_02220.md
# Brief Index: 2220

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to fire your weapon. Survive for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of increasingly difficult enemies in this side-scrolling space shooter for 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_PROJECTILE = (150, 255, 255)
        self.COLOR_ENEMY_RED = (255, 80, 80)
        self.COLOR_ENEMY_BLUE = (100, 150, 255)
        self.COLOR_ENEMY_PURPLE = (200, 100, 255)
        self.COLOR_ENEMY_PROJECTILE = (255, 100, 200)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 50)]
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_BAR = (50, 50, 80)
        
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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 24)
            self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables
        self.player = {}
        self.projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.step_reward = 0

        self.player = {
            "pos": np.array([100.0, self.HEIGHT / 2.0]),
            "vel": np.array([0.0, 0.0]),
            "radius": 12,
            "hp": 3,
            "max_hp": 3,
            "shoot_cooldown": 0,
            "invincibility_timer": 0,
        }
        self.player_shoot_cooldown_max = 8 # frames
        self.prev_space_held = False
        self.screen_shake = 0

        self.projectiles.clear()
        self.enemies.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        # Difficulty settings
        self.enemy_spawn_timer = 0
        self.base_enemy_spawn_interval = self.FPS * 2
        self.enemy_spawn_interval = self.base_enemy_spawn_interval
        self.base_enemy_projectile_speed = 4.0
        self.enemy_projectile_speed = self.base_enemy_projectile_speed

        # Background stars
        if not self.stars:
            for _ in range(150):
                self.stars.append({
                    "pos": np.array([self.np_random.random() * self.WIDTH, self.np_random.random() * self.HEIGHT]),
                    "speed": 1 + self.np_random.random() * 2,
                    "size": int(self.np_random.random() * 2) + 1,
                    "color": random.choice([(100,100,100), (150,150,150), (200,200,200)])
                })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic
        self._handle_input(movement, space_held)
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        self._handle_collisions()
        self._spawn_enemies()
        self._update_difficulty()
        
        self.steps += 1
        
        # Calculate rewards
        self.step_reward += 0.01  # Small survival reward per frame
        terminated = self._check_termination()

        if terminated and self.game_won:
            self.step_reward += 50
        elif terminated and self.game_over:
            self.step_reward -= 100

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        accel = 1.2
        if movement == 1: self.player["vel"][1] -= accel # Up
        if movement == 2: self.player["vel"][1] += accel # Down
        if movement == 3: self.player["vel"][0] -= accel # Left
        if movement == 4: self.player["vel"][0] += accel # Right

        # Shooting
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        
        if space_held and not self.prev_space_held and self.player["shoot_cooldown"] == 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                "pos": self.player["pos"].copy() + np.array([self.player["radius"], 0]),
                "vel": np.array([15.0, 0.0]),
                "radius": 4,
            })
            self.player["shoot_cooldown"] = self.player_shoot_cooldown_max
        
        self.prev_space_held = space_held

    def _update_player(self):
        # Apply friction/damping
        self.player["vel"] *= 0.9

        # Update position
        self.player["pos"] += self.player["vel"]

        # Clamp position to screen bounds
        self.player["pos"][0] = np.clip(self.player["pos"][0], self.player["radius"], self.WIDTH - self.player["radius"])
        self.player["pos"][1] = np.clip(self.player["pos"][1], self.player["radius"], self.HEIGHT - self.player["radius"])

        # Invincibility frames
        if self.player["invincibility_timer"] > 0:
            self.player["invincibility_timer"] -= 1

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            if enemy["type"] == "red":
                enemy["pos"] += enemy["vel"]
            elif enemy["type"] == "blue":
                enemy["pos"][0] += enemy["vel"][0]
                enemy["pos"][1] = enemy["spawn_y"] + math.sin(self.steps * 0.05) * 50
            elif enemy["type"] == "purple":
                enemy["pos"] += enemy["vel"]
                if enemy["pos"][1] <= enemy["radius"] or enemy["pos"][1] >= self.HEIGHT - enemy["radius"]:
                    enemy["vel"][1] *= -1

            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                direction = self.player["pos"] - enemy["pos"]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                
                self.enemy_projectiles.append({
                    "pos": enemy["pos"].copy(),
                    "vel": direction * self.enemy_projectile_speed,
                    "radius": 5,
                })
                enemy["shoot_cooldown"] = self.np_random.integers(90, 150)
        
        # Remove off-screen enemies
        self.enemies = [e for e in self.enemies if e["pos"][0] > -e["radius"]]

    def _update_projectiles(self):
        # Player projectiles
        for p in self.projectiles:
            p["pos"] += p["vel"]
        self.projectiles = [p for p in self.projectiles if 0 < p["pos"][0] < self.WIDTH]

        # Enemy projectiles
        for p in self.enemy_projectiles:
            p["pos"] += p["vel"]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p["pos"][0] < self.WIDTH and 0 < p["pos"][1] < self.HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _handle_collisions(self):
        # Player projectiles vs enemies
        for p in self.projectiles[:]:
            for e in self.enemies[:]:
                if np.linalg.norm(p["pos"] - e["pos"]) < p["radius"] + e["radius"]:
                    self.projectiles.remove(p)
                    e["hp"] -= 1
                    e["hit_timer"] = 5 # Flash effect
                    if e["hp"] <= 0:
                        # sfx: explosion.wav
                        self._create_explosion(e["pos"], e["radius"])
                        self.enemies.remove(e)
                        reward_map = {"red": 1, "blue": 2, "purple": 3}
                        self.step_reward += reward_map[e["type"]]
                        self.score += reward_map[e["type"]] * 10
                    break
        
        # Player vs enemy projectiles
        if self.player["invincibility_timer"] == 0:
            for p in self.enemy_projectiles[:]:
                if np.linalg.norm(p["pos"] - self.player["pos"]) < p["radius"] + self.player["radius"]:
                    self.enemy_projectiles.remove(p)
                    self._player_hit()
                    break
        
        # Player vs enemies
        if self.player["invincibility_timer"] == 0:
            for e in self.enemies[:]:
                if np.linalg.norm(e["pos"] - self.player["pos"]) < e["radius"] + self.player["radius"]:
                    self._create_explosion(e["pos"], e["radius"])
                    self.enemies.remove(e)
                    self._player_hit()
                    break

    def _player_hit(self):
        # sfx: player_hit.wav
        self.player["hp"] -= 1
        self.player["invincibility_timer"] = self.FPS * 2 # 2 seconds of invincibility
        self.screen_shake = 10
        if self.player["hp"] <= 0:
            self.game_over = True
            self._create_explosion(self.player["pos"], self.player["radius"] * 2)

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            enemy_type = self.np_random.choice(["red", "blue", "purple"], p=[0.5, 0.3, 0.2])
            spawn_y = self.np_random.random() * (self.HEIGHT - 80) + 40
            
            if enemy_type == "red":
                self.enemies.append({
                    "pos": np.array([self.WIDTH + 20.0, spawn_y]),
                    "vel": np.array([-3.0, 0.0]),
                    "radius": 10, "hp": 1, "max_hp": 1, "type": "red",
                    "shoot_cooldown": self.np_random.integers(60, 120), "hit_timer": 0
                })
            elif enemy_type == "blue":
                self.enemies.append({
                    "pos": np.array([self.WIDTH + 20.0, spawn_y]),
                    "vel": np.array([-2.0, 0.0]),
                    "radius": 13, "hp": 2, "max_hp": 2, "type": "blue",
                    "spawn_y": spawn_y, "shoot_cooldown": self.np_random.integers(45, 90), "hit_timer": 0
                })
            elif enemy_type == "purple":
                self.enemies.append({
                    "pos": np.array([self.WIDTH + 20.0, spawn_y]),
                    "vel": np.array([-2.5, 2.5 * self.np_random.choice([-1, 1])]),
                    "radius": 15, "hp": 3, "max_hp": 3, "type": "purple",
                    "shoot_cooldown": self.np_random.integers(30, 75), "hit_timer": 0
                })
            
            self.enemy_spawn_timer = self.enemy_spawn_interval

    def _update_difficulty(self):
        difficulty_tier = self.steps // (self.FPS * 10)
        self.enemy_spawn_interval = self.base_enemy_spawn_interval - difficulty_tier * 10
        self.enemy_spawn_interval = max(15, self.enemy_spawn_interval)
        self.enemy_projectile_speed = self.base_enemy_projectile_speed + difficulty_tier * 0.25

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_won = True
            return True
        return False

    def _create_explosion(self, pos, size):
        num_particles = int(size * 2)
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 4
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy() + (self.np_random.random(2) - 0.5) * size,
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        render_surface = self.screen
        if self.screen_shake > 0:
            render_surface = self.screen.copy()
            offset_x = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            offset_y = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT))
            temp_surf.fill(self.COLOR_BG)
            self._render_game(temp_surf)
            render_surface.blit(temp_surf, (offset_x, offset_y))
            self.screen_shake = int(self.screen_shake * 0.9)
        else:
            self.screen.fill(self.COLOR_BG)
            self._render_game(self.screen)

        self._render_ui(render_surface)
        
        arr = pygame.surfarray.array3d(render_surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self, surface):
        # Background Stars
        for star in self.stars:
            star["pos"][0] -= star["speed"]
            if star["pos"][0] < 0:
                star["pos"][0] = self.WIDTH
                star["pos"][1] = self.np_random.random() * self.HEIGHT
            pygame.draw.circle(surface, star["color"], star["pos"].astype(int), star["size"])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"]
            pygame.gfxdraw.filled_circle(surface, int(p["pos"][0]), int(p["pos"][1]), int(p["life"]/5), (*color, alpha))

        # Enemy Projectiles
        for p in self.enemy_projectiles:
            pygame.draw.circle(surface, self.COLOR_ENEMY_PROJECTILE, p["pos"].astype(int), p["radius"])
            pygame.draw.circle(surface, (255,255,255), p["pos"].astype(int), p["radius"]//2)

        # Player Projectiles
        for p in self.projectiles:
            start_pos = p["pos"]
            end_pos = p["pos"] - p["vel"] * 0.5
            pygame.draw.line(surface, self.COLOR_PLAYER_PROJECTILE, start_pos.astype(int), end_pos.astype(int), 3)

        # Enemies
        for e in self.enemies:
            color_map = {"red": self.COLOR_ENEMY_RED, "blue": self.COLOR_ENEMY_BLUE, "purple": self.COLOR_ENEMY_PURPLE}
            color = color_map[e["type"]]
            if e["hit_timer"] > 0:
                color = (255, 255, 255)
                e["hit_timer"] -= 1
            pos = e["pos"].astype(int)
            radius = e["radius"]
            pygame.draw.circle(surface, color, pos, radius)
            pygame.draw.circle(surface, (0,0,0), pos, radius - 3)

        # Player
        if not self.game_over:
            pos = self.player["pos"].astype(int)
            radius = self.player["radius"]
            is_invincible = self.player["invincibility_timer"] > 0
            
            # Blinking effect when invincible
            if not is_invincible or self.steps % 10 < 5:
                # Engine trail
                for i in range(5):
                    p_pos = self.player["pos"] - np.array([radius * (i * 0.5), 0]) + (self.np_random.random(2) - 0.5) * 4
                    p_size = max(1, 4 - i)
                    p_color = random.choice([(255,200,0), (255,100,0)])
                    pygame.draw.circle(surface, p_color, p_pos.astype(int), p_size)

                # Ship body
                p1 = (pos[0] + radius, pos[1])
                p2 = (pos[0] - radius, pos[1] - radius * 0.8)
                p3 = (pos[0] - radius, pos[1] + radius * 0.8)
                pygame.gfxdraw.aapolygon(surface, [p1, p2, p3], self.COLOR_PLAYER)
                pygame.gfxdraw.filled_polygon(surface, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self, surface):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (10, 10))

        # Timer
        time_left = self.GAME_DURATION_SECONDS - (self.steps / self.FPS)
        time_text = self.font_large.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        surface.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Health
        if not self.game_over:
            hp_bar_width = 100
            hp_bar_height = 15
            hp_bar_x = 10
            hp_bar_y = self.HEIGHT - hp_bar_height - 10
            
            health_ratio = self.player["hp"] / self.player["max_hp"]
            
            pygame.draw.rect(surface, self.COLOR_UI_BAR, (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
            pygame.draw.rect(surface, self.COLOR_PLAYER, (hp_bar_x, hp_bar_y, int(hp_bar_width * health_ratio), hp_bar_height))
            pygame.draw.rect(surface, self.COLOR_TEXT, (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height), 1)

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
        elif self.game_won:
            msg = "YOU SURVIVED!"
        else:
            msg = None
            
        if msg:
            text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            surface.blit(text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_hp": self.player["hp"],
            "time_remaining": self.GAME_DURATION_SECONDS - (self.steps / self.FPS),
        }

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    print(env.user_guide)

    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Frame rate control ---
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()