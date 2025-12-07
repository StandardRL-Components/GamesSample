
# Generated: 2025-08-27T22:01:21.527045
# Source Brief: brief_02986.md
# Brief Index: 2986

        
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
        "Controls: ←→ to move, ↑ to jump (up to 3 times). Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a heavily armed robot through a side-scrolling cityscape, blasting enemies and leaping over obstacles to reach the exit."
    )

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.LEVEL_WIDTH = 3200
        self.GROUND_HEIGHT = 60

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GROUND = (40, 42, 58)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 30)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_ENEMY_GLOW = (255, 80, 80, 40)
        self.COLOR_PLAYER_PROJ = (137, 221, 255)
        self.COLOR_ENEMY_PROJ = (255, 180, 80)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (200, 50, 50)]
        self.COLOR_EXIT = (220, 120, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (50, 200, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 50, 50)
        
        # Physics and game constants
        self.GRAVITY = 0.4
        self.MAX_STEPS = 5000
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 4.0
        self.PLAYER_JUMP_STRENGTH = -9.0
        self.PLAYER_MAX_JUMPS = 3
        self.PLAYER_SHOOT_COOLDOWN = 10 # frames
        self.PLAYER_INVULNERABILITY_FRAMES = 60
        self.ENEMY_COUNT = 5
        self.ENEMY_HEALTH = 1
        self.ENEMY_SPEED = 1.0
        self.ENEMY_SHOOT_COOLDOWN = 90 # frames
        self.BASE_ENEMY_PROJ_SPEED = 4.0

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.camera_x = 0
        self.background_layers = []
        self.last_player_x = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        
        # Player state
        self.player = {
            "x": 100, "y": self.SCREEN_HEIGHT - self.GROUND_HEIGHT - 50,
            "vx": 0, "vy": 0,
            "width": 30, "height": 50,
            "health": self.PLAYER_MAX_HEALTH,
            "on_ground": False,
            "jumps_left": self.PLAYER_MAX_JUMPS,
            "shoot_cooldown": 0,
            "invuln_timer": 0,
            "facing": 1 # 1 for right, -1 for left
        }
        self.last_player_x = self.player["x"]

        # Enemies
        self.enemies = []
        for i in range(self.ENEMY_COUNT):
            self._spawn_enemy(i)
        
        # Other lists
        self.projectiles = []
        self.particles = []

        # Parallax background
        self.background_layers = [
            {"depth": 0.2, "color": (20, 22, 40), "y_offset": 50, "height": 150},
            {"depth": 0.4, "color": (25, 28, 48), "y_offset": 100, "height": 100},
            {"depth": 0.6, "color": (30, 35, 55), "y_offset": 150, "height": 50},
        ]
        
        return self._get_observation(), self._get_info()

    def _spawn_enemy(self, index):
        # Spread enemies across the level
        x_pos = 600 + index * ((self.LEVEL_WIDTH - 800) / self.ENEMY_COUNT) + self.np_random.integers(-100, 100)
        y_pos = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - 40
        patrol_range = 100
        enemy = {
            "x": x_pos, "y": y_pos,
            "width": 35, "height": 40,
            "health": self.ENEMY_HEALTH,
            "patrol_start": x_pos - patrol_range,
            "patrol_end": x_pos + patrol_range,
            "direction": 1,
            "shoot_cooldown": self.np_random.integers(0, self.ENEMY_SHOOT_COOLDOWN),
            "is_active": True
        }
        if index < len(self.enemies):
            self.enemies[index] = enemy
        else:
            self.enemies.append(enemy)

    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.player["vx"] = -self.PLAYER_SPEED
            self.player["facing"] = -1
        elif movement == 4: # Right
            self.player["vx"] = self.PLAYER_SPEED
            self.player["facing"] = 1
        else:
            self.player["vx"] *= 0.8 # Friction
        
        # Jumping
        if movement == 1 and self.player["jumps_left"] > 0:
            self.player["vy"] = self.PLAYER_JUMP_STRENGTH
            self.player["jumps_left"] -= 1
            self.player["on_ground"] = False
            reward += 1 # Reward for jumping
            # sfx: jump

        # Shooting
        if space_held and self.player["shoot_cooldown"] <= 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            proj_x = self.player["x"] + self.player["width"] / 2
            proj_y = self.player["y"] + 20
            self.projectiles.append({
                "x": proj_x, "y": proj_y, "vx": 10 * self.player["facing"], "owner": "player"
            })
            # sfx: player_shoot

        # --- Update Game State ---
        self.steps += 1
        
        # Update player
        self.player["vy"] += self.GRAVITY
        self.player["x"] += self.player["vx"]
        self.player["y"] += self.player["vy"]
        
        # Player bounds and ground collision
        self.player["x"] = np.clip(self.player["x"], 0, self.LEVEL_WIDTH - self.player["width"])
        if self.player["y"] + self.player["height"] >= self.SCREEN_HEIGHT - self.GROUND_HEIGHT:
            self.player["y"] = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - self.player["height"]
            self.player["vy"] = 0
            if not self.player["on_ground"]:
                self.player["on_ground"] = True
                self.player["jumps_left"] = self.PLAYER_MAX_JUMPS
                # sfx: land

        # Reward for progress
        progress = self.player["x"] - self.last_player_x
        if progress > 0:
            reward += 0.1 * progress / self.PLAYER_SPEED
        else:
            reward += -0.02 # Small penalty for moving away or standing still
        self.last_player_x = self.player["x"]
        
        # Update cooldowns
        if self.player["shoot_cooldown"] > 0: self.player["shoot_cooldown"] -= 1
        if self.player["invuln_timer"] > 0: self.player["invuln_timer"] -= 1

        # Update enemies
        enemy_proj_speed = self.BASE_ENEMY_PROJ_SPEED + (self.steps / 500) * 0.05
        for enemy in self.enemies:
            if not enemy["is_active"]: continue
            
            # Patrol
            enemy["x"] += self.ENEMY_SPEED * enemy["direction"]
            if enemy["x"] > enemy["patrol_end"] or enemy["x"] < enemy["patrol_start"]:
                enemy["direction"] *= -1

            # Shoot
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN
                direction_to_player = np.sign(self.player["x"] - enemy["x"])
                if direction_to_player != 0:
                    self.projectiles.append({
                        "x": enemy["x"] + enemy["width"] / 2, "y": enemy["y"] + enemy["height"] / 2,
                        "vx": enemy_proj_speed * direction_to_player, "owner": "enemy"
                    })
                    # sfx: enemy_shoot

        # Update projectiles and check collisions
        new_projectiles = []
        player_rect = pygame.Rect(self.player["x"], self.player["y"], self.player["width"], self.player["height"])
        
        for p in self.projectiles:
            p["x"] += p["vx"]
            
            p_rect = pygame.Rect(p["x"]-3, p["y"]-3, 6, 6)
            collided = False

            if p["owner"] == "player":
                for i, enemy in enumerate(self.enemies):
                    if not enemy["is_active"]: continue
                    enemy_rect = pygame.Rect(enemy["x"], enemy["y"], enemy["width"], enemy["height"])
                    if p_rect.colliderect(enemy_rect):
                        self._create_explosion(enemy["x"] + enemy["width"]/2, enemy["y"] + enemy["height"]/2)
                        self.score += 10
                        reward += 5
                        self._spawn_enemy(i) # Respawn
                        collided = True
                        # sfx: explosion
                        break
            
            elif p["owner"] == "enemy":
                if p_rect.colliderect(player_rect) and self.player["invuln_timer"] <= 0:
                    self.player["health"] -= 20
                    self.player["invuln_timer"] = self.PLAYER_INVULNERABILITY_FRAMES
                    reward -= 1
                    collided = True
                    # sfx: player_hit
            
            # Keep projectile if it's on screen and hasn't collided
            if not collided and 0 < p["x"] < self.LEVEL_WIDTH:
                new_projectiles.append(p)
        self.projectiles = new_projectiles
        
        # Update particles
        self.particles = [pa for pa in self.particles if pa["life"] > 0]
        for pa in self.particles:
            pa["x"] += pa["vx"]
            pa["y"] += pa["vy"]
            pa["life"] -= 1

        # Update camera
        self.camera_x = self.player["x"] - self.SCREEN_WIDTH / 3
        self.camera_x = np.clip(self.camera_x, 0, self.LEVEL_WIDTH - self.SCREEN_WIDTH)
        
        # --- Check Termination ---
        terminated = False
        if self.player["health"] <= 0:
            reward = -100
            terminated = True
            self.game_over = True
        
        if self.player["x"] >= self.LEVEL_WIDTH - self.player["width"] - 50: # Reached exit
            reward = 100
            terminated = True
            self.game_over = True
            self.score += 1000

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_explosion(self, x, y):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(15, 30),
                "color": random.choice(self.COLOR_EXPLOSION)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax Background
        for layer in self.background_layers:
            cam_offset = self.camera_x * layer["depth"]
            # Tiling for infinite scroll effect
            tile_width = self.SCREEN_WIDTH * 1.5
            start_x = - (cam_offset % tile_width)
            x = start_x
            while x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, layer["color"], 
                                 (x, layer["y_offset"], tile_width * 0.95, layer["height"]))
                x += tile_width
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT))

        # Exit Portal
        exit_x = self.LEVEL_WIDTH - 50 - self.camera_x
        exit_y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT - 100
        if exit_x < self.SCREEN_WIDTH:
            glow_size = 20 + 5 * math.sin(self.steps * 0.1)
            pygame.gfxdraw.filled_circle(self.screen, int(exit_x), int(exit_y), int(glow_size), (*self.COLOR_EXIT, 50))
            pygame.gfxdraw.filled_circle(self.screen, int(exit_x), int(exit_y), 15, self.COLOR_EXIT)
            pygame.gfxdraw.aacircle(self.screen, int(exit_x), int(exit_y), 15, self.COLOR_EXIT)

        # Enemies
        for enemy in self.enemies:
            if not enemy["is_active"]: continue
            ex, ey = int(enemy["x"] - self.camera_x), int(enemy["y"])
            if -enemy["width"] < ex < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, ex + enemy["width"]//2, ey + enemy["height"]//2, enemy["width"]//2 + 5, self.COLOR_ENEMY_GLOW)
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (ex, ey, enemy["width"], enemy["height"]))

        # Player
        px, py = int(self.player["x"] - self.camera_x), int(self.player["y"])
        p_w, p_h = self.player["width"], self.player["height"]
        
        # Invulnerability flash
        if self.player["invuln_timer"] > 0 and self.steps % 4 < 2:
            pass # Don't draw player
        else:
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, px + p_w//2, py + p_h//2, p_w//2 + 15, self.COLOR_PLAYER_GLOW)
            # Body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px, py, p_w, p_h))
            # Head/Cockpit
            pygame.draw.circle(self.screen, self.COLOR_BG, (px + p_w//2, py + 15), 8)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (px + p_w//2, py + 15), 8, 2)
            # Cannon
            cannon_x = px + p_w//2
            cannon_y = py + 20
            cannon_len = 20
            if self.player["facing"] > 0:
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (cannon_x, cannon_y), (cannon_x + cannon_len, cannon_y), 5)
            else:
                pygame.draw.line(self.screen, self.COLOR_PLAYER, (cannon_x, cannon_y), (cannon_x - cannon_len, cannon_y), 5)
        
        # Projectiles
        for p in self.projectiles:
            color = self.COLOR_PLAYER_PROJ if p["owner"] == "player" else self.COLOR_ENEMY_PROJ
            pos_x = int(p["x"] - self.camera_x)
            pos_y = int(p["y"])
            pygame.draw.line(self.screen, color, (pos_x-p["vx"]*0.5, pos_y), (pos_x, pos_y), 3)

        # Particles
        for pa in self.particles:
            alpha = int(255 * (pa["life"] / 30))
            color = (*pa["color"], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(pa["x"] - self.camera_x), int(pa["y"])))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player["health"] / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))
        
        # Jumps indicator
        for i in range(self.PLAYER_MAX_JUMPS):
            color = self.COLOR_PLAYER if i < self.player["jumps_left"] else self.COLOR_GROUND
            pygame.draw.circle(self.screen, color, (20 + i * 20, 45), 6)

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "player_x": self.player["x"],
            "jumps_left": self.player["jumps_left"],
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        move_action = 0 # No-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling & Rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Convert observation back to a Pygame Surface and draw it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()