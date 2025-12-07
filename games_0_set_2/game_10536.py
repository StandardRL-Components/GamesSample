import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "A retro-futuristic synthwave shooter. Dodge enemy fire, slow time, and flip gravity to survive the onslaught."
    )
    user_guide = (
        "Use arrow keys to move. Press space to toggle time slow and shift to flip projectile gravity."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_indicator = pygame.font.Font(None, 20)

        # --- Visual Style ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_GRID = (35, 30, 60)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_ENEMY = (255, 0, 128)
        self.COLOR_PLAYER_PROJ = (0, 255, 128)
        self.COLOR_ENEMY_PROJ = (255, 180, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SUN_OUTER = (255, 100, 0)
        self.COLOR_SUN_INNER = (255, 200, 0)

        # --- Game Mechanics Parameters ---
        self.MAX_STEPS = 5000
        self.PLAYER_SPEED = 6
        self.PLAYER_SIZE = 12
        self.PLAYER_FIRE_COOLDOWN_MAX = 8
        self.PLAYER_PROJ_SPEED = 12
        self.PLAYER_PROJ_SIZE = 4
        self.ENEMY_SIZE = 15
        self.ENEMY_BASE_SPEED = 1.5
        self.ENEMY_FIRE_COOLDOWN_MAX = 90
        self.ENEMY_PROJ_SIZE = 5
        self.TIME_SLOW_FACTOR = 0.25
        self.TIME_SLOW_COOLDOWN_MAX = 60
        self.GRAVITY_FLIP_COOLDOWN_MAX = 30
        self.INITIAL_ENEMY_SPAWN_RATE = 0.05
        self.ENEMY_SPAWN_RATE_INCREASE = 0.0001
        self.MAX_ENEMY_SPAWN_RATE = 0.3
        self.INITIAL_ENEMY_PROJ_SPEED = 2.0
        self.ENEMY_PROJ_SPEED_INCREASE = 0.02
        self.MAX_ENEMY_PROJ_SPEED = 6.0

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_fire_cooldown = 0
        self.is_time_slowed = False
        self.time_slow_cooldown = 0
        self.is_gravity_flipped = False
        self.gravity_flip_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.enemy_spawn_rate = 0
        self.enemy_projectile_speed = 0
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.8], dtype=np.float32)
        self.player_fire_cooldown = 0

        self.is_time_slowed = False
        self.time_slow_cooldown = 0
        self.is_gravity_flipped = False
        self.gravity_flip_cooldown = 0

        self.prev_space_held = False
        self.prev_shift_held = False

        self.enemy_spawn_rate = self.INITIAL_ENEMY_SPAWN_RATE
        self.enemy_projectile_speed = self.INITIAL_ENEMY_PROJ_SPEED

        self.player_projectiles.clear()
        self.enemies.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.1  # Survival reward

        # --- 1. Handle Input & State Toggles ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if space_pressed and self.time_slow_cooldown <= 0:
            self.is_time_slowed = not self.is_time_slowed
            self.time_slow_cooldown = self.TIME_SLOW_COOLDOWN_MAX

        if shift_pressed and self.gravity_flip_cooldown <= 0:
            self.is_gravity_flipped = not self.is_gravity_flipped
            self.gravity_flip_cooldown = self.GRAVITY_FLIP_COOLDOWN_MAX

        # --- 2. Update Game Logic ---
        time_factor = self.TIME_SLOW_FACTOR if self.is_time_slowed else 1.0

        # Update Cooldowns
        self.time_slow_cooldown = max(0, self.time_slow_cooldown - 1)
        self.gravity_flip_cooldown = max(0, self.gravity_flip_cooldown - 1)
        self.player_fire_cooldown = max(0, self.player_fire_cooldown - 1)

        # Player Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

        # Player Firing (Automatic)
        if self.player_fire_cooldown <= 0:
            direction = np.array([0, -1], dtype=np.float32)
            self.player_projectiles.append({
                "pos": self.player_pos.copy(),
                "vel": direction * self.PLAYER_PROJ_SPEED,
                "trail": [self.player_pos.copy() for _ in range(3)]
            })
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

        # Enemy Spawning & Progression
        self.enemy_spawn_rate = min(self.MAX_ENEMY_SPAWN_RATE, self.enemy_spawn_rate + self.ENEMY_SPAWN_RATE_INCREASE * time_factor)
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_projectile_speed = min(self.MAX_ENEMY_PROJ_SPEED, self.enemy_projectile_speed + self.ENEMY_PROJ_SPEED_INCREASE)

        if self.np_random.random() < self.enemy_spawn_rate * time_factor:
            spawn_x = self.np_random.uniform(self.ENEMY_SIZE, self.SCREEN_WIDTH - self.ENEMY_SIZE)
            self.enemies.append({
                "pos": np.array([spawn_x, -self.ENEMY_SIZE]),
                "vel": np.array([self.np_random.choice([-1, 1]) * self.ENEMY_BASE_SPEED, self.ENEMY_BASE_SPEED * 0.5]),
                "fire_cooldown": self.np_random.integers(0, self.ENEMY_FIRE_COOLDOWN_MAX),
                "flash": 0
            })

        # Update Player Projectiles
        for proj in self.player_projectiles[:]:
            proj["pos"] += proj["vel"]
            proj["trail"].append(proj["pos"].copy())
            if len(proj["trail"]) > 3: proj["trail"].pop(0)

            if self.is_gravity_flipped:
                if proj["pos"][0] < 0 or proj["pos"][0] > self.SCREEN_WIDTH:
                    proj["vel"][0] *= -1
            
            if proj["pos"][1] < 0 or proj["pos"][1] > self.SCREEN_HEIGHT:
                if not self.is_gravity_flipped or (proj["pos"][0] > 0 and proj["pos"][0] < self.SCREEN_WIDTH):
                    self.player_projectiles.remove(proj)

        # Update Enemies
        for enemy in self.enemies[:]:
            enemy["pos"] += enemy["vel"] * time_factor
            if enemy["pos"][0] < self.ENEMY_SIZE or enemy["pos"][0] > self.SCREEN_WIDTH - self.ENEMY_SIZE:
                enemy["vel"][0] *= -1
            if enemy["pos"][1] > self.SCREEN_HEIGHT + self.ENEMY_SIZE:
                self.enemies.remove(enemy)
                continue

            enemy["fire_cooldown"] = max(0, enemy["fire_cooldown"] - 1 * time_factor)
            if enemy["fire_cooldown"] <= 0:
                direction = self.player_pos - enemy["pos"]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                    self.enemy_projectiles.append({
                        "pos": enemy["pos"].copy(),
                        "vel": direction * self.enemy_projectile_speed
                    })
                    enemy["fire_cooldown"] = self.ENEMY_FIRE_COOLDOWN_MAX
                    enemy["flash"] = 5 # Flash duration
            enemy["flash"] = max(0, enemy["flash"] - 1)

        # Update Enemy Projectiles
        for proj in self.enemy_projectiles[:]:
            proj["pos"] += proj["vel"] * time_factor
            if not (0 < proj["pos"][0] < self.SCREEN_WIDTH and 0 < proj["pos"][1] < self.SCREEN_HEIGHT):
                self.enemy_projectiles.remove(proj)

        # Update Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"] * time_factor
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # --- 3. Collision Detection & Rewards ---
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj["pos"] - enemy["pos"]) < self.PLAYER_PROJ_SIZE + self.ENEMY_SIZE:
                    reward += 1.0
                    self.score += 10
                    for _ in range(20):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                        self.particles.append({
                            "pos": enemy["pos"].copy(),
                            "vel": vel,
                            "life": self.np_random.integers(15, 30),
                            "color": self.COLOR_ENEMY
                        })
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    if enemy in self.enemies: self.enemies.remove(enemy)
                    break

        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles:
            if np.linalg.norm(proj["pos"] - self.player_pos) < self.ENEMY_PROJ_SIZE + self.PLAYER_SIZE:
                self.game_over = True
                for _ in range(50):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(2, 6)
                    vel = np.array([math.cos(angle), math.sin(angle)]) * speed
                    self.particles.append({
                        "pos": self.player_pos.copy(),
                        "vel": vel,
                        "life": self.np_random.integers(20, 40),
                        "color": self.COLOR_PLAYER
                    })
                break
        
        # --- 4. Termination Check ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS

        if terminated and self.game_over:
            reward = -100.0
        if truncated and not terminated:
            reward += 10.0

        self.steps += 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_particles()
        for proj in self.enemy_projectiles:
            self._render_projectile(proj, self.COLOR_ENEMY_PROJ, self.ENEMY_PROJ_SIZE)
        for proj in self.player_projectiles:
            self._render_player_projectile(proj)
        for enemy in self.enemies:
            self._render_enemy(enemy)
        if not self.game_over:
            self._render_player()

    def _render_background(self):
        # Grid
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Synthwave Sun
        sun_pos = (self.SCREEN_WIDTH // 2, 0)
        for i in range(15, 0, -1):
            alpha = 50 - i * 3
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, sun_pos[0], sun_pos[1], 100 + i * 10, (*self.COLOR_SUN_OUTER, alpha))
        pygame.gfxdraw.filled_circle(self.screen, sun_pos[0], sun_pos[1], 100, self.COLOR_SUN_INNER)
        for i in range(1, 10):
             pygame.draw.line(self.screen, self.COLOR_BG, (0, sun_pos[1] + i * 8), (self.SCREEN_WIDTH, sun_pos[1] + i * 8), 3)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        size = int(self.PLAYER_SIZE)
        # Glow
        for i in range(size, 0, -2):
            alpha = 100 - (i / size) * 100
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i + 5, (*self.COLOR_PLAYER, int(alpha)))
        # Core
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (pos[0] - size//2, pos[1] - size//2, size, size))

    def _render_enemy(self, enemy):
        pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
        size = self.ENEMY_SIZE
        points = [
            (pos[0], pos[1] - size),
            (pos[0] - size, pos[1] + size),
            (pos[0] + size, pos[1] + size),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        if enemy["flash"] > 0:
            flash_color = (255, 255, 255)
            flash_size = size * 0.5
            flash_points = [
                (pos[0], pos[1] - flash_size),
                (pos[0] - flash_size, pos[1] + flash_size),
                (pos[0] + flash_size, pos[1] + flash_size),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, flash_points, flash_color)

    def _render_player_projectile(self, proj):
        start_pos = proj["trail"][0]
        end_pos = proj["pos"]
        pygame.draw.line(self.screen, (*self.COLOR_PLAYER_PROJ, 100), start_pos, end_pos, 3)
        pygame.gfxdraw.filled_circle(self.screen, int(end_pos[0]), int(end_pos[1]), self.PLAYER_PROJ_SIZE, self.COLOR_PLAYER_PROJ)
    
    def _render_projectile(self, proj, color, size):
        pos = (int(proj["pos"][0]), int(proj["pos"][1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = max(1, int(p["life"] / 10))
            if size > 0:
                pygame.draw.rect(self.screen, color, (pos[0], pos[1], size, size))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font_main.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 35))

        if self.is_time_slowed:
            slow_text = self.font_indicator.render("TIME SLOW", True, self.COLOR_PLAYER)
            self.screen.blit(slow_text, (self.SCREEN_WIDTH - slow_text.get_width() - 10, 10))
        
        if self.is_gravity_flipped:
            grav_text = self.font_indicator.render("GRAVITY FLIP", True, self.COLOR_ENEMY)
            self.screen.blit(grav_text, (self.SCREEN_WIDTH - grav_text.get_width() - 10, 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example usage for interactive play
if __name__ == '__main__':
    # Un-comment the next line to run headlessly
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame window for human play
    pygame.display.set_caption("Synthwave Siege")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    while not terminated and not truncated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        
        # Movement
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        
        if up: movement = 1
        elif down: movement = 2
        elif left: movement = 3
        elif right: movement = 4
        else: movement = 0
            
        # Other actions
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the interactive play speed

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()