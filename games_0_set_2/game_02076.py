
# Generated: 2025-08-27T19:12:44.802659
# Source Brief: brief_02076.md
# Brief Index: 2076

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade shooter where the player must survive against hordes of
    zombies for a fixed amount of time. The game features real-time action,
    resource management (ammo), and escalating difficulty.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to shoot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of procedurally generated zombies in a top-down arena for 5 minutes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000
    PLAYER_SPEED = 3.5
    PLAYER_HEALTH_MAX = 100
    PLAYER_AMMO_MAX = 100
    PLAYER_RADIUS = 8
    PLAYER_HIT_COOLDOWN = 30 # 1 second of invulnerability at 30fps

    ZOMBIE_SPEED = 1.0
    ZOMBIE_HEALTH_MAX = 10
    ZOMBIE_RADIUS = 10
    ZOMBIE_SPAWN_INTERVAL = 50
    ZOMBIE_DIFFICULTY_INTERVAL = 600

    AMMO_CRATE_SPAWN_INTERVAL = 200
    AMMO_CRATE_VALUE = 25
    AMMO_CRATE_RADIUS = 10

    PROJECTILE_SPEED = 10
    PROJECTILE_RADIUS = 3
    FIRE_COOLDOWN = 6  # 5 shots per second at 30fps

    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_WALL = (60, 70, 80)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_ZOMBIE_HIT = (255, 150, 150)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_AMMO = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH = (50, 220, 50)
    COLOR_UI_HEALTH_BG = (100, 40, 40)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_last_move_dir = None
        self.player_hit_cooldown = None
        self.zombies = []
        self.projectiles = []
        self.ammo_crates = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fire_cooldown_timer = 0
        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = 0
        self.zombies_to_spawn = 1
        self.next_spawn_increase_step = self.ZOMBIE_DIFFICULTY_INTERVAL
        self.screen_shake = 0

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = 50
        self.player_last_move_dir = [0, -1] # Default aim up
        self.player_hit_cooldown = 0

        self.zombies = []
        self.projectiles = []
        self.ammo_crates = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.fire_cooldown_timer = 0
        self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
        self.ammo_spawn_timer = self.AMMO_CRATE_SPAWN_INTERVAL
        self.zombies_to_spawn = 1
        self.next_spawn_increase_step = self.ZOMBIE_DIFFICULTY_INTERVAL
        self.screen_shake = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], bool(action[1]), bool(action[2])
        
        reward = 0.01 # Survival reward per step

        shot_fired = self._handle_player_input(movement, space_held)
        if shot_fired:
            reward -= 0.1

        self._update_projectiles()
        self._update_zombies()
        self._update_particles()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        self._handle_spawning()
        self._update_timers()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100
            else:
                reward -= 100
        
        if collision_rewards > 0:
            self.score += collision_rewards
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held):
        move_vec = [0, 0]
        if movement == 1: move_vec[1] = -1
        elif movement == 2: move_vec[1] = 1
        elif movement == 3: move_vec[0] = -1
        elif movement == 4: move_vec[0] = 1
        
        if movement != 0:
            self.player_last_move_dir = move_vec

        self.player_pos[0] += move_vec[0] * self.PLAYER_SPEED
        self.player_pos[1] += move_vec[1] * self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)
        
        if space_held and self.fire_cooldown_timer == 0 and self.player_ammo > 0:
            self._fire_projectile()
            return True
        return False

    def _fire_projectile(self):
        # sfx: player_shoot.wav
        self.player_ammo -= 1
        self.fire_cooldown_timer = self.FIRE_COOLDOWN
        
        start_pos = list(self.player_pos)
        self.projectiles.append({"pos": start_pos, "vel": self.player_last_move_dir})

        for _ in range(10):
            angle = math.atan2(self.player_last_move_dir[1], self.player_last_move_dir[0]) + self.np_random.uniform(-0.5, 0.5)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(start_pos), "vel": vel, "life": 8, 
                "color": (255, self.np_random.integers(150, 255), 0), "radius": self.np_random.uniform(1, 3)
            })

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0] * self.PROJECTILE_SPEED
            p["pos"][1] += p["vel"][1] * self.PROJECTILE_SPEED
            if not (0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies(self):
        for z in self.zombies:
            if z["hit_timer"] > 0: z["hit_timer"] -= 1
            
            direction_x = self.player_pos[0] - z["pos"][0]
            direction_y = self.player_pos[1] - z["pos"][1]
            dist = math.hypot(direction_x, direction_y)
            
            if dist > 1:
                direction_x /= dist
                direction_y /= dist
            
            z["pos"][0] += direction_x * self.ZOMBIE_SPEED
            z["pos"][1] += direction_y * self.ZOMBIE_SPEED

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0: self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        for p in self.projectiles[:]:
            for z in self.zombies[:]:
                if math.hypot(p["pos"][0] - z["pos"][0], p["pos"][1] - z["pos"][1]) < self.PROJECTILE_RADIUS + self.ZOMBIE_RADIUS:
                    # sfx: zombie_hit.wav
                    if p in self.projectiles: self.projectiles.remove(p)
                    z["health"] -= 10
                    z["hit_timer"] = 5
                    if z["health"] <= 0:
                        # sfx: zombie_die.wav
                        reward += 1.0
                        self._create_blood_particles(z["pos"])
                        self.zombies.remove(z)
                    break 

        if self.player_hit_cooldown == 0:
            for z in self.zombies:
                if math.hypot(self.player_pos[0] - z["pos"][0], self.player_pos[1] - z["pos"][1]) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                    # sfx: player_hurt.wav
                    self.player_health = max(0, self.player_health - 10)
                    self.player_hit_cooldown = self.PLAYER_HIT_COOLDOWN
                    self.screen_shake = 10
                    break
        
        for crate in self.ammo_crates[:]:
            if math.hypot(self.player_pos[0] - crate["pos"][0], self.player_pos[1] - crate["pos"][1]) < self.PLAYER_RADIUS + self.AMMO_CRATE_RADIUS:
                # sfx: ammo_pickup.wav
                reward += 0.5
                self.player_ammo = min(self.PLAYER_AMMO_MAX, self.player_ammo + self.AMMO_CRATE_VALUE)
                self._create_pickup_particles(crate["pos"])
                self.ammo_crates.remove(crate)

        return reward

    def _create_blood_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": self.np_random.integers(10, 20), 
                "color": (self.np_random.integers(150, 220), 0, 0), "radius": self.np_random.uniform(2, 4)
            })

    def _create_pickup_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "life": self.np_random.integers(15, 25), 
                "color": self.COLOR_AMMO, "radius": self.np_random.uniform(1, 3)
            })

    def _handle_spawning(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
            for _ in range(self.zombies_to_spawn):
                side = self.np_random.integers(4)
                if side == 0: pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
                elif side == 1: pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
                elif side == 2: pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
                else: pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
                self.zombies.append({"pos": pos, "health": self.ZOMBIE_HEALTH_MAX, "hit_timer": 0})
        
        self.ammo_spawn_timer -= 1
        if self.ammo_spawn_timer <= 0 and not self.ammo_crates:
            self.ammo_spawn_timer = self.AMMO_CRATE_SPAWN_INTERVAL
            pos = [self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)]
            self.ammo_crates.append({"pos": pos, "spawn_time": self.steps})

        if self.steps >= self.next_spawn_increase_step:
            self.zombies_to_spawn += 1
            self.next_spawn_increase_step += self.ZOMBIE_DIFFICULTY_INTERVAL

    def _update_timers(self):
        if self.fire_cooldown_timer > 0: self.fire_cooldown_timer -= 1
        if self.player_hit_cooldown > 0: self.player_hit_cooldown -= 1
        if self.screen_shake > 0: self.screen_shake -= 1

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            return True
        return False

    def _get_observation(self):
        offset = [self.np_random.integers(-5, 6) if self.screen_shake > 0 else 0 for _ in range(2)]

        self.screen.fill(self.COLOR_BG)
        self._render_game(offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (offset[0], offset[1], self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 10)

        for crate in self.ammo_crates:
            x, y = int(crate["pos"][0] + offset[0]), int(crate["pos"][1] + offset[1])
            pulse = abs(math.sin((self.steps - crate["spawn_time"]) * 0.1)) * 5
            rect = pygame.Rect(x - self.AMMO_CRATE_RADIUS, y - self.AMMO_CRATE_RADIUS, self.AMMO_CRATE_RADIUS*2, self.AMMO_CRATE_RADIUS*2)
            pygame.draw.rect(self.screen, self.COLOR_AMMO, rect, 0, 3)
            pygame.gfxdraw.rectangle(self.screen, rect, (*self.COLOR_AMMO, int(100 + pulse * 10)))

        for z in self.zombies:
            x, y = int(z["pos"][0] + offset[0]), int(z["pos"][1] + offset[1])
            color = self.COLOR_ZOMBIE_HIT if z["hit_timer"] > 0 else self.COLOR_ZOMBIE
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.ZOMBIE_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.ZOMBIE_RADIUS, color)

        if self.player_health > 0 and not (self.player_hit_cooldown > 0 and self.steps % 4 < 2):
            px, py = int(self.player_pos[0] + offset[0]), int(self.player_pos[1] + offset[1])
            for i in range(5):
                alpha = max(0, 150 - i * 30)
                pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS + i, (*self.COLOR_PLAYER_GLOW, alpha))
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.PLAYER_RADIUS, self.COLOR_PLAYER)
            end_x = px + self.player_last_move_dir[0] * 12
            end_y = py + self.player_last_move_dir[1] * 12
            pygame.draw.line(self.screen, self.COLOR_PLAYER_GLOW, (px, py), (end_x, end_y), 3)

        for p in self.projectiles:
            x, y = int(p["pos"][0] + offset[0]), int(p["pos"][1] + offset[1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        for p in self.particles:
            x, y = int(p["pos"][0] + offset[0]), int(p["pos"][1] + offset[1])
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = (*p["color"], alpha)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(p["radius"]), color)

    def _render_ui(self):
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, int(200 * health_ratio), 20))
        
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (220, 12))
        
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        mins, secs = divmod(time_left, 60)
        timer_text = self.font_ui.render(f"TIME: {int(mins):02}:{int(secs):02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - 130, 12))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU SURVIVED" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 100, 100)
            text = self.font_game_over.render(message, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()