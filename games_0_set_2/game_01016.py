
# Generated: 2025-08-27T15:33:12.994382
# Source Brief: brief_01016.md
# Brief Index: 1016

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to aim the cannon. Press space to fire. Defend your base!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemy vectors in this fast-paced, minimalist shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 1500 # Approx 50 seconds at 30fps

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_PROJECTILE = (128, 255, 200)
    COLOR_ENEMY_BASE = (255, 50, 50)
    COLOR_ENEMY_PROJECTILE = (255, 100, 100)
    COLOR_EXPLOSION = (255, 200, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_HIGH = (0, 255, 0)
    COLOR_HEALTH_MED = (255, 255, 0)
    COLOR_HEALTH_LOW = (255, 0, 0)

    # Player Base/Cannon
    BASE_WIDTH, BASE_HEIGHT = 100, 20
    BASE_Y = SCREEN_HEIGHT - 30
    CANNON_LENGTH = 30
    CANNON_TURN_SPEED = 0.05  # Radians per step
    CANNON_ANGLE_LIMIT = math.pi / 2.2 # Limit aiming
    PLAYER_FIRE_COOLDOWN = 6 # Steps between shots
    PLAYER_PROJECTILE_SPEED = 15

    # Enemy config
    WAVE_CONFIG = [
        # Wave 1
        {"count": 5, "health": 1, "speed_range": (0.5, 1.0), "fire_rate": 60, "projectile_speed": 2.0, "shape": "triangle"},
        # Wave 2
        {"count": 7, "health": 2, "speed_range": (0.7, 1.2), "fire_rate": 45, "projectile_speed": 2.5, "shape": "square"},
        # Wave 3
        {"count": 10, "health": 3, "speed_range": (1.0, 1.5), "fire_rate": 30, "projectile_speed": 3.0, "shape": "pentagon"},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.current_wave = 0
        self.player_cannon_angle = 0
        self.player_fire_cooldown_timer = 0
        self.last_space_held = False
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []
        self.np_random = None
        
        self.reset()
        
        # Self-check after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 100
        self.current_wave = 0
        self.player_cannon_angle = 0
        self.player_fire_cooldown_timer = 0
        self.last_space_held = False
        
        self.player_projectiles.clear()
        self.enemies.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self._spawn_wave()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0

        # --- 1. Handle Player Input ---
        if movement == 3: # Left
            self.player_cannon_angle -= self.CANNON_TURN_SPEED
        elif movement == 4: # Right
            self.player_cannon_angle += self.CANNON_TURN_SPEED
        self.player_cannon_angle = np.clip(self.player_cannon_angle, -self.CANNON_ANGLE_LIMIT, self.CANNON_ANGLE_LIMIT)

        fired_this_step = False
        if space_held and not self.last_space_held and self.player_fire_cooldown_timer <= 0:
            self._fire_player_projectile()
            fired_this_step = True
        self.last_space_held = space_held

        # --- 2. Update Game Logic ---
        self._update_player()
        reward += self._update_player_projectiles(fired_this_step)
        reward += self._update_enemies()
        reward += self._update_enemy_projectiles()
        self._update_particles()
        
        # --- 3. Wave Progression ---
        if not self.enemies and not self.game_over:
            if self.current_wave < len(self.WAVE_CONFIG):
                reward += 50 # Wave clear reward
                self.score += 500
                self._spawn_wave()
            else: # All waves cleared
                self.game_over = True
                reward += 100 # Game win reward
        
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    # --- Update Logic Methods ---
    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > len(self.WAVE_CONFIG): return

        config = self.WAVE_CONFIG[self.current_wave - 1]
        for _ in range(config["count"]):
            enemy = {
                "pos": np.array([self.np_random.uniform(50, self.SCREEN_WIDTH - 50), self.np_random.uniform(-100, -20)], dtype=float),
                "vel": np.array([0, self.np_random.uniform(*config["speed_range"])], dtype=float),
                "health": config["health"],
                "fire_cooldown": self.np_random.integers(0, config["fire_rate"]),
                "fire_rate": config["fire_rate"],
                "projectile_speed": config["projectile_speed"],
                "shape": config["shape"],
                "size": 12 + config["health"]
            }
            self.enemies.append(enemy)

    def _fire_player_projectile(self):
        # sfx: player_shoot.wav
        self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
        angle = self.player_cannon_angle - math.pi / 2
        velocity = np.array([math.cos(angle), math.sin(angle)]) * self.PLAYER_PROJECTILE_SPEED
        start_pos = self._get_cannon_tip()
        self.player_projectiles.append({"pos": start_pos, "vel": velocity})

    def _update_player(self):
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1

    def _update_player_projectiles(self, fired_this_step):
        reward = 0
        projectiles_to_keep = []
        for p in self.player_projectiles:
            p["pos"] += p["vel"]
            
            hit_enemy = False
            for enemy in self.enemies:
                if np.linalg.norm(p["pos"] - enemy["pos"]) < enemy["size"]:
                    # sfx: enemy_hit.wav
                    enemy["health"] -= 1
                    hit_enemy = True
                    if enemy["health"] <= 0:
                        reward += 1
                        self.score += 100
                        self._create_explosion(enemy["pos"], 20, self.COLOR_EXPLOSION)
                        self.enemies.remove(enemy)
                    else:
                        self._create_explosion(p["pos"], 5, self.COLOR_PLAYER_PROJECTILE)
                    break 
            
            if hit_enemy: continue

            if not (0 < p["pos"][0] < self.SCREEN_WIDTH and p["pos"][1] > 0):
                if not fired_this_step: reward -= 0.01
                continue
            
            projectiles_to_keep.append(p)
        self.player_projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        base_pos = np.array([self.SCREEN_WIDTH / 2, self.BASE_Y])
        for enemy in self.enemies:
            enemy["pos"] += enemy["vel"]
            
            if enemy["pos"][1] > self.SCREEN_HEIGHT + 50:
                self.enemies.remove(enemy)
                continue

            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                enemy["fire_cooldown"] = enemy["fire_rate"] + self.np_random.integers(-10, 10)
                direction = base_pos - enemy["pos"]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    velocity = (direction / direction_norm) * enemy["projectile_speed"]
                    self.enemy_projectiles.append({"pos": enemy["pos"].copy(), "vel": velocity})
        return 0

    def _update_enemy_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        base_rect = pygame.Rect(self.SCREEN_WIDTH/2 - self.BASE_WIDTH/2, self.BASE_Y, self.BASE_WIDTH, self.BASE_HEIGHT)

        for p in self.enemy_projectiles:
            p["pos"] += p["vel"]
            
            if base_rect.collidepoint(p["pos"][0], p["pos"][1]):
                # sfx: base_hit.wav
                self.base_health -= 10
                reward -= 10
                self._create_explosion(p["pos"], 15, self.COLOR_ENEMY_BASE)
                continue

            if p["pos"][1] > self.BASE_Y + self.BASE_HEIGHT:
                reward += 0.1
                continue

            if not (0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT):
                continue

            projectiles_to_keep.append(p)
        self.enemy_projectiles = projectiles_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _check_termination(self):
        return self.base_health <= 0 or self.steps >= self.MAX_STEPS or (self.game_over and self.current_wave > len(self.WAVE_CONFIG))

    # --- Rendering Methods ---
    def _render_game(self):
        for p in self.particles: p.draw(self.screen)
            
        for p in self.player_projectiles:
            start_pos = (int(p["pos"][0]), int(p["pos"][1]))
            end_pos = (int(p["pos"][0] - p["vel"][0]*0.5), int(p["pos"][1] - p["vel"][1]*0.5))
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_PROJECTILE, start_pos, end_pos, 2)

        for p in self.enemy_projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_ENEMY_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_ENEMY_PROJECTILE)

        for enemy in self.enemies: self._draw_enemy_shape(enemy)

        base_rect = pygame.Rect(self.SCREEN_WIDTH/2 - self.BASE_WIDTH/2, self.BASE_Y, self.BASE_WIDTH, self.BASE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, base_rect, border_radius=3)
        
        cannon_tip = self._get_cannon_tip()
        cannon_base = (self.SCREEN_WIDTH / 2, self.BASE_Y)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, cannon_base, cannon_tip, 6)
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, cannon_base, cannon_tip)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{len(self.WAVE_CONFIG)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        health_pct = max(0, self.base_health / 100)
        health_bar_width = self.BASE_WIDTH * health_pct
        health_bar_rect = pygame.Rect(self.SCREEN_WIDTH/2 - self.BASE_WIDTH/2, self.BASE_Y - 12, health_bar_width, 8)
        
        if health_pct > 0.6: color = self.COLOR_HEALTH_HIGH
        elif health_pct > 0.3: color = self.COLOR_HEALTH_MED
        else: color = self.COLOR_HEALTH_LOW
        pygame.draw.rect(self.screen, color, health_bar_rect, border_radius=2)
        
        if self.game_over:
            msg = "YOU WIN!" if self.base_health > 0 else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Helper Methods ---
    def _get_cannon_tip(self):
        angle = self.player_cannon_angle - math.pi / 2
        return np.array([
            self.SCREEN_WIDTH / 2 + self.CANNON_LENGTH * math.cos(angle),
            self.BASE_Y + self.CANNON_LENGTH * math.sin(angle)
        ])

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            self.particles.append(Particle(pos, color, self.np_random))

    def _draw_enemy_shape(self, enemy):
        points = []
        size, x, y = enemy["size"], enemy["pos"][0], enemy["pos"][1]
        
        if enemy["shape"] == "triangle":
            points = [(x, y - size), (x - size / 1.15, y + size / 2), (x + size / 1.15, y + size / 2)]
        elif enemy["shape"] == "square":
            angle = (self.steps % 120 / 120) * math.pi * 2
            for i in range(4):
                a = angle + (math.pi / 2 * i) + (math.pi / 4)
                points.append((x + size * math.cos(a), y + size * math.sin(a)))
        elif enemy["shape"] == "pentagon":
            angle = (self.steps % 180 / 180) * math.pi * 2
            for i in range(5):
                a = angle + (2 * math.pi / 5 * i)
                points.append((x + size * math.cos(a), y + size * math.sin(a)))
        
        if points:
            int_points = [(int(px), int(py)) for px, py in points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ENEMY_BASE)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ENEMY_BASE)

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
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Particle Class for Effects ---
class Particle:
    def __init__(self, pos, color, np_random):
        self.x, self.y = pos
        self.color = color
        self.np_random = np_random
        self.vx = self.np_random.uniform(-2, 2)
        self.vy = self.np_random.uniform(-2, 2)
        self.rad = self.np_random.uniform(2, 5)
        self.life = self.np_random.integers(10, 25)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.rad -= 0.15
        return self.rad > 0

    def draw(self, surface):
        if self.rad > 0:
            alpha = max(0, min(255, int(255 * (self.life / 15))))
            radius = int(self.rad)
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
            surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)))

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    running = True
    total_reward = 0
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Vector Vindicator")
    clock = pygame.time.Clock()
    key_map = {pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]: running = False
        for key, move_action in key_map.items():
            if keys[key]: movement = move_action; break
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)
    pygame.quit()