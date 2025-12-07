import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:58:15.414853
# Source Brief: brief_00785.md
# Brief Index: 785
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
        "Defend your base from an onslaught of descending geometric shapes. "
        "Aim and fire to destroy enemies and survive for the duration."
    )
    user_guide = "Controls: Use ← and → arrow keys to aim your turret. Press space to fire."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 120

    # Colors
    COLOR_BG = (15, 15, 35)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER = (255, 255, 100)
    COLOR_LOW_LIFE = (255, 0, 0)
    ENEMY_COLORS = [
        (255, 0, 128),   # Magenta
        (0, 255, 0),     # Green
        (255, 128, 0),   # Orange
    ]

    # Player
    PLAYER_Y_POS = 380
    PLAYER_AIM_SPEED = 2.5  # degrees per frame
    PLAYER_AIM_LIMIT = 80  # degrees from vertical

    # Projectiles
    PROJECTILE_SPEED_INITIAL = 8.0
    PROJECTILE_RADIUS = 5
    PROJECTILE_SPEED_INCREASE = 1.05  # 5% increase

    # Enemies
    ENEMY_SPEED = 2.0
    ENEMY_SPAWN_PROB_INITIAL = 0.01
    ENEMY_SPAWN_PROB_INCREASE_PER_SEC = 0.001
    ENEMY_SIZE_RANGE = (12, 18)

    # Game
    MAX_STEPS = FPS * (GAME_DURATION_SECONDS + 10) # Safety margin
    STARTING_LIVES = 5
    WIN_CONDITION_LIVES = 5
    HITS_FOR_EXTRA_LIFE = 5


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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 28, bold=True)

        self.render_mode = render_mode
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.player_angle = 0.0
        self.game_timer = 0.0
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.last_space_state = False
        self.enemy_spawn_prob = self.ENEMY_SPAWN_PROB_INITIAL
        self.screen_shake = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.STARTING_LIVES
        self.player_angle = 0.0
        self.game_timer = float(self.GAME_DURATION_SECONDS)
        self.projectiles = []
        self.enemies = []
        self.particles = []
        self.last_space_state = False
        self.enemy_spawn_prob = self.ENEMY_SPAWN_PROB_INITIAL
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        self._handle_input(action)

        reward += self._update_projectiles()
        self._update_enemies() # This doesn't return reward, but causes life loss
        self._spawn_enemies()
        self._update_particles()
        self._update_timer_and_difficulty()

        terminated = self._check_termination()
        if terminated:
            if self.lives <= 0:
                reward = -100.0  # Lose condition
            elif self.game_timer <= 0 and self.lives >= self.WIN_CONDITION_LIVES:
                reward = 100.0  # Win condition
            else:
                reward = -50.0 # Time up but didn't meet win condition

        # Small penalty for losing a life is implicitly handled by large negative terminal reward
        # but we can add a per-step penalty to encourage avoiding it.
        # This is already handled by the _update_enemies logic not returning a reward.
        # Let's keep the reward structure simple as it is.

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Aiming
        if movement == 3:  # Left
            self.player_angle -= self.PLAYER_AIM_SPEED
        elif movement == 4:  # Right
            self.player_angle += self.PLAYER_AIM_SPEED
        self.player_angle = np.clip(self.player_angle, -self.PLAYER_AIM_LIMIT, self.PLAYER_AIM_LIMIT)

        # Firing on press (rising edge)
        if space_held and not self.last_space_state:
            self._fire_projectile()
        self.last_space_state = space_held

    def _fire_projectile(self):
        # sfx: player_shoot.wav
        angle_rad = math.radians(self.player_angle - 90)
        muzzle_offset = 20
        start_pos = [
            self.SCREEN_WIDTH / 2 + muzzle_offset * math.cos(angle_rad),
            self.PLAYER_Y_POS + muzzle_offset * math.sin(angle_rad)
        ]
        self.projectiles.append({
            "pos": start_pos,
            "angle_rad": angle_rad,
            "speed": self.PROJECTILE_SPEED_INITIAL,
            "hit_count": 0,
            "id": self.steps # Unique ID for this projectile
        })

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p["pos"][0] += p["speed"] * math.cos(p["angle_rad"])
            p["pos"][1] += p["speed"] * math.sin(p["angle_rad"])

            # Out of bounds check
            if not (0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(p)
                continue

            # Collision with enemies
            for e in self.enemies[:]:
                dist = math.hypot(p["pos"][0] - e["pos"][0], p["pos"][1] - e["pos"][1])
                if dist < self.PROJECTILE_RADIUS + e["size"]:
                    # sfx: enemy_hit.wav
                    self._create_particles(e["pos"], e["color"])
                    self.enemies.remove(e)
                    self.score += 1
                    reward += 1.0

                    p["hit_count"] += 1
                    p["speed"] *= self.PROJECTILE_SPEED_INCREASE

                    if p["hit_count"] >= self.HITS_FOR_EXTRA_LIFE:
                        # sfx: extra_life.wav
                        self.lives += 1
                        reward += 5.0
                        self.projectiles.remove(p) # Consume projectile
                    break # Projectile can only hit one enemy per frame
        return reward

    def _update_enemies(self):
        for e in self.enemies[:]:
            e["pos"][1] += self.ENEMY_SPEED
            e["angle"] = (e["angle"] + e["rot_speed"]) % 360
            if e["pos"][1] > self.SCREEN_HEIGHT + e["size"]:
                self.enemies.remove(e)
                self.lives -= 1
                # sfx: life_lost.wav
                self.screen_shake = 15 # Trigger screen shake

    def _spawn_enemies(self):
        if self.np_random.random() < self.enemy_spawn_prob:
            shape_type = self.np_random.choice(["circle", "square", "triangle"])
            size = self.np_random.uniform(self.ENEMY_SIZE_RANGE[0], self.ENEMY_SIZE_RANGE[1])
            self.enemies.append({
                "pos": [self.np_random.uniform(size, self.SCREEN_WIDTH - size), -size],
                "shape": shape_type,
                "color": random.choice(self.ENEMY_COLORS),
                "size": size,
                "angle": self.np_random.uniform(0, 360),
                "rot_speed": self.np_random.uniform(-2.5, 2.5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

    def _update_timer_and_difficulty(self):
        self.game_timer -= 1.0 / self.FPS
        self.enemy_spawn_prob += self.ENEMY_SPAWN_PROB_INCREASE_PER_SEC / self.FPS

    def _check_termination(self):
        if self.lives <= 0 or self.game_timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset[1] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            self.screen_shake = int(self.screen_shake * 0.9)

        self.screen.fill(self.COLOR_BG)
        self._render_game(render_offset)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        self._render_enemies(offset)
        self._render_projectiles(offset)
        self._render_particles(offset)
        self._render_player(offset)

    def _render_player(self, offset):
        x, y = self.SCREEN_WIDTH / 2 + offset[0], self.PLAYER_Y_POS + offset[1]
        base_width = 40
        height = 20
        top_width = 10
        points = [
            (-base_width / 2, height / 2), (base_width / 2, height / 2),
            (top_width / 2, -height / 2), (-top_width / 2, -height / 2)
        ]
        self._draw_rotated_polygon(self.screen, self.COLOR_PLAYER, points, self.player_angle, (x, y))

    def _render_projectiles(self, offset):
        for p in self.projectiles:
            x, y = int(p["pos"][0] + offset[0]), int(p["pos"][1] + offset[1])
            # Glow effect
            glow_radius = int(self.PROJECTILE_RADIUS * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PROJECTILE, 60), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            # Core
            pygame.gfxdraw.aacircle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)


    def _render_enemies(self, offset):
        for e in self.enemies:
            x, y = e["pos"][0] + offset[0], e["pos"][1] + offset[1]
            size = e["size"]
            if e["shape"] == "circle":
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(size), e["color"])
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), e["color"])
            else:
                if e["shape"] == "square":
                    points = [(-size, -size), (size, -size), (size, size), (-size, size)]
                else: # triangle
                    points = [(0, -size * 1.2), (-size, size * 0.8), (size, size * 0.8)]
                self._draw_rotated_polygon(self.screen, e["color"], points, e["angle"], (x,y))

    def _render_particles(self, offset):
        for p in self.particles:
            alpha = max(0, int(255 * (p["life"] / 40.0)))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0] + offset[0]), int(p["pos"][1] + offset[1]))
            size = int(p["size"] * (p["life"] / 40.0))
            if size > 0:
                # Using a surface for alpha blending particles
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timer
        timer_str = f"{max(0, self.game_timer):.1f}"
        timer_text = self.font_timer.render(timer_str, True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH / 2 - timer_text.get_width() / 2, 10))

        # Lives
        heart_size = 12
        for i in range(self.lives):
            self._draw_heart(15 + i * (heart_size * 2 + 5), 22, heart_size, self.COLOR_LOW_LIFE)

        # Low life warning
        if self.lives < 3 and self.lives > 0:
            alpha = 128 + 127 * math.sin(self.steps * 0.2)
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_LOW_LIFE, int(alpha)), (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 5)
            self.screen.blit(s, (0,0))


    def _draw_rotated_polygon(self, surface, color, points, angle, pivot):
        angle_rad = math.radians(angle)
        sin, cos = math.sin(angle_rad), math.cos(angle_rad)
        rotated_points = []
        for x, y in points:
            rx = x * cos - y * sin + pivot[0]
            ry = x * sin + y * cos + pivot[1]
            rotated_points.append((int(rx), int(ry)))
        pygame.gfxdraw.aapolygon(surface, rotated_points, color)
        pygame.gfxdraw.filled_polygon(surface, rotated_points, color)

    def _draw_heart(self, x, y, size, color):
        # Simple procedural heart for lives display
        p1 = (x, y + size * 0.25)
        p2 = (x + size, y - size * 0.5)
        p3 = (x + size * 2, y + size * 0.25)
        p4 = (x + size, y + size * 1.25)
        pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3,p4], color)
        pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4], color)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "timer": self.game_timer,
        }

if __name__ == '__main__':
    # This block will not be executed in the testing environment, but is useful for local development.
    # To run, you'll need to `pip install pygame`.
    # It also requires a display, so it will not run in a headless environment.
    # To run it, you might need to comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.

    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Create a window to display the game
    # This part requires a display.
    try:
        if "SDL_VIDEODRIVER" in os.environ:
             del os.environ["SDL_VIDEODRIVER"]
        pygame.display.init()
        pygame.display.set_caption("Geometric Onslaught")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        display_enabled = True
    except pygame.error:
        print("Pygame display could not be initialized. Running without visual output.")
        display_enabled = False


    running = True
    total_reward = 0
    while running:
        action = [0, 0, 0] # Default to no-op
        if display_enabled:
            # Human player controls
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space, shift]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        else:
            # In headless mode, use random actions
            action = env.action_space.sample()


        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if display_enabled:
            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()
            if not display_enabled: # Run one episode and stop in headless mode
                running = False


    if display_enabled:
        pygame.quit()