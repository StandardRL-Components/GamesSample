
# Generated: 2025-08-27T13:07:46.728920
# Source Brief: brief_00268.md
# Brief Index: 268

        
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
        "Controls: Arrow keys to move your ship. Press space to fire your laser. Survive for 60 seconds!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a ship in a top-down asteroid field, dodging and destroying asteroids for 60 seconds to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Player properties
        self.SHIP_SIZE = 12
        self.SHIP_SPEED = 5
        self.LASER_SPEED = 10
        self.LASER_COOLDOWN_FRAMES = 15 # ~4 shots per second

        # Asteroid properties
        self.NUM_ASTEROIDS = 3
        self.ASTEROID_MIN_SIZE = 20
        self.ASTEROID_MAX_SIZE = 40
        self.ASTEROID_MIN_SPEED = 1.0
        self.ASTEROID_SPEED_INCREASE_INTERVAL = self.FPS * 10 # every 10 seconds
        self.ASTEROID_SPEED_INCREASE_AMOUNT = 0.2

        # Reward structure
        self.REWARD_SURVIVAL_TICK = -0.01
        self.REWARD_ASTEROID_DESTROYED = 10
        self.REWARD_WIN = 100
        self.REWARD_PLAYER_DIED = -100

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (255, 255, 255)
        self.COLOR_LASER = (255, 80, 80)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_EXPLOSION = (255, 255, 180)

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
            self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 60)

        # Initialize state variables to be defined in reset()
        self.ship_pos = None
        self.asteroids = None
        self.lasers = None
        self.explosions = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.laser_cooldown = None
        self.prev_space_held = None
        self.base_asteroid_speed = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.laser_cooldown = 0
        self.prev_space_held = False
        self.base_asteroid_speed = self.ASTEROID_MIN_SPEED

        self.ship_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)

        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid(initial_spawn=True)

        self.lasers = []
        self.explosions = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward += self.REWARD_SURVIVAL_TICK

        # --- ACTION HANDLING ---
        movement = action[0]
        space_held = action[1] == 1

        # --- PLAYER MOVEMENT ---
        if movement == 1: self.ship_pos[1] -= self.SHIP_SPEED # Up
        if movement == 2: self.ship_pos[1] += self.SHIP_SPEED # Down
        if movement == 3: self.ship_pos[0] -= self.SHIP_SPEED # Left
        if movement == 4: self.ship_pos[0] += self.SHIP_SPEED # Right

        self.ship_pos[0] = np.clip(self.ship_pos[0], 0, self.WIDTH)
        self.ship_pos[1] = np.clip(self.ship_pos[1], 0, self.HEIGHT)

        # --- PLAYER FIRING ---
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

        is_fire_pressed = space_held and not self.prev_space_held
        if is_fire_pressed and self.laser_cooldown <= 0:
            # sfx: player_shoot.wav
            self.lasers.append(np.copy(self.ship_pos))
            self.laser_cooldown = self.LASER_COOLDOWN_FRAMES
        self.prev_space_held = space_held

        # --- UPDATE GAME ENTITIES ---
        self._update_lasers()
        reward_from_asteroids = self._update_asteroids()
        reward += reward_from_asteroids
        self._update_explosions()

        # --- COLLISION DETECTION (PLAYER) ---
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.ship_pos - asteroid["pos"])
            if dist < asteroid["size"] + self.SHIP_SIZE / 2:
                # sfx: player_explosion.wav
                self.game_over = True
                terminated = True
                reward += self.REWARD_PLAYER_DIED
                break

        # --- DIFFICULTY SCALING ---
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.base_asteroid_speed += self.ASTEROID_SPEED_INCREASE_AMOUNT

        # --- CHECK TERMINATION CONDITIONS ---
        if not terminated and self.steps >= self.MAX_STEPS:
            # sfx: win_jingle.wav
            terminated = True
            self.game_over = True
            reward += self.REWARD_WIN

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_lasers(self):
        new_lasers = []
        for laser_pos in self.lasers:
            laser_pos[1] -= self.LASER_SPEED # Lasers always travel up
            if laser_pos[1] > 0:
                new_lasers.append(laser_pos)
        self.lasers = new_lasers

    def _update_asteroids(self):
        reward = 0
        asteroids_to_keep = []
        lasers_to_keep = list(self.lasers)
        destroyed_lasers = set()

        for asteroid in self.asteroids:
            asteroid["pos"] += asteroid["vel"]
            asteroid["angle"] += asteroid["rot_speed"]

            hit = False
            for i, laser_pos in enumerate(lasers_to_keep):
                if i in destroyed_lasers: continue
                if np.linalg.norm(laser_pos - asteroid["pos"]) < asteroid["size"]:
                    # sfx: asteroid_explosion.wav
                    self.score += 10
                    reward += self.REWARD_ASTEROID_DESTROYED
                    self.explosions.append({"pos": asteroid["pos"], "life": 1.0, "max_size": asteroid["size"] * 1.5})
                    destroyed_lasers.add(i)
                    hit = True
                    break

            if hit:
                self._spawn_asteroid()
                continue

            if (asteroid["pos"][0] < -self.ASTEROID_MAX_SIZE or
                asteroid["pos"][0] > self.WIDTH + self.ASTEROID_MAX_SIZE or
                asteroid["pos"][1] < -self.ASTEROID_MAX_SIZE or
                asteroid["pos"][1] > self.HEIGHT + self.ASTEROID_MAX_SIZE):
                self._spawn_asteroid()
            else:
                asteroids_to_keep.append(asteroid)

        self.asteroids = asteroids_to_keep
        self.lasers = [laser for i, laser in enumerate(lasers_to_keep) if i not in destroyed_lasers]
        return reward

    def _update_explosions(self):
        self.explosions = [exp for exp in self.explosions if exp["life"] > 0]
        for exp in self.explosions:
            exp["life"] -= 0.05 # Fade out over 20 frames

    def _spawn_asteroid(self, initial_spawn=False):
        edge = self.np_random.integers(4)
        size = self.np_random.uniform(self.ASTEROID_MIN_SIZE, self.ASTEROID_MAX_SIZE)

        if initial_spawn:
            while True:
                pos = self.np_random.uniform([0, 0], [self.WIDTH, self.HEIGHT])
                if np.linalg.norm(pos - self.ship_pos) > 150:
                    break
        else:
            if edge == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -size])
            elif edge == 1: pos = np.array([self.WIDTH + size, self.np_random.uniform(0, self.HEIGHT)])
            elif edge == 2: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + size])
            else: pos = np.array([-size, self.np_random.uniform(0, self.HEIGHT)])

        center_target = np.array([self.WIDTH/2, self.HEIGHT/2]) + self.np_random.uniform(-150, 150, 2)
        direction = (center_target - pos)
        norm = np.linalg.norm(direction)
        if norm > 0: direction /= norm
        else: direction = np.array([0, -1])

        speed = self.base_asteroid_speed + self.np_random.uniform(-0.2, 0.2)
        vel = direction * max(0.1, speed)

        num_vertices = self.np_random.integers(6, 10)
        vertices = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            radius = size + self.np_random.uniform(-size * 0.2, size * 0.2)
            vertices.append((radius * math.cos(angle), radius * math.sin(angle)))

        self.asteroids.append({
            "pos": pos.astype(np.float32), "vel": vel.astype(np.float32),
            "size": size, "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.03, 0.03), "vertices": vertices
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for asteroid in self.asteroids:
            cos_a, sin_a = math.cos(asteroid["angle"]), math.sin(asteroid["angle"])
            points = []
            for x, y in asteroid["vertices"]:
                rot_x = x * cos_a - y * sin_a
                rot_y = x * sin_a + y * cos_a
                points.append((int(asteroid["pos"][0] + rot_x), int(asteroid["pos"][1] + rot_y)))
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)

        for laser_pos in self.lasers:
            x, y = int(laser_pos[0]), int(laser_pos[1])
            pygame.draw.line(self.screen, self.COLOR_LASER, (x, y), (x, y - 8), 3)
            pygame.gfxdraw.aacircle(self.screen, x, y - 4, 3, self.COLOR_LASER)

        if not self.game_over or self.steps >= self.MAX_STEPS:
            ship_points = [
                (self.ship_pos[0], self.ship_pos[1] - self.SHIP_SIZE),
                (self.ship_pos[0] - self.SHIP_SIZE / 1.5, self.ship_pos[1] + self.SHIP_SIZE / 2),
                (self.ship_pos[0] + self.SHIP_SIZE / 1.5, self.ship_pos[1] + self.SHIP_SIZE / 2)
            ]
            int_points = [(int(p[0]), int(p[1])) for p in ship_points]
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_SHIP)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_SHIP)

        for exp in self.explosions:
            current_radius = int(exp["max_size"] * (1.0 - exp["life"]))
            if current_radius > 0:
                pygame.gfxdraw.aacircle(self.screen, int(exp["pos"][0]), int(exp["pos"][1]), current_radius, self.COLOR_EXPLOSION)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    terminated = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        action = [movement, space_held, shift_held]

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        env.clock.tick(env.FPS)
    env.close()