
# Generated: 2025-08-28T06:28:40.550703
# Source Brief: brief_02919.md
# Brief Index: 2919

        
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
    user_guide = "Controls: Use arrow keys (↑, ↓, ←, →) to swipe and slice falling fruit. Avoid the bombs!"

    # Must be a short, user-facing description of the game:
    game_description = "A fast-paced arcade game where you slice falling fruit to score points while avoiding bombs. Hit three bombs and it's game over."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_SCORE = 1000
        self.MAX_BOMBS = 3
        self.MAX_STEPS = self.FPS * 35  # ~35 seconds max episode length

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (40, 50, 70)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_BOMB = (30, 30, 30)
        self.COLOR_BOMB_FUSE = (255, 200, 0)
        self.COLOR_SWIPE = (255, 255, 255)
        self.FRUIT_COLORS = {
            "apple": (220, 40, 40),
            "orange": (240, 140, 20),
            "lemon": (250, 250, 50),
            "lime": (50, 220, 50),
        }

        # Gameplay parameters
        self.FRUIT_RADIUS = 15
        self.BOMB_RADIUS = 18
        self.SWIPE_LENGTH = 250
        self.SWIPE_DURATION = 5  # frames
        self.GRAVITY = 0.3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False
        self.fruits = []
        self.bombs = []
        self.particles = []
        self.swipe_trails = []
        self.fruit_spawn_prob = 0.0
        self.bomb_spawn_prob = 0.0
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.bomb_hits = 0
        self.game_over = False

        self.fruits = []
        self.bombs = []
        self.particles = []
        self.swipe_trails = []

        self.fruit_spawn_prob = 0.03
        self.bomb_spawn_prob = 0.01

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        self.steps += 1

        # --- 1. Unpack Action & Handle Swipe ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement > 0:  # A swipe action was taken
            # Sound: Play swipe sound
            center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
            swipe_start = np.array([center_x, center_y], dtype=float)
            swipe_vec = np.array([0, 0], dtype=float)

            if movement == 1: swipe_vec = np.array([0, -1])  # Up
            elif movement == 2: swipe_vec = np.array([0, 1])   # Down
            elif movement == 3: swipe_vec = np.array([-1, 0])  # Left
            elif movement == 4: swipe_vec = np.array([1, 0])   # Right

            swipe_end = swipe_start + swipe_vec * self.SWIPE_LENGTH
            self.swipe_trails.append({
                "start": swipe_start, "end": swipe_end, "life": self.SWIPE_DURATION
            })

            # --- Check for collisions with fruits ---
            sliced_fruits = []
            for fruit in self.fruits:
                if self._check_line_circle_collision(swipe_start, swipe_end, fruit['pos'], fruit['radius']):
                    sliced_fruits.append(fruit)
                    # Sound: Play fruit slice sound
                    reward += 10
                    self.score += 10
                    self._create_particles(fruit['pos'], fruit['color'], 20)
            self.fruits = [f for f in self.fruits if f not in sliced_fruits]

            # --- Check for collisions with bombs ---
            hit_bombs = []
            for bomb in self.bombs:
                if self._check_line_circle_collision(swipe_start, swipe_end, bomb['pos'], bomb['radius']):
                    hit_bombs.append(bomb)
                    # Sound: Play explosion sound
                    reward -= 50
                    self.bomb_hits += 1
                    self._create_particles(bomb['pos'], self.COLOR_BOMB_FUSE, 40, is_explosion=True)
            self.bombs = [b for b in self.bombs if b not in hit_bombs]

        # --- 2. Update Game State ---
        self._update_difficulty()
        self._update_spawns()
        self._update_entities()

        # --- 3. Check for Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
        if self.bomb_hits >= self.MAX_BOMBS:
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_difficulty(self):
        self.bomb_spawn_prob = 0.01 + 0.001 * (self.steps // 100)
        self.fruit_spawn_prob = 0.03 + 0.002 * (self.steps // 100)

    def _update_spawns(self):
        if self.np_random.random() < self.fruit_spawn_prob:
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            self.fruits.append({
                "pos": np.array([self.np_random.uniform(50, self.WIDTH - 50), -self.FRUIT_RADIUS], dtype=float),
                "vel": np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 5)], dtype=float),
                "radius": self.FRUIT_RADIUS,
                "color": self.FRUIT_COLORS[fruit_type],
            })

        if self.np_random.random() < self.bomb_spawn_prob:
            self.bombs.append({
                "pos": np.array([self.np_random.uniform(50, self.WIDTH - 50), -self.BOMB_RADIUS], dtype=float),
                "vel": np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(2, 4)], dtype=float),
                "radius": self.BOMB_RADIUS,
            })

    def _update_entities(self):
        # Update fruits & bombs
        for item in self.fruits + self.bombs:
            item['vel'][1] += self.GRAVITY
            item['pos'] += item['vel']
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.HEIGHT + f['radius']]
        self.bombs = [b for b in self.bombs if b['pos'][1] < self.HEIGHT + b['radius']]

        # Update particles
        for p in self.particles:
            p['vel'][1] += self.GRAVITY * 0.5
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Update swipe trails
        for t in self.swipe_trails:
            t['life'] -= 1
        self.swipe_trails = [t for t in self.swipe_trails if t['life'] > 0]

    def _check_line_circle_collision(self, p1, p2, circle_center, r):
        v_p1_c = circle_center - p1
        v_p1_p2 = p2 - p1
        len_sq = np.dot(v_p1_p2, v_p1_p2)
        if len_sq == 0:
            return np.linalg.norm(v_p1_c) <= r
        t = max(0, min(1, np.dot(v_p1_c, v_p1_p2) / len_sq))
        closest_point = p1 + t * v_p1_p2
        return np.linalg.norm(circle_center - closest_point) <= r

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            if is_explosion:
                speed = self.np_random.uniform(2, 8)
                life = self.np_random.integers(20, 40)
                radius = self.np_random.uniform(2, 5)
                p_color = self.np_random.choice([(255, 200, 0), (255, 100, 0), (200, 20, 20)])
            else:
                speed = self.np_random.uniform(1, 5)
                life = self.np_random.integers(15, 30)
                radius = self.np_random.uniform(1, 3)
                p_color = color
            
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": life, "max_life": life,
                "radius": radius, "color": p_color
            })

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = tuple(int(self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp) for i in range(3))
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*p['color'], alpha))

        # Render swipe trails
        for t in self.swipe_trails:
            alpha = int(255 * (t['life'] / self.SWIPE_DURATION))
            width = int(10 * (t['life'] / self.SWIPE_DURATION))
            if width > 0:
                pygame.draw.aaline(self.screen, self.COLOR_SWIPE, t['start'], t['end'], alpha // 255)

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], (0, 0, 0, 50))

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], bomb['radius'], self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], bomb['radius'], (255, 255, 255, 50))
            fuse_end_x = pos[0] + 5
            fuse_end_y = pos[1] - bomb['radius']
            pygame.draw.line(self.screen, (100, 80, 80), (pos[0], pos[1] - bomb['radius'] + 5), (fuse_end_x, fuse_end_y), 3)
            spark_pos = (fuse_end_x + self.np_random.uniform(-1, 1), fuse_end_y + self.np_random.uniform(-1, 1))
            pygame.draw.circle(self.screen, self.COLOR_BOMB_FUSE, spark_pos, 3)

    def _render_ui(self):
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        bomb_icon_radius = 12
        for i in range(self.MAX_BOMBS):
            pos_x = self.WIDTH - 30 - i * (bomb_icon_radius * 2.5)
            pos_y = 25
            color = self.COLOR_BOMB if i < self.bomb_hits else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, int(pos_x), pos_y, bomb_icon_radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos_x), pos_y, bomb_icon_radius, self.COLOR_TEXT)

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
        obs, info = self.reset(seed=42)
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")

    running = True
    total_reward = 0
    action = env.action_space.noop()

    while running:
        movement_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement_action = 1
                elif event.key == pygame.K_DOWN: movement_action = 2
                elif event.key == pygame.K_LEFT: movement_action = 3
                elif event.key == pygame.K_RIGHT: movement_action = 4

        action = [movement_action, 0, 0] if movement_action > 0 else env.action_space.noop()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

    env.close()