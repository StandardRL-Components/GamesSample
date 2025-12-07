import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:45:50.915674
# Source Brief: brief_02469.md
# Brief Index: 2469
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    game_description = (
        "Navigate your ship through a perilous asteroid field. Survive as long as you can to reach your destination."
    )
    user_guide = (
        "Controls: Use ← and → arrow keys to steer your ship and avoid the asteroids."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_TIME = 90  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS
        self.WIN_TIME_STEPS = (self.MAX_TIME - 10) * self.FPS  # Win if you survive 80 seconds

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (220, 230, 255)
        self.COLOR_PLAYER_GLOW = (200, 220, 255)
        self.COLOR_ASTEROID = (120, 120, 130)
        self.COLOR_DESTINATION = (255, 255, 150)
        self.COLOR_SPARK = (255, 165, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Player constants
        self.PLAYER_Y_POS = self.HEIGHT - 80
        self.PLAYER_THRUST = 0.4
        self.PLAYER_DRAG = 0.95
        self.PLAYER_MAX_VEL_X = 7
        self.PLAYER_HITBOX_RADIUS = 10

        # Asteroid constants
        self.ASTEROID_MIN_RADIUS = 10
        self.ASTEROID_MAX_RADIUS = 40
        self.INITIAL_SPAWN_RATE = 0.2  # per second
        self.SPAWN_RATE_INCREASE = 0.1 # per 10 seconds

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.font_big = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_vel_x = 0.0
        self.hull = 100.0
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.player_trail = []
        self.world_y = 0.0
        self.asteroid_spawn_rate = 0.0
        self.last_action = [0, 0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = [self.WIDTH / 2, self.PLAYER_Y_POS]
        self.player_vel_x = 0.0
        self.hull = 100.0
        self.asteroids = []
        self.particles = []
        self.player_trail = []
        self.world_y = 0.0
        self.asteroid_spawn_rate = self.INITIAL_SPAWN_RATE
        self.last_action = [0, 0, 0]

        self._create_stars(200)

        obs = self._get_observation()
        if self.render_mode == "human":
            self._render_frame()
        return obs, self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.last_action = action
        reward = 0.0
        
        self._handle_input(action)
        self._update_player()
        self._update_world_scroll()
        self._update_asteroids()
        collision_reward = self._handle_collisions()
        reward += collision_reward
        self._update_particles()
        self._update_trail()

        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.asteroid_spawn_rate += self.SPAWN_RATE_INCREASE

        reward += 0.01  # Small reward for surviving a step

        terminated = False
        win = False
        if self.hull <= 0:
            terminated = True
            reward -= 50
            # Sound: Player explosion
            self._create_explosion(self.player_pos, 50, self.COLOR_SPARK)
        elif self.steps >= self.WIN_TIME_STEPS:
            terminated = True
            win = True
            reward += 100
            # Sound: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 25

        if terminated:
            self.game_over = True

        self.score += reward
        obs = self._get_observation()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.player_vel_x -= self.PLAYER_THRUST
        elif movement == 4:  # Right
            self.player_vel_x += self.PLAYER_THRUST

    def _update_player(self):
        self.player_vel_x *= self.PLAYER_DRAG
        self.player_vel_x = np.clip(self.player_vel_x, -self.PLAYER_MAX_VEL_X, self.PLAYER_MAX_VEL_X)
        self.player_pos[0] += self.player_vel_x
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)

    def _update_world_scroll(self):
        oscillation_period_steps = 2 * self.FPS
        base_speed = 8
        amplitude = 4
        scroll_speed = base_speed + amplitude * math.sin(2 * math.pi * self.steps / oscillation_period_steps)
        self.world_y += scroll_speed

    def _update_asteroids(self):
        spawn_chance = self.asteroid_spawn_rate / self.FPS
        if self.np_random.random() < spawn_chance:
            x = self.np_random.uniform(0, self.WIDTH)
            y = self.world_y - self.ASTEROID_MAX_RADIUS
            radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
            self.asteroids.append({'pos': [x, y], 'radius': radius})

        screen_bottom_world_y = self.world_y + self.HEIGHT + self.ASTEROID_MAX_RADIUS
        self.asteroids = [a for a in self.asteroids if a['pos'][1] < screen_bottom_world_y]

    def _handle_collisions(self):
        collision_reward = 0
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            asteroid_screen_y = asteroid['pos'][1] - self.world_y + self.HEIGHT
            dist = math.hypot(self.player_pos[0] - asteroid['pos'][0], self.player_pos[1] - asteroid_screen_y)
            if dist < asteroid['radius'] + self.PLAYER_HITBOX_RADIUS:
                if i not in asteroids_to_remove:
                    # Sound: Hull impact
                    damage = asteroid['radius'] * 0.5
                    self.hull -= damage
                    self.hull = max(0, self.hull)
                    penalty = 2.0 + (asteroid['radius'] / self.ASTEROID_MAX_RADIUS) * 8.0
                    collision_reward -= penalty
                    self._create_explosion(self.player_pos, int(asteroid['radius']), self.COLOR_SPARK)
                    asteroids_to_remove.append(i)
        
        if asteroids_to_remove:
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
        return collision_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _update_trail(self):
        self.player_trail.insert(0, {'pos': self.player_pos.copy(), 'life': 15})
        if len(self.player_trail) > 20:
            self.player_trail.pop()
        for t in self.player_trail:
            t['life'] -= 1
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_stars()
        self._render_destination_hint()
        self._render_asteroids()
        self._render_trail()
        self._render_player()
        self._render_particles()

    def _render_ui(self):
        # Hull Bar
        hull_pct = max(0, self.hull / 100.0)
        bar_color = (int(255 * (1 - hull_pct)), int(255 * hull_pct), 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, bar_color, (10, 10, int(200 * hull_pct), 20))
        pygame.draw.rect(self.screen, (200, 200, 200), (10, 10, 200, 20), 2)
        hull_text = self.font_small.render("HULL", True, self.COLOR_TEXT)
        self.screen.blit(hull_text, (15, 12))

        # Timer
        time_left = max(0, self.MAX_TIME - (self.steps / self.FPS))
        timer_text = self.font_big.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 15, 10))

        if self.game_over:
            win = self.hull > 0 and self.steps >= self.WIN_TIME_STEPS
            msg = "DESTINATION REACHED" if win else "MISSION FAILED"
            color = (100, 255, 100) if win else (255, 100, 100)
            msg_text = self.font_big.render(msg, True, color)
            self.screen.blit(msg_text, (self.WIDTH // 2 - msg_text.get_width() // 2, self.HEIGHT // 2 - msg_text.get_height() // 2))
    
    def _render_frame(self):
        if self.render_mode == "human":
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _render_stars(self):
        for star in self.stars:
            screen_y = (star['y'] - self.world_y * star['z']) % self.HEIGHT
            pygame.draw.circle(self.screen, star['color'], (int(star['x']), int(screen_y)), star['size'])

    def _render_destination_hint(self):
        if self.steps > self.WIN_TIME_STEPS - (10 * self.FPS):
            progress = np.clip((self.steps - (self.WIN_TIME_STEPS - (10 * self.FPS))) / (10 * self.FPS), 0, 1)
            size = int(5 + 30 * progress)
            alpha = int(50 + 200 * progress)
            pulse = 1 + 0.1 * math.sin(self.steps * 0.2)
            size = int(size * pulse)
            if size <= 0: return
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, self.COLOR_DESTINATION + (alpha,), (size, size), size)
            self.screen.blit(surf, (self.WIDTH//2 - size, 20), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            x, y = asteroid['pos']
            radius = int(asteroid['radius'])
            screen_y = y - self.world_y + self.HEIGHT
            if -radius < screen_y < self.HEIGHT + radius:
                pygame.gfxdraw.aacircle(self.screen, int(x), int(screen_y), radius, self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(screen_y), radius, self.COLOR_ASTEROID)

    def _render_trail(self):
        for t in self.player_trail:
            alpha = int(255 * (t['life'] / 15.0))
            if alpha > 0:
                s = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_PLAYER_GLOW + (alpha,), (5, 5), 3)
                self.screen.blit(s, (int(t['pos'][0]-5), int(t['pos'][1]-5)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_player(self):
        movement = self.last_action[0]
        if movement in [3, 4]:
            for _ in range(2):
                offset_x = 10 if movement == 3 else -10
                vel_x = self.np_random.uniform(2, 4) if movement == 3 else self.np_random.uniform(-4, -2)
                self.particles.append({
                    'pos': [self.player_pos[0] + offset_x, self.player_pos[1] + 5],
                    'vel': [vel_x, self.np_random.uniform(-0.5, 0.5)],
                    'life': self.np_random.integers(8, 15), 'max_life': 15,
                    'color': self.COLOR_SPARK, 'radius': self.np_random.uniform(1, 3)
                })

        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        p1, p2, p3 = (x, y - 15), (x - 10, y + 10), (x + 10, y + 10)
        
        glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        glow_points = [(p[0] - x + 30, p[1] - y + 30) for p in [p1, p2, p3]]
        pygame.draw.polygon(glow_surf, self.COLOR_PLAYER_GLOW + (50,), glow_points)
        pygame.draw.polygon(glow_surf, self.COLOR_PLAYER_GLOW + (30,), glow_points, width=5)
        self.screen.blit(glow_surf, (x-30, y-30), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p.get('max_life', p['life'] + 1)))
            radius = int(p['radius'])
            if radius <= 0 or alpha <=0: continue
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, p['color'] + (alpha,), (radius, radius), radius)
            self.screen.blit(s, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)

    def _create_stars(self, num_stars):
        self.stars = [{
            'x': self.np_random.uniform(0, self.WIDTH),
            'y': self.np_random.uniform(0, self.HEIGHT),
            'z': self.np_random.uniform(0.1, 0.6),
            'size': int(self.np_random.choice([1, 1, 1, 2])),
            'color': self.np_random.choice([(100,100,120), (150,150,180), (200,200,220)])
        } for _ in range(num_stars)]

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life, 'max_life': life, 'color': color,
                'radius': self.np_random.uniform(1, 4)
            })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "hull": self.hull}

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    env.reset(seed=42)
    env.validate_implementation()
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    print("Playing game manually. Use Left/Right arrow keys. Press Q to quit.")
    
    while not terminated:
        action = [0, 0, 0]  # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}, Hull: {info['hull']:.1f}")
            obs, info = env.reset()
            terminated = False # Remove this line to stop after one game
            
    env.close()