import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Use ↑/↓ to aim your jump, Space to jump, and Shift for a boost."
    game_description = "Guide your hopping spaceship through a treacherous asteroid field to reach the finish line before time runs out."
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (64, 224, 208)
    COLOR_PLAYER_GLOW = (128, 255, 240, 50)
    COLOR_ASTEROID = (130, 130, 150)
    COLOR_FINISH_LINE = (0, 255, 127)
    COLOR_UI_TEXT = (240, 240, 255)
    COLOR_AIM_INDICATOR = (255, 255, 255, 100)

    # Physics & Gameplay
    FPS = 30
    GRAVITY = 0.8
    GROUND_Y = SCREEN_HEIGHT - 40
    LEVEL_WIDTH = 8000
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Player
    PLAYER_X = 100
    PLAYER_RADIUS = 12
    JUMP_POWER_MIN = 10
    JUMP_POWER_MAX = 22
    JUMP_POWER_STEP = 0.5
    BOOST_MULTIPLIER = 1.3

    # Asteroids
    ASTEROID_MIN_RADIUS = 15
    ASTEROID_MAX_RADIUS = 40
    ASTEROID_INITIAL_SPEED = 3.0
    ASTEROID_SPEED_INCREASE_INTERVAL = 300
    ASTEROID_SPEED_INCREASE_AMOUNT = 0.2

    # Rewards
    REWARD_SURVIVAL = 0.01
    REWARD_FINISH = 50.0
    REWARD_CRASH = -20.0
    REWARD_NEAR_MISS = 5.0
    NEAR_MISS_DISTANCE = 40

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.player_pos = np.array([0.0, 0.0])
        self.player_vel_y = 0.0
        self.on_ground = True
        self.jump_power = 0.0
        self.last_space_held = False

        self.asteroids = []
        self.stars = []
        self.particles = []

        self.world_x = 0.0
        self.scroll_speed = 0.0

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([float(self.PLAYER_X), float(self.GROUND_Y)])
        self.player_vel_y = 0.0
        self.on_ground = True
        self.jump_power = (self.JUMP_POWER_MIN + self.JUMP_POWER_MAX) / 2
        self.last_space_held = False

        self.world_x = 0.0
        self.scroll_speed = self.ASTEROID_INITIAL_SPEED

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False

        self._generate_stars()
        self._generate_asteroids()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = self.REWARD_SURVIVAL
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_world()

            collision_reward, collision_termination = self._check_collisions()
            reward += collision_reward
            if collision_termination:
                self.game_over = True
                # sfx: explosion

            finish_reward, finish_termination = self._check_finish()
            reward += finish_reward
            if finish_termination:
                self.game_over = True
                self.game_won = True
                # sfx: victory fanfare

        self._update_particles()
        self.steps += 1

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            # sfx: timeout buzzer

        terminated = self.game_over
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Adjust jump power
        if movement == 1:  # Up
            self.jump_power = min(self.JUMP_POWER_MAX, self.jump_power + self.JUMP_POWER_STEP)
        elif movement == 2:  # Down
            self.jump_power = max(self.JUMP_POWER_MIN, self.jump_power - self.JUMP_POWER_STEP)

        # Initiate jump on space press (rising edge)
        if space_held and not self.last_space_held and self.on_ground:
            power = self.jump_power * (self.BOOST_MULTIPLIER if shift_held else 1.0)
            self.player_vel_y = -power
            self.on_ground = False
            # sfx: jump_sound
            if shift_held:
                # sfx: boost_sound
                self._spawn_particle_burst(self.player_pos, 20, (255, 200, 50), 3, 5, -math.pi / 2, math.pi / 4)

        self.last_space_held = space_held

    def _update_player(self):
        if not self.on_ground:
            self.player_vel_y += self.GRAVITY
            self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] >= self.GROUND_Y:
            if not self.on_ground:
                # sfx: land_sound
                self._spawn_particle_burst(self.player_pos, 10, (200, 200, 200), 1, 2, -math.pi, math.pi)
            self.player_pos[1] = self.GROUND_Y
            self.player_vel_y = 0
            self.on_ground = True

        # Clamp to top of screen
        self.player_pos[1] = max(self.PLAYER_RADIUS, self.player_pos[1])

    def _update_world(self):
        # Increase difficulty
        if self.steps > 0 and self.steps % self.ASTEROID_SPEED_INCREASE_INTERVAL == 0:
            self.scroll_speed += self.ASTEROID_SPEED_INCREASE_AMOUNT

        self.world_x += self.scroll_speed

        # Scroll stars
        for star in self.stars:
            star['pos'][0] -= self.scroll_speed * star['depth']
        self.stars = [s for s in self.stars if s['pos'][0] > -5]

        # Scroll asteroids
        for asteroid in self.asteroids:
            asteroid['pos'][0] -= self.scroll_speed
        self.asteroids = [a for a in self.asteroids if a['pos'][0] > -a['radius']]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_collisions(self):
        reward = 0.0
        terminated = False
        player_circle = (self.player_pos, self.PLAYER_RADIUS)

        min_dist_sq = float('inf')

        for asteroid in self.asteroids:
            asteroid_circle = (asteroid['pos'], asteroid['radius'])
            dist_sq = (player_circle[0][0] - asteroid_circle[0][0]) ** 2 + (
                        player_circle[0][1] - asteroid_circle[0][1]) ** 2

            min_dist_sq = min(min_dist_sq, dist_sq)

            # Collision
            if dist_sq < (player_circle[1] + asteroid_circle[1]) ** 2:
                self._spawn_particle_burst(self.player_pos, 50, (255, 100, 50), 2, 8)
                return self.REWARD_CRASH, True

            # Near miss reward
            if not asteroid.get('near_missed', False):
                near_miss_rad = player_circle[1] + asteroid_circle[1] + self.NEAR_MISS_DISTANCE
                if dist_sq < near_miss_rad ** 2:
                    reward += self.REWARD_NEAR_MISS
                    asteroid['near_missed'] = True
                    # sfx: near_miss_whoosh
                    self._spawn_particle_burst(asteroid['pos'], 15, (255, 255, 0), 1, 3, angle_min=-math.pi / 2,
                                               angle_max=math.pi / 2)

        return reward, terminated

    def _check_finish(self):
        if self.world_x >= self.LEVEL_WIDTH:
            return self.REWARD_FINISH, True
        return 0.0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            pos = (int(star['pos'][0]), int(star['pos'][1]))
            size = int(star['size'] * star['depth'])
            if size > 0:
                color_val = int(100 + 155 * star['depth'])
                color = (color_val, color_val, color_val)
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_game(self):
        # Ground
        pygame.draw.line(self.screen, self.COLOR_ASTEROID, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)

        # Finish line
        finish_x = self.LEVEL_WIDTH - self.world_x
        if 0 < finish_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, 0), (finish_x, self.SCREEN_HEIGHT), 5)
            for y in range(0, self.SCREEN_HEIGHT, 20):
                pygame.draw.rect(self.screen, (0, 0, 0), (finish_x - 5, y, 10, 10))
                pygame.draw.rect(self.screen, (0, 0, 0), (finish_x - 5, y + 10, 10, 10))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)

        # Asteroids
        for asteroid in self.asteroids:
            pos = (int(asteroid['pos'][0]), int(asteroid['pos'][1]))
            radius = int(asteroid['radius'])
            if pos[0] + radius > 0 and pos[0] - radius < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ASTEROID)

        # Player
        if not (self.game_over and not self.game_won):
            self._render_player()

        # Aim indicator
        if self.on_ground:
            aim_length = 20 + (self.jump_power - self.JUMP_POWER_MIN) * 2
            end_pos = (self.player_pos[0], self.player_pos[1] - aim_length)
            pygame.draw.line(self.screen, self.COLOR_AIM_INDICATOR, (int(self.player_pos[0]), int(self.player_pos[1])),
                             (int(end_pos[0]), int(end_pos[1])), 2)

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))

        # Glow effect
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Main body
        points = [
            (pos[0], pos[1] - self.PLAYER_RADIUS),
            (pos[0] - self.PLAYER_RADIUS * 0.8, pos[1] + self.PLAYER_RADIUS * 0.6),
            (pos[0] + self.PLAYER_RADIUS * 0.8, pos[1] + self.PLAYER_RADIUS * 0.6),
        ]
        pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]),
                                     int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]),
                                int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.game_over:
            msg = "LEVEL COMPLETE" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH_LINE if self.game_won else (255, 80, 80)
            over_text = self.font_game_over.render(msg, True, color)
            over_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "world_x": self.world_x,
            "scroll_speed": self.scroll_speed,
        }

    def _generate_stars(self):
        self.stars.clear()
        for _ in range(200):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)],
                'size': self.np_random.integers(1, 3),
                'depth': self.np_random.uniform(0.1, 0.6)
            })

    def _generate_asteroids(self):
        self.asteroids.clear()
        current_x = 400
        while current_x < self.LEVEL_WIDTH:
            gap_height = self.np_random.integers(self.PLAYER_RADIUS * 4, self.PLAYER_RADIUS * 8)
            gap_y = self.np_random.integers(self.PLAYER_RADIUS, self.GROUND_Y - gap_height)

            num_asteroids = self.np_random.integers(3, 7)
            for _ in range(num_asteroids):
                radius = self.np_random.integers(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)

                # Define potential spawn regions
                spawn_region_above = (radius, gap_y - radius)
                spawn_region_below = (gap_y + gap_height + radius, self.GROUND_Y - radius)

                # Check if regions are valid (low < high)
                can_spawn_above = spawn_region_above[1] > spawn_region_above[0]
                can_spawn_below = spawn_region_below[1] > spawn_region_below[0]

                y = None
                # Try to spawn in a randomly chosen valid region
                if self.np_random.random() > 0.5:
                    if can_spawn_above:
                        y = self.np_random.uniform(spawn_region_above[0], spawn_region_above[1])
                    elif can_spawn_below:
                        y = self.np_random.uniform(spawn_region_below[0], spawn_region_below[1])
                else:
                    if can_spawn_below:
                        y = self.np_random.uniform(spawn_region_below[0], spawn_region_below[1])
                    elif can_spawn_above:
                        y = self.np_random.uniform(spawn_region_above[0], spawn_region_above[1])

                # If a valid spawn position was found, create the asteroid
                if y is not None:
                    x_offset = self.np_random.uniform(-150, 150)
                    self.asteroids.append({
                        'pos': np.array([current_x + x_offset, y]),
                        'radius': radius,
                        'near_missed': False,
                    })

            current_x += self.np_random.integers(400, 600)

    def _spawn_particle(self, pos, vel, life, size, color):
        self.particles.append({
            'pos': pos.copy(),
            'vel': vel.copy(),
            'life': life,
            'max_life': life,
            'size': size,
            'color': color,
        })

    def _spawn_particle_burst(self, pos, count, color, min_speed, max_speed, angle_min=-math.pi, angle_max=math.pi):
        for _ in range(count):
            angle = self.np_random.uniform(angle_min, angle_max)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(10, 25)
            size = self.np_random.uniform(1, 4)
            self._spawn_particle(pos, vel, life, size, color)

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
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)

    terminated = False
    total_reward = 0.0

    while not terminated:
        keys = pygame.key.get_pressed()

        # Map keyboard to MultiDiscrete action
        movement = 0  # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        # Left/Right are unused in this game

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                terminated = False  # to allow restart loop

    print(f"Game Over! Final Score: {total_reward:.2f}")
    pygame.quit()