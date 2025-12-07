import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to jump. Hold space for a power jump. Hold shift for better air control."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hop between procedurally generated platforms to reach the top. Collect coins for points but don't fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG_TOP = (0, 0, 32)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50)
    COLOR_PLATFORM = (128, 128, 128)
    COLOR_COIN = (255, 215, 0)
    COLOR_BONUS_COIN = (255, 105, 180)
    COLOR_TEXT = (255, 255, 255)

    # Physics
    GRAVITY = 0.5
    JUMP_VELOCITY = -10
    POWER_JUMP_VELOCITY = -14
    HORIZONTAL_JUMP_VELOCITY = 6
    DOWN_STOMP_VELOCITY = 8
    AIR_CONTROL_FORCE = 0.6
    MAX_VEL_X = 7
    MAX_VEL_Y = 15
    FRICTION = 0.9

    # Game Mechanics
    MAX_STEPS = 5000
    WIN_Y_THRESHOLD = 20
    PLATFORM_COUNT = 10
    PLATFORM_MIN_WIDTH = 50
    PLATFORM_MAX_WIDTH = 120
    PLATFORM_HEIGHT = 10
    PLATFORM_Y_GAP = 70
    INITIAL_SCROLL_SPEED = 1.0
    SCROLL_ACCELERATION = 0.00004  # speed increases by 0.02 every 500 steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state attributes are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hopper = {}
        self.platforms = deque()
        self.coins = []
        self.particles = []
        self.platform_scroll_speed = self.INITIAL_SCROLL_SPEED
        self.np_random = None
        self.target_platform = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_scroll_speed = self.INITIAL_SCROLL_SPEED

        self.hopper = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 100], dtype=float),
            "vel": np.array([0.0, 0.0], dtype=float),
            "size": np.array([20, 20], dtype=int),
            "on_ground": True,
            "squash": 1.0
        }

        self.platforms = deque()
        self.coins = []
        self.particles = []

        # Create initial platforms
        start_platform = self._create_platform(
            y=self.hopper["pos"][1] + self.hopper["size"][1],
            width=self.SCREEN_WIDTH,
            is_start=True
        )
        self.platforms.append(start_platform)

        y = start_platform['rect'].y - self.PLATFORM_Y_GAP
        for _ in range(self.PLATFORM_COUNT):
            new_platform = self._create_platform(y)
            self.platforms.append(new_platform)
            self._spawn_coins_on_platform(new_platform)
            y -= self.PLATFORM_Y_GAP

        self.target_platform = self.platforms[1]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Physics ---
        self._update_hopper_physics(shift_held)

        # --- Update World ---
        self._scroll_world()

        # --- Check Collisions & Collections ---
        landing_reward = self._handle_collisions()
        collection_reward = self._handle_collections()
        reward += landing_reward + collection_reward

        # --- Update Particles ---
        self._update_particles()

        # --- Update Game State ---
        self.steps += 1
        self.platform_scroll_speed += self.SCROLL_ACCELERATION

        # --- Calculate Reward & Check Termination ---
        terminated = False

        # Survival reward
        reward += 0.1

        # Distance penalty
        if self.hopper["on_ground"] and self.target_platform:
            dist = abs(self.hopper["pos"][0] - self.target_platform['rect'].centerx)
            reward -= 0.02 * (dist / (self.SCREEN_WIDTH / 2))  # Normalize penalty

        # Check for termination
        if self.hopper["pos"][1] > self.SCREEN_HEIGHT + self.hopper["size"][1]:
            terminated = True
            reward -= 10  # Fall penalty
        elif self.platforms and self.platforms[-1]['is_top'] and self.hopper['pos'][1] < self.WIN_Y_THRESHOLD:
            terminated = True
            reward += 100  # Win bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if self.hopper["on_ground"]:
            self.hopper["vel"][0] = 0  # Reset horizontal velocity on jump
            jump_vel = self.POWER_JUMP_VELOCITY if space_held else self.JUMP_VELOCITY

            if movement == 1:  # Up
                self.hopper["vel"][1] = jump_vel
                self.hopper["on_ground"] = False
                self._spawn_particles(self.hopper["pos"] + [0, self.hopper["size"][1] / 2], 10, self.COLOR_PLAYER, (0, 1))
            elif movement == 2:  # Down
                self.hopper["vel"][1] = self.DOWN_STOMP_VELOCITY
                self.hopper["on_ground"] = False
            elif movement == 3:  # Left
                self.hopper["vel"][1] = jump_vel * 0.8
                self.hopper["vel"][0] = -self.HORIZONTAL_JUMP_VELOCITY
                self.hopper["on_ground"] = False
                self._spawn_particles(self.hopper["pos"] + [0, self.hopper["size"][1] / 2], 10, self.COLOR_PLAYER, (0, 1))
            elif movement == 4:  # Right
                self.hopper["vel"][1] = jump_vel * 0.8
                self.hopper["vel"][0] = self.HORIZONTAL_JUMP_VELOCITY
                self.hopper["on_ground"] = False
                self._spawn_particles(self.hopper["pos"] + [0, self.hopper["size"][1] / 2], 10, self.COLOR_PLAYER, (0, 1))

            if not self.hopper["on_ground"]:
                self.hopper["squash"] = 1.5  # Stretch on jump
                # sfx: jump

    def _update_hopper_physics(self, shift_held):
        # Apply air control if shift is held
        if not self.hopper["on_ground"] and shift_held:
            if self.hopper["vel"][0] > 0: self.hopper["vel"][0] += self.AIR_CONTROL_FORCE
            if self.hopper["vel"][0] < 0: self.hopper["vel"][0] -= self.AIR_CONTROL_FORCE

        # Apply gravity
        if not self.hopper["on_ground"]:
            self.hopper["vel"][1] += self.GRAVITY

        # Apply friction
        self.hopper["vel"][0] *= self.FRICTION

        # Clamp velocity
        self.hopper["vel"][0] = np.clip(self.hopper["vel"][0], -self.MAX_VEL_X, self.MAX_VEL_X)
        self.hopper["vel"][1] = np.clip(self.hopper["vel"][1], -self.MAX_VEL_Y, self.MAX_VEL_Y)

        # Update position
        self.hopper["pos"] += self.hopper["vel"]

        # Horizontal screen bounds
        if self.hopper["pos"][0] < 0:
            self.hopper["pos"][0] = 0
            self.hopper["vel"][0] = 0
        if self.hopper["pos"][0] > self.SCREEN_WIDTH - self.hopper["size"][0]:
            self.hopper["pos"][0] = self.SCREEN_WIDTH - self.hopper["size"][0]
            self.hopper["vel"][0] = 0

        # Squash and stretch animation
        self.hopper["squash"] += (1.0 - self.hopper["squash"]) * 0.2

    def _scroll_world(self):
        for p in self.platforms:
            p['rect'].y += self.platform_scroll_speed
        for c in self.coins:
            c['pos'][1] += self.platform_scroll_speed

        if self.platforms and self.platforms[0]['rect'].top > self.SCREEN_HEIGHT:
            self.platforms.popleft()

            # Check if we need to add a new top platform
            if not any(p['is_top'] for p in self.platforms):
                is_top = len(self.platforms) > 30  # Make the top platform appear after some progress
                y = self.platforms[-1]['rect'].y - self.PLATFORM_Y_GAP
                new_platform = self._create_platform(y, is_top=is_top)
                self.platforms.append(new_platform)
                if not is_top:
                    self._spawn_coins_on_platform(new_platform)

    def _handle_collisions(self):
        self.hopper["on_ground"] = False
        player_rect = pygame.Rect(tuple(self.hopper["pos"]), tuple(self.hopper["size"]))

        for i, p in enumerate(self.platforms):
            if player_rect.colliderect(p['rect']) and self.hopper["vel"][1] > 0:
                # Check if player was above the platform in the last frame
                if self.hopper["pos"][1] + self.hopper["size"][1] - self.hopper["vel"][1] <= p['rect'].top:
                    self.hopper["pos"][1] = p['rect'].top - self.hopper["size"][1]
                    self.hopper["vel"][1] = 0
                    self.hopper["on_ground"] = True
                    self.hopper["squash"] = 0.5  # Squash on landing
                    self._spawn_particles(self.hopper["pos"] + [self.hopper["size"][0] / 2, self.hopper["size"][1]], 15,
                                          self.COLOR_PLATFORM)
                    # sfx: land

                    # Update target platform to the next one above
                    if i + 1 < len(self.platforms):
                        self.target_platform = self.platforms[i + 1]

                    return 0  # No specific reward for just landing
        return 0

    def _handle_collections(self):
        reward = 0
        player_rect = pygame.Rect(tuple(self.hopper["pos"]), tuple(self.hopper["size"]))

        uncollected_coins = []
        for coin in self.coins:
            coin_rect = pygame.Rect(coin['pos'][0] - coin['radius'], coin['pos'][1] - coin['radius'],
                                     coin['radius'] * 2, coin['radius'] * 2)
            if player_rect.colliderect(coin_rect):
                # This coin is collected
                if coin['is_bonus']:
                    reward += 5
                    self.score += 5
                    self._spawn_particles(coin['pos'], 20, self.COLOR_BONUS_COIN, life=30)
                else:
                    reward += 1
                    self.score += 1
                    self._spawn_particles(coin['pos'], 15, self.COLOR_COIN, life=20)
                # sfx: coin_get
            else:
                # This coin remains
                uncollected_coins.append(coin)
        self.coins = uncollected_coins
        return reward

    def _create_platform(self, y, width=None, is_start=False, is_top=False):
        if width is None:
            width = self.np_random.integers(self.PLATFORM_MIN_WIDTH, self.PLATFORM_MAX_WIDTH + 1)

        if is_top:
            width = self.SCREEN_WIDTH / 3

        x = self.np_random.integers(0, self.SCREEN_WIDTH - width + 1)
        if is_start:
            x = (self.SCREEN_WIDTH - width) / 2

        return {
            'rect': pygame.Rect(x, y, width, self.PLATFORM_HEIGHT),
            'is_bonus_risk': width < self.PLATFORM_MIN_WIDTH * 1.3 and not is_start,
            'is_top': is_top
        }

    def _spawn_coins_on_platform(self, platform):
        if self.np_random.random() < 0.7:  # 70% chance to have a coin
            is_bonus = platform['is_bonus_risk'] and self.np_random.random() < 0.5
            color = self.COLOR_BONUS_COIN if is_bonus else self.COLOR_COIN
            radius = 8 if is_bonus else 5

            coin_x = platform['rect'].x + platform['rect'].width / 2
            coin_y = platform['rect'].y - radius - 2

            self.coins.append({
                'pos': np.array([coin_x, coin_y]),
                'radius': radius,
                'is_bonus': is_bonus,
                'color': color
            })

    def _spawn_particles(self, pos, count, color, vel_dir=(0, -1), life=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            if vel_dir[1] == 1:  # Downward burst
                vel[1] = abs(vel[1])
            elif vel_dir[1] == -1:  # Upward burst
                vel[1] = -abs(vel[1])

            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += self.GRAVITY * 0.1  # Particles have less gravity
            p['life'] -= 1
            if p['life'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Simple gradient
        self.screen.fill(self.COLOR_BG_BOTTOM)
        top_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, top_rect, 0)
        # A more complex gradient was here, but it's slow. This is faster.
        # For a true gradient, you can pre-render it to a surface.
        # For now, a solid color or simple rect is sufficient.
        # Let's revert to the original gradient as performance is not the issue
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))


    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                try:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha)
                    )
                except TypeError: # Sometimes the color tuple might not have alpha
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], 255)
                    )


        # Render platforms
        for p in self.platforms:
            color = self.COLOR_PLATFORM
            if p['is_top']:
                color = self.COLOR_COIN  # Make top platform golden
            pygame.draw.rect(self.screen, color, p['rect'], border_radius=3)

        # Render coins
        for c in self.coins:
            pygame.gfxdraw.filled_circle(self.screen, int(c['pos'][0]), int(c['pos'][1]), c['radius'], c['color'])
            pygame.gfxdraw.aacircle(self.screen, int(c['pos'][0]), int(c['pos'][1]), c['radius'], c['color'])

        # Render player
        squashed_size = (self.hopper["size"][0] / self.hopper["squash"], self.hopper["size"][1] * self.hopper["squash"])
        squashed_pos = (
            self.hopper["pos"][0] + (self.hopper["size"][0] - squashed_size[0]) / 2,
            self.hopper["pos"][1] + (self.hopper["size"][1] - squashed_size[1])
        )
        player_rect = pygame.Rect(squashed_pos, squashed_size)

        # Glow effect
        glow_size = max(0, player_rect.width * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_size / 2, glow_size / 2), glow_size / 2)
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size / 2, player_rect.centery - glow_size / 2),
                         special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        steps_text = self.font_large.render(f"TIME: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Arcade Hopper")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        # --- Action mapping for human play ---
        movement = 0  # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            print("Press 'R' to restart.")

        # --- Control frame rate ---
        clock.tick(GameEnv.TARGET_FPS)

    env.close()