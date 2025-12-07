
# Generated: 2025-08-27T16:34:57.516404
# Source Brief: brief_01266.md
# Brief Index: 1266

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ←→ to move, ↑ to jump. Collect coins and reach the green exit."

    # Must be a short, user-facing description of the game:
    game_description = "Guide a robot through a procedurally generated side-scrolling platformer, collecting coins and reaching the exit before time runs out."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLATFORM = (100, 100, 120)
        self.COLOR_ROBOT = (60, 120, 255)
        self.COLOR_ROBOT_JUMP = (150, 200, 255)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_PARTICLE = (150, 150, 150)

        # Game constants
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10.5
        self.MOVE_SPEED = 5
        self.MAX_VEL_Y = 15
        self.ROBOT_SIZE = (24, 32)
        self.COIN_SIZE = 16
        self.MAX_STEPS = 1000
        self.TIME_LIMIT_SECONDS = 60

        # State variables (will be reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_message = ""
        self.time_left = 0
        self.robot_pos = [0.0, 0.0]
        self.robot_vel = [0.0, 0.0]
        self.on_ground = False
        self.can_jump = True
        self.camera_x = 0.0
        self.platforms = []
        self.coins = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.particles = []
        self.starfield = []

        # Initialize state variables
        self.reset()
        # self.validate_implementation() # For development; commented out for submission

    def _generate_starfield(self):
        self.starfield = []
        for _ in range(150):
            self.starfield.append({
                'pos': [random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)],
                'speed': random.uniform(0.1, 0.5),
                'color': random.randint(150, 220)
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_message = ""
        self.time_left = self.TIME_LIMIT_SECONDS * 30  # 30 FPS

        # Reset robot
        self.robot_pos = [50.0, 200.0]
        self.robot_vel = [0.0, 0.0]
        self.on_ground = False
        self.can_jump = True

        # Procedurally generate a finite level
        self.platforms = []
        self.coins = []
        self.platforms.append(pygame.Rect(-10, 350, 200, 50))
        current_x = 190.0
        platform_y = 350.0
        level_length_platforms = 30

        for _ in range(level_length_platforms):
            gap = self.np_random.uniform(40, 80)
            platform_width = self.np_random.uniform(80, 200)
            current_x += gap

            platform_y += self.np_random.uniform(-60, 60)
            platform_y = np.clip(platform_y, 200, self.SCREEN_HEIGHT - 50)

            platform_rect = pygame.Rect(current_x, platform_y, platform_width, self.SCREEN_HEIGHT - platform_y)
            self.platforms.append(platform_rect)

            if self.np_random.random() < 0.7:
                coin_x = current_x + platform_width / 2 - self.COIN_SIZE / 2
                coin_y = platform_y - 40 - self.np_random.uniform(0, 20)
                self.coins.append(pygame.Rect(coin_x, coin_y, self.COIN_SIZE, self.COIN_SIZE))

            current_x += platform_width

        self.exit_rect = pygame.Rect(current_x + 50, platform_y - 50, 40, 50)

        self.camera_x = 0.0
        self.particles = []
        if not self.starfield:
            self._generate_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0

        # Update game state
        self.steps += 1
        self.time_left -= 1

        # Handle player input
        target_vel_x = 0.0
        if movement == 3:  # Left
            target_vel_x = -self.MOVE_SPEED
            reward -= 0.02
        elif movement == 4:  # Right
            target_vel_x = self.MOVE_SPEED
            reward += 0.1

        self.robot_vel[0] += (target_vel_x - self.robot_vel[0]) * 0.25

        if movement == 1 and self.on_ground and self.can_jump:
            self.robot_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            self.can_jump = False
            # sfx: jump
        if movement != 1:
            self.can_jump = True

        # Apply physics
        self.robot_vel[1] += self.GRAVITY
        self.robot_vel[1] = min(self.robot_vel[1], self.MAX_VEL_Y)

        # Collision detection and resolution
        self.robot_pos[0] += self.robot_vel[0]
        robot_rect = pygame.Rect(int(self.robot_pos[0]), int(self.robot_pos[1]), *self.ROBOT_SIZE)
        for platform in self.platforms:
            if robot_rect.colliderect(platform):
                if self.robot_vel[0] > 0:
                    robot_rect.right = platform.left
                elif self.robot_vel[0] < 0:
                    robot_rect.left = platform.right
                self.robot_pos[0] = float(robot_rect.x)
                self.robot_vel[0] = 0

        self.on_ground = False
        self.robot_pos[1] += self.robot_vel[1]
        robot_rect = pygame.Rect(int(self.robot_pos[0]), int(self.robot_pos[1]), *self.ROBOT_SIZE)
        for platform in self.platforms:
            if robot_rect.colliderect(platform):
                if self.robot_vel[1] > 0:
                    robot_rect.bottom = platform.top
                    self.on_ground = True
                    self.robot_vel[1] = 0
                elif self.robot_vel[1] < 0:
                    robot_rect.top = platform.bottom
                    self.robot_vel[1] = 0
                self.robot_pos[1] = float(robot_rect.y)

        # Interactions
        robot_rect = pygame.Rect(int(self.robot_pos[0]), int(self.robot_pos[1]), *self.ROBOT_SIZE)
        
        # Coin collection
        collected_coins = [coin for coin in self.coins if robot_rect.colliderect(coin)]
        if collected_coins:
            self.coins = [c for c in self.coins if c not in collected_coins]
            num_collected = len(collected_coins)
            self.score += num_collected
            reward += num_collected * 1.0
            # sfx: coin_collect

        # Particle effects
        if self.on_ground and abs(self.robot_vel[0]) > 1 and self.steps % 3 == 0:
            self.particles.append({
                'pos': [self.robot_pos[0] + self.ROBOT_SIZE[0] / 2, self.robot_pos[1] + self.ROBOT_SIZE[1] - 2],
                'vel': [-self.robot_vel[0] * 0.1, self.np_random.uniform(-1, 0)],
                'life': 10
            })
        self._update_particles()

        # Check termination conditions
        terminated = False
        if robot_rect.colliderect(self.exit_rect):
            reward += 100
            terminated = True
            self.termination_message = "LEVEL COMPLETE!"
            # sfx: level_complete
        elif self.robot_pos[1] > self.SCREEN_HEIGHT:
            reward -= 100
            terminated = True
            self.termination_message = "GAME OVER"
            # sfx: fall
        elif self.time_left <= 0:
            reward -= 100
            terminated = True
            self.termination_message = "TIME UP!"
            # sfx: timeout
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.termination_message = "STEP LIMIT"

        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # Update camera to keep player in the left third of the screen
        self.camera_x = self.robot_pos[0] - self.SCREEN_WIDTH / 3
        self.camera_x = max(0, self.camera_x)

        # --- Main Render Pass ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.starfield:
            star_x = (star['pos'][0] - self.camera_x * star['speed']) % self.SCREEN_WIDTH
            c = star['color']
            pygame.draw.circle(self.screen, (c, c, c), (star_x, star['pos'][1]), 1)

    def _render_game_elements(self):
        # Platforms
        for p in self.platforms:
            cam_p = p.move(-self.camera_x, 0)
            if cam_p.right > 0 and cam_p.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, cam_p)

        # Coins
        for c in self.coins:
            cam_c = c.move(-self.camera_x, 0)
            if cam_c.right > 0 and cam_c.left < self.SCREEN_WIDTH:
                anim_width = self.COIN_SIZE * abs(math.sin(self.steps * 0.15))
                anim_rect = pygame.Rect(
                    cam_c.centerx - anim_width / 2, cam_c.y, anim_width, self.COIN_SIZE)
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, anim_rect)
                pygame.draw.ellipse(self.screen, tuple(min(255, x + 50) for x in self.COLOR_COIN), anim_rect, 2)

        # Exit
        cam_exit = self.exit_rect.move(-self.camera_x, 0)
        if cam_exit.right > 0 and cam_exit.left < self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_EXIT, cam_exit, border_radius=3)
            glow_surf = pygame.Surface(cam_exit.inflate(10, 10).size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_EXIT, 30), glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, cam_exit.inflate(10, 10).topleft)

        # Particles
        for p in self.particles:
            cam_pos = (p['pos'][0] - self.camera_x, p['pos'][1])
            radius = p['life'] / 3
            if radius > 0:
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, cam_pos, radius)

        # Robot
        robot_color = self.COLOR_ROBOT_JUMP if not self.on_ground else self.COLOR_ROBOT
        robot_rect_cam = pygame.Rect(int(self.robot_pos[0] - self.camera_x), int(self.robot_pos[1]), *self.ROBOT_SIZE)
        pygame.draw.rect(self.screen, robot_color, robot_rect_cam, border_radius=4)
        eye_x = robot_rect_cam.centerx + (5 if self.robot_vel[0] >= 0 else -5)
        eye_y = robot_rect_cam.centery - 5
        pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (eye_x, eye_y), 2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_seconds = max(0, self.time_left // 30)
        time_color = (255, 100, 100) if time_seconds <= 10 else self.COLOR_TEXT
        time_text = self.font_ui.render(f"TIME: {time_seconds}", True, time_color)
        self.screen.blit(time_text, (10, 35))

        if self.game_over:
            msg_surf = self.font_game_over.render(self.termination_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Beginning implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Example usage for manual testing
if __name__ == '__main__':
    # To play the game manually, you may need to set the SDL_VIDEODRIVER.
    # For example, on Linux:
    # import os
    # os.environ['SDL_VIDEODRIVER'] = 'x11'

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False

    # Pygame setup for a visible display
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("RoboCollector")

    while not done:
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        action = [movement, 0, 0]  # Space and shift are unused

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        env.clock.tick(30)

    print(f"Game Over. Final Info: {info}")
    env.close()