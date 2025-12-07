import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Short, user-facing control string
    user_guide = "Controls: Press space to jump from a platform. Aim for higher platforms to increase your score."

    # Short, user-facing description of the game
    game_description = "A minimalist side-scrolling arcade game where the player jumps between procedurally generated platforms to reach a target height."

    # Frames auto-advance at 30fps
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TARGET_HEIGHT_METERS = 500
        self.PIXELS_PER_METER = 10
        self.WIN_CONDITION_Y = -(self.TARGET_HEIGHT_METERS * self.PIXELS_PER_METER)

        # Physics and game constants
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -10
        self.MAX_EPISODE_STEPS = 5000

        # Colors
        self.COLOR_BG_TOP = (20, 30, 80)
        self.COLOR_BG_BOTTOM = (60, 80, 150)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 60)
        self.COLOR_PLATFORM = (180, 180, 190)
        self.COLOR_TEXT = (255, 255, 255)

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self._create_background_surface()

        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.last_space_held = None
        self.platforms = None
        self.camera_y = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.max_height_reached = None
        self.landed_platforms = None
        self.next_platform_id = None
        self.difficulty_factor = None
        self.particles = None
        self.last_landed_platform_id = None

        self.np_random = None  # Will be initialized in reset

        # Initialize state
        # A seed is not passed here, so the environment will be initialized with a random seed.
        # A specific seed can be passed to the reset method later.
        self.reset()

    def _create_background_surface(self):
        """Pre-renders the background gradient for performance."""
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT - 50.0])
        self.player_vel_y = 0.0
        self.on_ground = True
        self.last_space_held = False

        self.camera_y = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.max_height_reached = 0.0
        self.landed_platforms = set()
        self.next_platform_id = 0
        self.difficulty_factor = 1.0
        self.particles = []
        self.last_landed_platform_id = -1

        self._generate_initial_platforms()

        return self._get_observation(), self._get_info()

    def _generate_platform(self, y_pos, width=None):
        """Generates a single platform."""
        p_width = width if width is not None else self.np_random.integers(80, 150)
        # The high parameter for np.random.integers is exclusive. To get a value up to
        # and including self.SCREEN_WIDTH - p_width, the high bound must be
        # self.SCREEN_WIDTH - p_width + 1. This also fixes the crash when p_width is
        # self.SCREEN_WIDTH, as the call becomes integers(0, 1), which correctly returns 0.
        p_x = self.np_random.integers(0, self.SCREEN_WIDTH - p_width + 1)

        platform = {
            "id": self.next_platform_id,
            "rect": pygame.Rect(p_x, y_pos, p_width, 15),
            "base_y": y_pos,
            "amplitude": self.np_random.uniform(10, 40),
            "frequency": self.np_random.uniform(0.01, 0.03),
            "phase": self.np_random.uniform(0, 2 * math.pi),
        }
        self.next_platform_id += 1
        return platform

    def _generate_initial_platforms(self):
        """Creates the starting platforms."""
        self.platforms = []
        # Starting platform
        start_platform = self._generate_platform(self.SCREEN_HEIGHT - 40, width=self.SCREEN_WIDTH)
        start_platform["amplitude"] = 0  # Static start platform
        self.platforms.append(start_platform)

        # Subsequent platforms
        for i in range(1, 15):
            y = self.SCREEN_HEIGHT - 40 - self.np_random.integers(80, 120) * i
            self.platforms.append(self._generate_platform(y))

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # --- Action Handling ---
        movement = action[0]  # Unused
        space_held = action[1] == 1
        shift_held = action[2] == 1  # Unused

        jump_action = space_held and not self.last_space_held
        if jump_action and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            reward -= 0.1  # Small penalty for action to encourage fewer, better jumps
            self._create_jump_particles(self.player_pos)

        self.last_space_held = space_held

        # --- Physics and State Update ---
        self.steps += 1

        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.difficulty_factor += 0.1

        # Apply gravity
        if not self.on_ground:
            self.player_vel_y += self.GRAVITY

        # Update player position
        self.player_pos[1] += self.player_vel_y

        # Update platform positions
        for p in self.platforms:
            p["rect"].y = p["base_y"] + math.sin(self.steps * p["frequency"] * self.difficulty_factor + p["phase"]) * p["amplitude"]

        # --- Collision Detection ---
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)

        if self.player_vel_y > 0:  # Only check for landing if moving down
            for p in self.platforms:
                if player_rect.colliderect(p["rect"]) and player_rect.bottom < p["rect"].centery:
                    self.player_pos[1] = p["rect"].top - 10
                    self.player_vel_y = 0
                    self.on_ground = True

                    if p["id"] not in self.landed_platforms:
                        reward += 1.0  # Reward for landing on a new platform
                        self.landed_platforms.add(p["id"])

                    self.last_landed_platform_id = p["id"]
                    break

        # --- Update Camera ---
        # Camera follows player upwards, keeping them in the bottom half of the screen
        scroll_threshold = self.camera_y + self.SCREEN_HEIGHT / 2
        if self.player_pos[1] < scroll_threshold:
            self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT / 2

        # --- Reward Calculation ---
        current_height = max(0, (self.SCREEN_HEIGHT - 50 - self.player_pos[1]) / self.PIXELS_PER_METER)

        if current_height > self.max_height_reached:
            reward += (current_height - self.max_height_reached) * 0.1
            self.max_height_reached = current_height

        reward -= 0.02  # Time penalty per step

        # --- Platform Management ---
        # Remove platforms that are off-screen below
        self.platforms = [p for p in self.platforms if p["rect"].top < self.camera_y + self.SCREEN_HEIGHT + 50]

        # Add new platforms at the top
        highest_y = min(p["base_y"] for p in self.platforms) if self.platforms else self.camera_y
        if highest_y > self.camera_y - 50:
            new_y = highest_y - self.np_random.integers(80, 120)
            self.platforms.append(self._generate_platform(new_y))

        # --- Update Particles ---
        self._update_particles()

        # --- Termination Conditions ---
        # 1. Fell off the bottom of the screen
        if self.player_pos[1] > self.camera_y + self.SCREEN_HEIGHT:
            terminated = True
            reward -= 10.0  # Large penalty for falling

        # 2. Reached target height
        if self.max_height_reached >= self.TARGET_HEIGHT_METERS:
            terminated = True
            reward += 100.0  # Large reward for winning

        # 3. Max steps reached
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True  # Terminated, not truncated, as it's a time limit

        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render platforms
        for p in self.platforms:
            render_rect = p["rect"].move(0, -self.camera_y)
            pygame.gfxdraw.box(self.screen, render_rect, self.COLOR_PLATFORM)

        # Render player
        player_screen_pos = (
            int(self.player_pos[0]),
            int(self.player_pos[1] - self.camera_y)
        )

        # Player Glow
        glow_radius = 20
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (player_screen_pos[0] - glow_radius, player_screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player square
        player_rect = pygame.Rect(player_screen_pos[0] - 10, player_screen_pos[1] - 10, 20, 20)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

        # Render particles
        for particle in self.particles:
            pos = (int(particle['pos'][0]), int(particle['pos'][1] - self.camera_y))
            pygame.draw.circle(self.screen, particle['color'], pos, int(particle['radius']))

    def _render_ui(self):
        height_text = f"Height: {self.max_height_reached:.1f}m"
        text_surface = self.font.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        score_text = f"Score: {self.score:.1f}"
        score_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surface.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surface, score_rect)

        if self.game_over:
            if self.max_height_reached >= self.TARGET_HEIGHT_METERS:
                end_text = "GOAL REACHED!"
            else:
                end_text = "GAME OVER"

            end_surface = self.font.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_surface, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.max_height_reached,
        }

    def _create_jump_particles(self, pos):
        """Create a burst of particles for the jump effect."""
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': 20,  # frames
                'color': (255, 255, 255, 150)
            })

    def _update_particles(self):
        """Update position and lifespan of particles."""
        active_particles = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95  # Shrink
            if p['lifespan'] > 0 and p['radius'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the game directly for testing
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset(seed=42)

    running = True
    total_reward = 0.0

    # 0=none, 1=up, 2=down, 3=left, 4=right
    # 0=released, 1=held
    # 0=released, 1=held
    action = [0, 0, 0]  # No-op

    # Use a separate screen for rendering if running directly
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("PitLoch Jumper")

    while running:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'r'
                    print("Resetting environment.")
                    obs, info = env.reset(seed=42)
                    total_reward = 0.0
                if event.key == pygame.K_q:
                    running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        action[1] = 1 if keys[pygame.K_SPACE] else 0

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Height: {info['height']:.1f}m")
            # To stop the game on termination, comment out the reset line
            obs, info = env.reset(seed=42)
            total_reward = 0.0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control frame rate
        env.clock.tick(30)

    env.close()