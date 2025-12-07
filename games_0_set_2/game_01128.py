import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space for a small hop, or hold Shift for a large jump. "
        "Avoid the red obstacles!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling arcade game where you control a hopping spaceship. "
        "Dodge obstacles to reach the end of each stage before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 60

        # Colors
        self.COLOR_BG = pygame.Color("#1E1E2E")
        self.COLOR_FLOOR = pygame.Color("#45475A")
        self.COLOR_PLAYER = pygame.Color("#A6E3A1")
        self.COLOR_PLAYER_GLOW = pygame.Color("#A6E3A1")
        self.COLOR_PLAYER_COCKPIT = pygame.Color("#11111B")
        self.COLOR_PLAYER_THRUST = pygame.Color("#F9E2AF")
        self.COLOR_OBSTACLE = pygame.Color("#F38BA8")
        self.COLOR_TEXT = pygame.Color("#CDD6F4")
        self.COLOR_STAR = pygame.Color("#BAC2DE")

        # Physics & Gameplay
        self.GRAVITY = -0.35
        self.SMALL_JUMP_STRENGTH = 7.5
        self.LARGE_JUMP_STRENGTH = 11.0
        self.PLAYER_FORWARD_SPEED = 3.5
        self.PLAYER_WIDTH = 30
        self.PLAYER_HEIGHT = 30
        self.PLAYER_SCREEN_X = 120
        self.FLOOR_Y = self.SCREEN_HEIGHT - 60

        # Stage & Difficulty
        self.MAX_STAGES = 3
        self.STAGE_DURATION_SECONDS = 60
        self.STAGE_DURATION_FRAMES = self.STAGE_DURATION_SECONDS * self.FPS
        self.STAGE_LENGTH_PIXELS = (
            self.PLAYER_FORWARD_SPEED * self.STAGE_DURATION_FRAMES
        )

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_stage = pygame.font.Font(None, 48)

        # --- State Variables ---
        self.player_world_x = 0
        self.player_y = 0
        self.player_vel_y = 0
        self.is_jumping = False
        self.on_ground = True

        self.obstacles = []
        self.stars = []

        self.score = 0
        self.stage = 1
        self.timer = 0
        self.steps = 0
        self.game_over = False

        self.obstacle_base_speed = 0
        self.obstacle_spawn_timer = 0
        self.obstacle_spawn_interval = 0

        # For reward calculation
        self.min_dist_to_obstacle_airborne = float("inf")

        # Initialize state
        # A seed is not passed here, but the super().reset() in the reset method will handle it.
        self.reset()

        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_world_x = 0
        self.player_y = self.FLOOR_Y
        self.player_vel_y = 0
        self.is_jumping = False
        self.on_ground = True

        self.obstacles = []
        self._init_stars()

        self.score = 0
        self.stage = 1
        self.timer = self.STAGE_DURATION_FRAMES
        self.steps = 0
        self.game_over = False

        self._update_difficulty_for_stage()
        self.obstacle_spawn_timer = (
            self.obstacle_spawn_interval
        )  # Spawn first obstacle soon

        self.min_dist_to_obstacle_airborne = float("inf")

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # --- Game Logic ---
        self._handle_input(space_held, shift_held)
        self._update_player()
        self._update_world()

        # --- State Updates ---
        self.steps += 1
        self.timer -= 1

        # --- Reward and Termination ---
        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            # Apply large terminal penalty if not a stage win
            if self.player_world_x < self.STAGE_LENGTH_PIXELS:
                reward -= 100

        # Check for stage completion
        if not terminated and self.player_world_x >= self.STAGE_LENGTH_PIXELS:
            reward += 100
            self.score += 1000  # Player-facing score bonus
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                terminated = True  # Game won!
                self.game_over = True
            else:
                self._advance_to_next_stage()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, space_held, shift_held):
        # sfx: jump_sound.play()
        if self.on_ground:
            if shift_held:
                self.player_vel_y = self.LARGE_JUMP_STRENGTH
                self.is_jumping = True
            elif space_held:
                self.player_vel_y = self.SMALL_JUMP_STRENGTH
                self.is_jumping = True

    def _update_player(self):
        # Constant forward motion
        self.player_world_x += self.PLAYER_FORWARD_SPEED

        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_y -= self.player_vel_y

        self.on_ground = False
        if self.player_y >= self.FLOOR_Y:
            self.player_y = self.FLOOR_Y
            self.player_vel_y = 0
            if self.is_jumping:  # Just landed
                self.is_jumping = False
            self.on_ground = True

    def _update_world(self):
        # Move obstacles
        for obs in self.obstacles:
            obs["pos"][0] -= self.obstacle_base_speed
            if obs["type"] == "sine":
                obs["pos"][1] = obs["initial_y"] + obs["sine_amp"] * math.sin(
                    (self.steps + obs["sine_phase"]) * obs["sine_freq"]
                )

        # Remove off-screen obstacles
        self.obstacles = [
            obs for obs in self.obstacles if obs["pos"][0] + obs["size"][0] > 0
        ]

        # Spawn new obstacles
        self.obstacle_spawn_timer += 1
        if self.obstacle_spawn_timer >= self.obstacle_spawn_interval:
            self._spawn_obstacle()
            self.obstacle_spawn_timer = 0
            # Add some randomness to spawn timing
            self.obstacle_spawn_interval = self.np_random.integers(70, 100)

    def _calculate_reward(self):
        reward = 0.1  # Survival reward

        just_landed = self.on_ground and self.is_jumping

        # Penalize jumps based on clearance
        if just_landed:
            if self.min_dist_to_obstacle_airborne < 5:
                reward -= 5  # Risky near-miss
            elif self.min_dist_to_obstacle_airborne > self.PLAYER_HEIGHT + 20:
                reward -= 10  # Inefficiently high jump
            self.min_dist_to_obstacle_airborne = float("inf")  # Reset for next jump

        # Reward for clearing obstacles
        player_left_edge = self.PLAYER_SCREEN_X - self.PLAYER_WIDTH / 2
        for obs in self.obstacles:
            obs_screen_x = obs["pos"][0] - (self.player_world_x - self.PLAYER_SCREEN_X)
            obs_right_edge = obs_screen_x + obs["size"][0]
            if not obs["cleared"] and obs_right_edge < player_left_edge:
                obs["cleared"] = True
                reward += 1
                self.score += 10

        # Update jump riskiness metric while airborne
        if not self.on_ground:
            player_rect = pygame.Rect(
                self.PLAYER_SCREEN_X - self.PLAYER_WIDTH / 2,
                self.player_y - self.PLAYER_HEIGHT,
                self.PLAYER_WIDTH,
                self.PLAYER_HEIGHT,
            )
            min_dist_this_frame = float("inf")
            for obs in self.obstacles:
                obs_screen_x = obs["pos"][0] - (
                    self.player_world_x - self.PLAYER_SCREEN_X
                )
                # Check if player is horizontally aligned with the obstacle
                if (
                    player_rect.right > obs_screen_x
                    and player_rect.left < obs_screen_x + obs["size"][0]
                ):
                    dist_to_top = obs["pos"][1] - player_rect.bottom
                    if dist_to_top > 0:  # Only consider obstacles we are jumping OVER
                        min_dist_this_frame = min(min_dist_this_frame, dist_to_top)

            self.min_dist_to_obstacle_airborne = min(
                self.min_dist_to_obstacle_airborne, min_dist_this_frame
            )

        return reward

    def _check_termination(self):
        # Timer ran out
        if self.timer <= 0:
            return True

        # Collision with obstacles
        player_rect = pygame.Rect(
            self.PLAYER_SCREEN_X - self.PLAYER_WIDTH / 2,
            self.player_y - self.PLAYER_HEIGHT,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )
        camera_x = self.player_world_x - self.PLAYER_SCREEN_X
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                obs["pos"][0] - camera_x, obs["pos"][1], *obs["size"]
            )
            if player_rect.colliderect(obs_rect):
                # sfx: explosion_sound.play()
                return True

        return False

    def _advance_to_next_stage(self):
        self.player_world_x = 0
        self.timer = self.STAGE_DURATION_FRAMES
        self.obstacles = []
        self._update_difficulty_for_stage()
        self.obstacle_spawn_timer = 0

    def _update_difficulty_for_stage(self):
        self.obstacle_base_speed = 1.0 + (self.stage - 1) * 0.2
        self.obstacle_spawn_interval = 90

    def _spawn_obstacle(self):
        height = self.np_random.integers(40, 120)
        y_pos = self.FLOOR_Y - height
        width = self.np_random.integers(30, 60)

        # Ensure spawn is off-screen to the right
        spawn_x = self.player_world_x + self.SCREEN_WIDTH

        obs_type = (
            self.np_random.choice(["linear", "sine"]) if self.stage > 1 else "linear"
        )

        new_obstacle = {
            "pos": [spawn_x, y_pos],
            "size": [width, height],
            "type": obs_type,
            "initial_y": y_pos,
            "cleared": False,
            "sine_amp": self.np_random.uniform(10, 30) if obs_type == "sine" else 0,
            "sine_freq": self.np_random.uniform(0.02, 0.05) if obs_type == "sine" else 0,
            "sine_phase": self.np_random.uniform(0, 2 * math.pi),
        }
        self.obstacles.append(new_obstacle)

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "stage": self.stage,
            "timer": self.timer // self.FPS,
            "world_progress": self.player_world_x / self.STAGE_LENGTH_PIXELS,
        }

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_starfield()
        self._render_floor()
        self._render_obstacles()
        self._render_player()
        self._render_ui()

    def _init_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.FLOOR_Y)
            size = self.np_random.choice([1, 2, 3])
            self.stars.append({"pos": [x, y], "size": size})

    def _render_starfield(self):
        camera_x = self.player_world_x - self.PLAYER_SCREEN_X
        for star in self.stars:
            # Parallax effect
            scroll_speed = star["size"] * 0.2
            star_x = (star["pos"][0] - camera_x * scroll_speed) % self.SCREEN_WIDTH
            pygame.draw.circle(
                self.screen,
                self.COLOR_STAR,
                (int(star_x), int(star["pos"][1])),
                star["size"] // 2,
            )

    def _render_floor(self):
        pygame.draw.rect(
            self.screen,
            self.COLOR_FLOOR,
            (0, self.FLOOR_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.FLOOR_Y),
        )

    def _render_obstacles(self):
        camera_x = self.player_world_x - self.PLAYER_SCREEN_X
        for obs in self.obstacles:
            obs_rect = pygame.Rect(
                int(obs["pos"][0] - camera_x), int(obs["pos"][1]), *obs["size"]
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_rect)
            pygame.draw.rect(
                self.screen, self.COLOR_OBSTACLE.lerp((0, 0, 0), 0.3), obs_rect, 2
            )

    def _render_player(self):
        player_rect = pygame.Rect(
            self.PLAYER_SCREEN_X - self.PLAYER_WIDTH / 2,
            self.player_y - self.PLAYER_HEIGHT,
            self.PLAYER_WIDTH,
            self.PLAYER_HEIGHT,
        )

        # Glow effect
        glow_color = self.COLOR_PLAYER_GLOW
        glow_radius = int(self.PLAYER_WIDTH * 0.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            s, (glow_color.r, glow_color.g, glow_color.b, 50), (glow_radius, glow_radius), glow_radius
        )
        self.screen.blit(
            s, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius)
        )

        # Main body
        body_points = [
            (player_rect.centerx, player_rect.top),
            (player_rect.right, player_rect.centery),
            (player_rect.centerx, player_rect.bottom),
            (player_rect.left, player_rect.centery),
        ]
        pygame.gfxdraw.aapolygon(self.screen, body_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, body_points, self.COLOR_PLAYER)

        # Cockpit
        pygame.draw.circle(
            self.screen,
            self.COLOR_PLAYER_COCKPIT,
            player_rect.center,
            int(self.PLAYER_WIDTH * 0.2),
        )

        # Thrust effect when jumping
        if self.is_jumping and self.player_vel_y > 0:
            thrust_height = self.player_vel_y * 1.5
            thrust_points = [
                (player_rect.centerx - 5, player_rect.bottom),
                (player_rect.centerx + 5, player_rect.bottom),
                (player_rect.centerx, player_rect.bottom + thrust_height),
            ]
            pygame.gfxdraw.aapolygon(
                self.screen, thrust_points, self.COLOR_PLAYER_THRUST
            )
            pygame.gfxdraw.filled_polygon(
                self.screen, thrust_points, self.COLOR_PLAYER_THRUST
            )

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer // self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_stage.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        stage_pos = (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, 10)
        self.screen.blit(stage_text, stage_pos)

        # Progress bar
        progress_ratio = min(1.0, self.player_world_x / self.STAGE_LENGTH_PIXELS)
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 5
        bar_y = self.SCREEN_HEIGHT - 20
        pygame.draw.rect(
            self.screen, self.COLOR_FLOOR, (10, bar_y, bar_width, bar_height)
        )
        pygame.draw.rect(
            self.screen,
            self.COLOR_PLAYER,
            (10, bar_y, bar_width * progress_ratio, bar_height),
        )

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    
    # We need to unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")

    # Use a separate display for human play
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")

    obs, info = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        # --- Human Controls ---
        movement = 0  # unused
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering for Human ---
        # The environment already rendered to its internal surface. We just need to display it.
        # We need to get the surface from the env, which isn't standard.
        # So we'll just blit the returned observation array.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()
    pygame.quit()