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


# Set the SDL_VIDEODRIVER to dummy for headless operation, which is required for Gymnasium
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press Space to jump. Hold Shift and press Space for a boosted jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-view arcade game where the player controls a hopping spaceship, aiming to reach a target altitude while dodging procedurally generated obstacles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_HEIGHT = 500
        self.MAX_STEPS = 1000
        self.MAX_COLLISIONS = 3

        # Physics constants
        self.GRAVITY = 0.25
        self.JUMP_STRENGTH = -7.5
        self.BOOST_JUMP_STRENGTH = -10
        self.HORIZONTAL_JUMP_VEL = 2.5
        self.HORIZONTAL_DRAG = 0.95
        self.KNOCKBACK_VEL = 3

        # Visuals
        self.COLOR_BG = (10, 15, 45)  # Dark Blue
        self.COLOR_PLAYER = (0, 200, 255)  # Bright Blue
        self.COLOR_PLAYER_GLOW = (0, 100, 255, 50)
        self.COLOR_OBSTACLE = (255, 50, 100)  # Red
        self.COLOR_STAR = (200, 200, 255)
        self.PARTICLE_COLORS = [(0, 255, 255), (255, 0, 255)]  # Cyan, Magenta
        self.UI_COLOR = (255, 255, 255)
        self.UI_BAR_BG = (50, 50, 100)
        self.UI_BAR_FILL = (100, 200, 255)

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_size = 12
        self.player_rect = None
        self.obstacles = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None  # Max height reached
        self.collisions = None
        self.game_over = None
        self.win = None
        self.prev_space_held = None
        self.obstacle_speed = None
        self.next_obstacle_spawn_y = None
        self.collision_cooldown = 0

        # Run self-check to ensure API compliance before first reset
        # self.validate_implementation() # This will fail if called before the first reset populates state
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_vel = [0, 0]
        self.obstacles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.collisions = 0
        self.game_over = False
        self.win = False
        self.prev_space_held = False
        self.collision_cooldown = 0

        self.obstacle_speed = 2.0
        self.next_obstacle_spawn_y = self.player_pos[1] - 150

        # Generate a static starfield
        if self.stars is None:
            self.stars = []
            for _ in range(150):
                self.stars.append(
                    (
                        self.np_random.uniform(0, self.WIDTH),
                        self.np_random.uniform(0, self.HEIGHT),
                        self.np_random.uniform(0.5, 1.5),  # size
                    )
                )

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        # movement = action[0] # 0-4: none/up/down/left/right -> Unused
        space_held = action[1] == 1
        shift_held = action[2] == 1

        reward = 0
        terminated = False

        # --- Action Handling ---
        # Jump is triggered on the rising edge of the space button
        jump_triggered = space_held and not self.prev_space_held
        if jump_triggered:
            if shift_held:
                # Boosted jump
                self.player_vel[1] = self.BOOST_JUMP_STRENGTH
                # Sound: boost_jump.wav
            else:
                # Normal jump
                self.player_vel[1] = self.JUMP_STRENGTH
                # Sound: jump.wav

            # Add horizontal velocity and particles
            self.player_vel[0] += self.np_random.uniform(
                -self.HORIZONTAL_JUMP_VEL, self.HORIZONTAL_JUMP_VEL
            )
            self._create_particles(20)

        self.prev_space_held = space_held

        # --- Physics and Game Logic Update ---
        # Apply gravity
        self.player_vel[1] += self.GRAVITY

        # Apply horizontal drag
        self.player_vel[0] *= self.HORIZONTAL_DRAG

        # Update player position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]

        # Keep player within horizontal bounds
        if self.player_pos[0] < self.player_size:
            self.player_pos[0] = self.player_size
            self.player_vel[0] = 0
        if self.player_pos[0] > self.WIDTH - self.player_size:
            self.player_pos[0] = self.WIDTH - self.player_size
            self.player_vel[0] = 0

        # Prevent player from falling through the floor
        if self.player_pos[1] > self.HEIGHT - self.player_size:
            self.player_pos[1] = self.HEIGHT - self.player_size
            if self.player_vel[1] > 0:
                self.player_vel[1] = 0

        # Update player rect for collision
        self.player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size / 2,
            self.player_pos[1] - self.player_size / 2,
            self.player_size,
            self.player_size,
        )

        # Update score (height)
        old_score = self.score
        current_height = max(0, int(self.HEIGHT - 50 - self.player_pos[1]))
        if current_height > self.score:
            reward += 0.1 * (current_height - self.score)
            self.score = current_height

        # Penalize jumping without gaining height
        if jump_triggered and self.score <= old_score:
            reward -= 0.2

        # --- Obstacle Management ---
        # Update obstacle positions
        for obs in self.obstacles:
            obs["rect"].x -= self.obstacle_speed

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs["rect"].right > 0]

        # Spawn new obstacles as player climbs
        if self.player_pos[1] < self.next_obstacle_spawn_y:
            self._spawn_obstacle_row()
            self.next_obstacle_spawn_y -= self.np_random.uniform(100, 150)

        # --- Collision Detection ---
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1

        if self.collision_cooldown == 0:
            for obs in self.obstacles:
                if self.player_rect.colliderect(obs["rect"]):
                    self.collisions += 1
                    reward -= 5.0
                    self.collision_cooldown = 30  # 1 second invincibility
                    self.player_vel[1] = self.KNOCKBACK_VEL  # Knock down
                    self._create_particles(10, self.COLOR_OBSTACLE)
                    # Sound: collision.wav
                    break

        # --- Particle Update ---
        self._update_particles()

        # --- Step and Difficulty Update ---
        self.steps += 1
        if self.steps % 50 == 0:
            self.obstacle_speed += 0.1

        # --- Termination Check ---
        truncated = False
        if self.score >= self.WIN_HEIGHT:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100.0
        elif self.collisions >= self.MAX_COLLISIONS:
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            truncated = True # Use truncated for time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_obstacle_row(self):
        gap_size = self.np_random.uniform(120, 180)
        gap_y = self.next_obstacle_spawn_y

        obstacle_height = 25

        # Create two obstacles with a gap between them
        # This implementation uses a single gap in a full row
        gap_x_start = self.np_random.uniform(50, self.WIDTH - 50 - gap_size)

        # Obstacle 1 (left of gap)
        if gap_x_start > 0:
            rect1 = pygame.Rect(self.WIDTH, gap_y, gap_x_start, obstacle_height)
            self.obstacles.append({"rect": rect1})

        # Obstacle 2 (right of gap)
        if gap_x_start + gap_size < self.WIDTH:
            rect2 = pygame.Rect(
                self.WIDTH + gap_x_start + gap_size,
                gap_y,
                self.WIDTH - (gap_x_start + gap_size),
                obstacle_height,
            )
            self.obstacles.append({"rect": rect2})

    def _create_particles(self, count, color=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            p_color = color if color is not None else random.choice(self.PARTICLE_COLORS)
            self.particles.append(
                {
                    "pos": list(self.player_pos),
                    "vel": vel,
                    "lifetime": self.np_random.integers(20, 40),
                    "color": p_color,
                }
            )

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += self.GRAVITY * 0.5  # Particles are affected by a bit of gravity
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"])

        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifetime"] / 40))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p["pos"][0] - 2), int(p["pos"][1] - 2)))

        # Draw player
        if self.collision_cooldown > 0 and (self.steps // 3) % 2 == 0:
            # Flicker when invincible
            pass
        else:
            # Glow effect
            glow_size = self.player_size * 2.5
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surf,
                self.COLOR_PLAYER_GLOW,
                (glow_size / 2, glow_size / 2),
                glow_size / 2,
            )
            self.screen.blit(
                glow_surf,
                (
                    int(self.player_pos[0] - glow_size / 2),
                    int(self.player_pos[1] - glow_size / 2),
                ),
            )

            # Player ship (triangle)
            p1 = (self.player_pos[0], self.player_pos[1] - self.player_size)
            p2 = (
                self.player_pos[0] - self.player_size / 2,
                self.player_pos[1] + self.player_size / 2,
            )
            p3 = (
                self.player_pos[0] + self.player_size / 2,
                self.player_pos[1] + self.player_size / 2,
            )
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _draw_heart(self, surface, center_x, center_y, size, color):
        """Draws an anti-aliased, filled heart centered at (center_x, center_y)."""
        # These points define a heart shape around (0,0)
        # The shape is chosen to be visually pleasing.
        raw_points = [
            (0, 0.4 * size),  # Bottom point
            (-0.5 * size, 0.1 * size),
            (-0.9 * size, -0.3 * size),
            (-0.5 * size, -0.8 * size),
            (0, -0.4 * size),  # Top indent
            (0.5 * size, -0.8 * size),
            (0.9 * size, -0.3 * size),
            (0.5 * size, 0.1 * size),
        ]

        # Find the geometrical center of the raw points to adjust positioning
        y_min = -0.8 * size
        y_max = 0.4 * size
        y_offset = (y_min + y_max) / 2

        # Translate points to be centered around the desired (center_x, center_y)
        points = [(p[0] + center_x, p[1] + center_y - y_offset) for p in raw_points]

        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        # Height bar on the left
        bar_height = self.HEIGHT - 40
        bar_x = 20
        pygame.draw.rect(self.screen, self.UI_BAR_BG, (bar_x, 20, 15, bar_height))

        fill_ratio = min(1.0, self.score / self.WIN_HEIGHT)
        fill_height = bar_height * fill_ratio
        pygame.draw.rect(
            self.screen,
            self.UI_BAR_FILL,
            (bar_x, 20 + bar_height - fill_height, 15, fill_height),
        )

        # Height text
        height_text = self.font_small.render(f"{self.score}m", True, self.UI_COLOR)
        self.screen.blit(height_text, (bar_x + 25, 20))

        # Collision icons (top right)
        for i in range(self.MAX_COLLISIONS):
            pos = (self.WIDTH - 30 - i * 25, 25)
            if i < self.collisions:
                # Draw a filled 'X' for a lost life
                pygame.draw.line(
                    self.screen,
                    self.COLOR_OBSTACLE,
                    (pos[0] - 8, pos[1] - 8),
                    (pos[0] + 8, pos[1] + 8),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    self.COLOR_OBSTACLE,
                    (pos[0] - 8, pos[1] + 8),
                    (pos[0] + 8, pos[1] - 8),
                    3,
                )
            else:
                # Draw a heart for a remaining life
                self._draw_heart(
                    self.screen, int(pos[0]), int(pos[1]), 10, self.COLOR_PLAYER
                )

        # Game Over / Win Text
        if self.game_over:
            if self.win:
                msg = "GOAL REACHED!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_OBSTACLE

            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions": self.collisions,
        }

    def close(self):
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == "__main__":
    # For human play, we want a window.
    # If you are running this on a server, you might need to switch to 'dummy'.
    os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv()
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    print(env.user_guide)

    while running:
        # --- Action gathering for human play ---
        keys = pygame.key.get_pressed()

        movement = 0  # Not used in this game
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(
                f"Episode finished. Total reward: {total_reward:.2f}, Final Info: {info}"
            )
            # In a real scenario you might auto-reset, here we wait for 'R'

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control frame rate
        clock.tick(30)  # Match the intended FPS

    env.close()