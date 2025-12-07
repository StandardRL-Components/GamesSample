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


# Define a custom Brick class that inherits from pygame.Rect to store color
class Brick(pygame.Rect):
    def __init__(self, x, y, width, height, color):
        super().__init__(x, y, width, height)
        self.color = color


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based arcade game. Control a paddle to destroy all bricks with a bouncing ball before the time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.fps = 30  # For auto_advance=True, this sets the game speed

        # Visuals & Colors
        self.COLOR_BG = pygame.Color("#0d0f25")
        self.COLOR_GRID = pygame.Color("#1f224f")
        self.COLOR_PADDLE = pygame.Color("#ffffff")
        self.COLOR_BALL = pygame.Color("#ffff00")
        self.COLOR_TEXT = pygame.Color("#ffffff")
        self.BRICK_COLORS = [
            pygame.Color("#ff4757"), pygame.Color("#ff7f50"),
            pygame.Color("#ffa502"), pygame.Color("#2ed573"),
            pygame.Color("#1e90ff"), pygame.Color("#7d5fff")
        ]
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 50)

        # Game Constants
        self.max_steps = 1800  # 60 seconds at 30 FPS
        self.paddle_width = 100
        self.paddle_height = 16
        self.paddle_speed = 10
        self.ball_radius = 7
        self.ball_speed = 7

        # State variables (initialized in reset)
        self.paddle = None
        self.ball = None
        self.ball_velocity = None
        self.ball_attached = None
        self.bricks = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.ball_y_history = []

        # Initialize state
        # The reset method is called here to set up the initial state
        # self.reset() is called by the user/wrapper, not in __init__
        # A seed is needed for the first reset, so we'll do it lazily
        self._initial_reset_done = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initial_reset_done = True

        # Reset paddle
        paddle_y = self.screen_height - self.paddle_height * 2
        self.paddle = pygame.Rect(
            (self.screen_width - self.paddle_width) // 2,
            paddle_y,
            self.paddle_width,
            self.paddle_height,
        )

        # Reset ball
        self.ball_attached = True
        self._attach_ball()
        self.ball_velocity = [0, 0]
        self.ball_y_history.clear()

        # Reset bricks
        self.bricks.clear()
        self._create_bricks()

        # Reset particles
        self.particles.clear()

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if not self._initial_reset_done:
            # This is to ensure reset() is called before the first step
            # which is a requirement of the Gymnasium API
            self.reset()
            
        reward = 0
        terminated = False

        if self.game_over:
            # If the game is already over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        # 1. Handle player input
        self._handle_input(movement, space_held)

        # 2. Update game logic
        brick_hit_reward = self._update_ball()
        reward += brick_hit_reward
        if brick_hit_reward > 0:
            self.score += int(brick_hit_reward)
        self._update_particles()

        # 3. Check for termination conditions
        self.steps += 1

        if not self.bricks:  # Win condition
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100

        # Ball out of bounds loss condition (handled in _update_ball)
        if self.game_over and not self.win:
            terminated = True
            reward -= 100

        if self.steps >= self.max_steps and not terminated:  # Time-out loss condition
            self.game_over = True
            terminated = True
            reward -= 100

        if self.auto_advance:
            self.clock.tick(self.fps)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += self.paddle_speed

        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.screen_width - self.paddle_width, self.paddle.x))

        # Launch ball
        if self.ball_attached and space_held:
            self.ball_attached = False
            # Sound: Launch sound
            angle = self.np_random.uniform(-math.pi * 0.8, -math.pi * 0.2)
            self.ball_velocity = [
                self.ball_speed * math.cos(angle),
                self.ball_speed * math.sin(angle)
            ]

    def _update_ball(self):
        if self.ball_attached:
            self._attach_ball()
            return 0

        # Move ball
        self.ball.x += self.ball_velocity[0]
        self.ball.y += self.ball_velocity[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.screen_width:
            self.ball_velocity[0] *= -1
            self.ball.x = max(0, min(self.screen_width - self.ball.width, self.ball.x))
            # Sound: Wall bounce
        if self.ball.top <= 0:
            self.ball_velocity[1] *= -1
            self.ball.y = max(0, min(self.screen_height - self.ball.height, self.ball.y))
            # Sound: Wall bounce

        # Out of bounds (bottom)
        if self.ball.top >= self.screen_height:
            self.game_over = True
            # Sound: Lose sound
            return 0

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_velocity[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_velocity[1] *= -1
            # Add "spin" based on hit location
            hit_offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle_width / 2)
            self.ball_velocity[0] += hit_offset * 2
            # Clamp horizontal velocity to prevent extreme angles
            self.ball_velocity[0] = max(-self.ball_speed, min(self.ball_speed, self.ball_velocity[0]))
            # Sound: Paddle bounce

        # Brick collisions
        hit_brick_idx = self.ball.collidelist(self.bricks)
        if hit_brick_idx != -1:
            brick = self.bricks.pop(hit_brick_idx)

            # Create particles
            for _ in range(15):
                self._create_particle(brick.center, brick.color)

            # Determine bounce direction (simple approach: reverse vertical velocity)
            self.ball_velocity[1] *= -1
            # Sound: Brick break
            return 1  # Reward for hitting a brick

        # Anti-softlock mechanism
        self.ball_y_history.append(self.ball.y)
        if len(self.ball_y_history) > 60:  # Check over 2 seconds
            self.ball_y_history.pop(0)
            if max(self.ball_y_history) - min(self.ball_y_history) < 2.0:
                self.ball_velocity[1] += self.np_random.choice([-0.5, 0.5])

        return 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _attach_ball(self):
        self.ball = pygame.Rect(
            self.paddle.centerx - self.ball_radius,
            self.paddle.top - self.ball_radius * 2,
            self.ball_radius * 2,
            self.ball_radius * 2
        )

    def _create_bricks(self):
        brick_rows = 6
        brick_cols = 10
        brick_width = (self.screen_width - (brick_cols + 1) * 4) // brick_cols
        brick_height = 20
        y_offset = 50

        for r in range(brick_rows):
            for c in range(brick_cols):
                x = c * (brick_width + 4) + 4
                y = r * (brick_height + 4) + y_offset
                color = self.BRICK_COLORS[r % len(self.BRICK_COLORS)]
                brick = Brick(x, y, brick_width, brick_height, color)
                self.bricks.append(brick)

    def _create_particle(self, pos, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        self.particles.append({
            'pos': list(pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': self.np_random.integers(15, 30),
            'color': color
        })

    def _get_observation(self):
        # Lazy initialization of game state
        if not self._initial_reset_done:
            self.reset()
            
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_background_grid()
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_grid(self):
        for x in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.screen_width, y))

    def _render_game(self):
        # Draw bricks
        for brick in self.bricks:
            pygame.draw.rect(self.screen, brick.color, brick, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * (255 / 30))))
            size = max(1, int(p['life'] * (5 / 30)))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*p['color'][:3], alpha))
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw ball with glow
        center = (int(self.ball.centerx), int(self.ball.centery))
        glow_radius = self.ball_radius + 4

        # Draw a larger, semi-transparent circle for the glow
        glow_color = (*self.COLOR_BALL[:3], 100)
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
        self.screen.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius))

        # Draw the main ball
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = (self.max_steps - self.steps) / self.fps
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = pygame.Color("gold") if self.win else pygame.Color("red")
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bricks_left": len(self.bricks),
            "game_over": self.game_over
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset(seed=42)

    # Pygame window for human play
    pygame.display.init()
    render_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Breakout")

    action = env.action_space.sample()
    action.fill(0)  # Start with no-op

    running = True
    while running:
        # Reset action at the start of each frame
        action[0] = 0  # No movement
        action[1] = 0  # Space released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)  # Pause for 2 seconds
            obs, info = env.reset()

    env.close()