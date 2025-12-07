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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. ↑↓ to add/remove paddles. "
        "Press space to activate the top paddle, or shift to activate all paddles for a combo."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack paddles vertically to return bouncing balls in this top-down puzzle pong game. "
        "Higher stacks give more points but are riskier. Return 25 balls to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH = 80
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPACING = 2
        self.PADDLE_SPEED = 12
        self.MAX_PADDLES = 3
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 25

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PADDLE = (0, 200, 200)
        self.COLOR_PADDLE_ACTIVE = (150, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_BORDER = (100, 100, 120)

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
        self.font_big = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.ball_speed = 0
        self.paddle_x = 0
        self.paddle_stack_count = 1
        self.balls_returned = 0
        self.particles = []
        self.paddles_active_this_frame = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self._prev_move_up = False
        self._prev_move_down = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_returned = 0

        self.paddle_x = self.WIDTH // 2
        self.paddle_stack_count = 1

        self.ball_pos = [self.WIDTH // 2, self.HEIGHT // 4]
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.ball_vel = [math.cos(angle) * self.ball_speed, -math.sin(angle) * self.ball_speed]

        self.particles = []
        self.paddles_active_this_frame = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Initialize attributes to fix AttributeError on first step
        self._prev_move_up = False
        self._prev_move_down = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.1  # Survival reward
        self.game_over = self.steps >= self.MAX_STEPS

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if not self.game_over:
            # Movement
            if movement == 3:  # Left
                self.paddle_x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle_x += self.PADDLE_SPEED

            # Clamp paddle position
            self.paddle_x = max(self.PADDLE_WIDTH // 2, min(self.WIDTH - self.PADDLE_WIDTH // 2, self.paddle_x))

            # Stack changes (on press, not hold)
            if movement == 1 and not self._prev_move_up:
                self.paddle_stack_count = min(self.MAX_PADDLES, self.paddle_stack_count + 1)
            if movement == 2 and not self._prev_move_down:
                self.paddle_stack_count = max(1, self.paddle_stack_count - 1)

            self._prev_move_up = movement == 1
            self._prev_move_down = movement == 2

            # Paddle activation (on press, not hold)
            space_pressed = space_held and not self.prev_space_held
            shift_pressed = shift_held and not self.prev_shift_held
            self.paddles_active_this_frame.clear()
            shift_missed = False

            if space_pressed:
                self.paddles_active_this_frame.append(self.paddle_stack_count - 1)
            elif shift_pressed:
                self.paddles_active_this_frame = list(range(self.paddle_stack_count))
                shift_missed = True  # Assume miss until a hit is confirmed

        # --- Update Game Logic ---
        if not self.game_over:
            # Move Ball
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS,
                                    self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Ball vs Wall Collision
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball_pos[0] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos[0]))
            if ball_rect.top <= 0:
                self.ball_vel[1] *= -1
                self.ball_pos[1] = max(self.BALL_RADIUS, self.ball_pos[1])

            # Ball vs Paddle Collision
            for i in range(self.paddle_stack_count):
                paddle_y = self.HEIGHT - (i + 1) * (self.PADDLE_HEIGHT + self.PADDLE_SPACING)
                paddle_rect = pygame.Rect(self.paddle_x - self.PADDLE_WIDTH // 2, paddle_y, self.PADDLE_WIDTH,
                                          self.PADDLE_HEIGHT)

                if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
                    if i in self.paddles_active_this_frame:
                        shift_missed = False  # Hit confirmed, not a miss

                        # Calculate reward based on number of active paddles
                        hit_reward = len(self.paddles_active_this_frame)
                        reward += hit_reward
                        self.score += hit_reward

                        self.balls_returned += 1

                        # Reverse ball velocity and add slight random horizontal spin
                        self.ball_vel[1] *= -1
                        self.ball_vel[0] += self.np_random.uniform(-0.5, 0.5)
                        # Nudge ball out of paddle
                        self.ball_pos[1] = paddle_rect.top - self.BALL_RADIUS

                        self._create_particles(self.ball_pos, 20)

                        # Increase speed every 5 returns
                        if self.balls_returned > 0 and self.balls_returned % 5 == 0:
                            self.ball_speed += 0.5
                            # Rescale velocity vector to new speed
                            current_magnitude = math.sqrt(self.ball_vel[0] ** 2 + self.ball_vel[1] ** 2)
                            if current_magnitude > 0:
                                scale = self.ball_speed / current_magnitude
                                self.ball_vel = [self.ball_vel[0] * scale, self.ball_vel[1] * scale]

                        if self.balls_returned >= self.WIN_CONDITION:
                            self.game_over = True
                            reward += 100
                            self.score += 100

                        break  # Only one hit per frame

            if shift_missed:
                reward -= 0.2

            # Ball vs Bottom (Game Over)
            if ball_rect.bottom >= self.HEIGHT:
                self.game_over = True
                reward -= 100
                self.score -= 100

        # Update Particles
        self._update_particles()

        # Update button states for next frame
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self.steps += 1
        terminated = self.game_over

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(
                {'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'size': self.np_random.uniform(1, 3)})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw borders
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30.0))))
            color = (*self.COLOR_PARTICLE, alpha)
            s = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw ball with glow
        self._draw_glowing_circle(self.screen, self.ball_pos, self.BALL_RADIUS, self.COLOR_BALL, 15)

        # Draw paddles
        for i in range(self.paddle_stack_count):
            paddle_y = self.HEIGHT - (i + 1) * (self.PADDLE_HEIGHT + self.PADDLE_SPACING)
            rect = pygame.Rect(self.paddle_x - self.PADDLE_WIDTH // 2, paddle_y, self.PADDLE_WIDTH,
                                self.PADDLE_HEIGHT)

            color = self.COLOR_PADDLE
            if i in self.paddles_active_this_frame:
                color = self.COLOR_PADDLE_ACTIVE
                # Draw glow for active paddle
                glow_rect = rect.inflate(8, 8)
                glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*color, 50), glow_surf.get_rect(), border_radius=5)
                self.screen.blit(glow_surf, glow_rect.topleft)

            pygame.draw.rect(self.screen, color, rect, border_radius=3)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_size):
        # Draw glow
        glow_radius = radius + glow_size
        for i in range(glow_size, 0, -1):
            alpha = int(50 * (1 - i / glow_size))
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius + i, (*color, alpha))

        # Draw main circle
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), radius, color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), radius, color)

    def _render_ui(self):
        score_text = self.font_big.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        balls_text = self.font_big.render(f"Balls: {self.balls_returned} / {self.WIN_CONDITION}", True,
                                          self.COLOR_TEXT)
        text_rect = balls_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(balls_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text_str = "YOU WIN!" if self.balls_returned >= self.WIN_CONDITION else "GAME OVER"
            end_text = self.font_big.render(end_text_str, True,
                                            self.COLOR_BALL if self.balls_returned >= self.WIN_CONDITION else (
                                            255, 50, 50))
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_returned": self.balls_returned,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    # To use, you might need to unset the dummy video driver:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")

    # To display the game, we need a screen
    pygame.display.set_caption("Puzzle Pong")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    done = False

    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input
        keys = pygame.key.get_pressed()

        # Map keys to action space
        movement = 0  # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Reset if the episode is over
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            pygame.time.wait(2000)  # Pause before restarting

        # Control the frame rate
        env.clock.tick(30)

    env.close()