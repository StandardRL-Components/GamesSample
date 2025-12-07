
# Generated: 2025-08-27T20:21:12.465210
# Source Brief: brief_02430.md
# Brief Index: 2430

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# A simple particle class for effects
class Particle:
    def __init__(self, pos, vel, color, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.vel *= 0.95 # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            # Fade out effect
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color = self.color + (alpha,)
            radius = int(2 + 6 * (self.lifespan / self.max_lifespan))
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            surface.blit(temp_surf, self.pos - pygame.Vector2(radius, radius))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to aim the launcher. Press Space to launch. "
        "After launch, use ← and → to move the paddle."
    )

    game_description = (
        "A vibrant, modern take on a brick-breaking classic. Aim your shot, "
        "launch the ball, and clear all the bricks to win. Don't let the ball get past your paddle!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8.0
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 6.0
        self.MAX_STEPS = 2500
        self.MAX_LIVES = 3

        # --- Colors ---
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 200, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.BRICK_COLORS = [
            (0, 255, 255), (255, 0, 255), (255, 165, 0), 
            (75, 0, 130), (0, 255, 0), (255, 69, 0)
        ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.game_phase = "aiming" # 'aiming' or 'in_play'
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.launcher_angle = 0.0
        self.bricks = []
        self.particles = []
        self.prev_space_held = False
        self.rng = None

        self.reset()
        
        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()

        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.particles = []
        
        self._reset_level(full_reset=True)
        
        return self._get_observation(), self._get_info()

    def _reset_level(self, full_reset=False):
        """Resets the ball and paddle. If full_reset, also resets bricks."""
        self.game_phase = "aiming"
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        # Place ball on a virtual launcher above the playfield
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, 50)
        self.ball_vel = pygame.Vector2(0, 0)
        self.launcher_angle = 0.0

        if full_reset:
            self._create_bricks()

    def _create_bricks(self):
        self.bricks = []
        brick_width, brick_height = 50, 20
        rows, cols = 6, 11
        x_gap, y_gap = 5, 5
        total_grid_width = cols * (brick_width + x_gap) - x_gap
        start_x = (self.WIDTH - total_grid_width) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                color = self.rng.choice(self.BRICK_COLORS)
                brick = pygame.Rect(
                    start_x + c * (brick_width + x_gap),
                    start_y + r * (brick_height + y_gap),
                    brick_width,
                    brick_height,
                )
                self.bricks.append({"rect": brick, "color": color})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        reward = -0.01  # Small penalty for time to encourage efficiency

        # --- Game Logic ---
        if self.game_phase == "aiming":
            self._handle_aiming(movement, space_pressed)
        elif self.game_phase == "in_play":
            event_reward = self._handle_in_play(movement)
            reward += event_reward

        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if len(self.bricks) == 0:
            reward += 50  # Big reward for winning
            terminated = True
            # // Sound effect: Game Win
        elif self.lives <= 0:
            terminated = True
            # // Sound effect: Game Over
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_aiming(self, movement, space_pressed):
        # Adjust launcher angle
        if movement == 3:  # Left
            self.launcher_angle -= 0.05
        elif movement == 4:  # Right
            self.launcher_angle += 0.05
        self.launcher_angle = np.clip(self.launcher_angle, -math.pi / 2.2, math.pi / 2.2)

        if space_pressed:
            self.game_phase = "in_play"
            self.ball_vel = pygame.Vector2(
                math.sin(self.launcher_angle),
                math.cos(self.launcher_angle)
            ) * self.BALL_SPEED
            # // Sound effect: Launch

    def _handle_in_play(self, movement):
        # Move paddle
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # Move ball
        self.ball_pos += self.ball_vel

        # --- Collision Detection ---
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        # Walls
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # // Sound effect: Wall bounce
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # // Sound effect: Wall bounce

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            # Add horizontal influence based on hit location
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2.0
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_SPEED
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            # // Sound effect: Paddle bounce

        # Bricks
        step_reward = 0
        for brick_data in self.bricks[:]:
            brick = brick_data["rect"]
            if ball_rect.colliderect(brick):
                # // Sound effect: Brick break
                step_reward += 1
                self.score += 100

                # Determine bounce direction
                clip_rect = ball_rect.clip(brick)
                if clip_rect.width < clip_rect.height:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                
                self._create_particles_at(brick.center, brick_data["color"])
                self.bricks.remove(brick_data)
                break # Only break one brick per frame

        # Bottom wall (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            step_reward -= 10 # Penalty for losing a life
            if self.lives > 0:
                self._reset_level(full_reset=False)
                # // Sound effect: Lose life
        
        return step_reward

    def _create_particles_at(self, pos, color):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.rng.randint(15, 30)
            self.particles.append(Particle(pos, vel, color, lifespan))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw bricks
        for brick_data in self.bricks:
            pygame.draw.rect(self.screen, brick_data["color"], brick_data["rect"], border_radius=3)
            # Add a slight 3D effect
            darker_color = [max(0, c-40) for c in brick_data["color"]]
            pygame.draw.rect(self.screen, darker_color, brick_data["rect"], 2, border_radius=3)

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw ball
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW + (80,))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw aiming UI
        if self.game_phase == 'aiming':
            launcher_start = (self.WIDTH / 2, 0)
            length = 50
            launcher_end = (
                launcher_start[0] + length * math.sin(self.launcher_angle),
                launcher_start[1] + length * math.cos(self.launcher_angle)
            )
            pygame.draw.line(self.screen, self.COLOR_PADDLE, launcher_start, launcher_end, 2)
            pygame.draw.circle(self.screen, self.COLOR_PADDLE, (int(launcher_start[0]), int(launcher_start[1])), 5)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win Text
        if self.game_over:
            if len(self.bricks) == 0:
                end_text_str = "YOU WIN!"
            else:
                end_text_str = "GAME OVER"
            
            end_text = self.font_main.render(end_text_str, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_remaining": len(self.bricks)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    # Set the video driver to dummy to run headless
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # To visualize the game, you would need a different setup
    # This example just runs a few random steps
    obs, info = env.reset()
    print("Initial Info:", info)
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Episode finished.")
            print("Final Info:", info)
            obs, info = env.reset()
            print("Reset Info:", info)

    env.close()