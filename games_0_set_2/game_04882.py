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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade game. Control the paddle to bounce a ball and destroy all the blocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (100, 110, 130)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = {
            10: (0, 200, 100),   # Green
            20: (0, 150, 255),   # Blue
            30: (255, 80, 80),    # Red
        }

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.combo = 0
        self.last_hit_was_block = False
        self.game_over = False

        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_radius = 0
        self.blocks = []
        self.particles = []

        # Initialize state variables, required for the first observation
        # self.reset() is not called in __init__ to allow for proper seeding.
        # It will be called by the environment wrapper.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.last_hit_was_block = False
        self.game_over = False

        # Paddle
        paddle_width, paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.WIDTH - paddle_width) // 2,
            self.HEIGHT - paddle_height - 10,
            paddle_width,
            paddle_height
        )
        self.paddle_speed = 12

        # Ball
        self.ball_radius = 7
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.ball_radius - 1], dtype=np.float64)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards angle
        ball_speed = 6
        self.ball_vel = np.array([math.cos(angle) * ball_speed, math.sin(angle) * ball_speed], dtype=np.float64)

        # Blocks
        self.blocks = []
        block_rows, block_cols = 4, 5
        block_width, block_height = 80, 20
        total_block_width = block_cols * block_width + (block_cols - 1) * 10
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 50
        point_values = [30, 30, 20, 20, 10]

        for i in range(block_rows):
            for j in range(block_cols):
                points = point_values[j]
                color = self.BLOCK_COLORS[points]
                rect = pygame.Rect(
                    start_x + j * (block_width + 10),
                    start_y + i * (block_height + 10),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": rect, "points": points, "color": color})

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # 1. Handle Input
        movement = action[0]
        if movement in [3, 4]: # Left or Right movement
            reward -= 0.02
            if movement == 3: # Left
                self.paddle.x -= self.paddle_speed
            elif movement == 4: # Right
                self.paddle.x += self.paddle_speed

        # Clamp paddle to screen
        self.paddle.x = max(10, min(self.WIDTH - self.paddle.width - 10, self.paddle.x))

        # 2. Update Game State
        self.ball_pos += self.ball_vel

        # 3. Handle Collisions
        # Ball vs Walls
        if self.ball_pos[0] - self.ball_radius < 10 or self.ball_pos[0] + self.ball_radius > self.WIDTH - 10:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], 10 + self.ball_radius, self.WIDTH - 10 - self.ball_radius)
            self._reset_combo()
        if self.ball_pos[1] - self.ball_radius < 10:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = 10 + self.ball_radius
            self._reset_combo()

        # Ball vs Paddle
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_radius, self.ball_pos[1] - self.ball_radius, self.ball_radius*2, self.ball_radius*2)
        if self.paddle.colliderect(ball_rect) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.paddle.top - self.ball_radius

            # Change horizontal velocity based on hit location
            offset = (self.ball_pos[0] - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel[0] = offset * 8.0

            # Normalize velocity to maintain speed
            current_speed = np.linalg.norm(self.ball_vel)
            if current_speed > 0:
                self.ball_vel = (self.ball_vel / current_speed) * 8.0

            self._reset_combo()

        # Ball vs Blocks
        hit_a_block = False
        for block in self.blocks[:]:
            if block["rect"].colliderect(ball_rect):
                hit_a_block = True

                # Collision response
                self._handle_block_collision(ball_rect, block["rect"])

                # Rewards and score
                reward += 0.1 # Continuous feedback for hitting
                reward += block["points"] / 10.0 # Event-based reward
                self.score += block["points"]

                # Combo logic
                if self.last_hit_was_block:
                    self.combo += 1
                if self.combo > 1:
                    reward += (self.combo - 1)
                    self.score += (self.combo - 1) * 5 # Bonus score for combos
                self.last_hit_was_block = True

                # Visuals and state change
                self._spawn_particles(block["rect"].center, block["color"])
                self.blocks.remove(block)
                break # Only handle one block collision per frame

        if not hit_a_block and not self.paddle.colliderect(ball_rect):
            self.last_hit_was_block = False

        # Ball out of bounds (bottom)
        if self.ball_pos[1] - self.ball_radius > self.HEIGHT:
            self.lives -= 1
            reward -= 10
            self._reset_combo()
            if self.lives > 0:
                # Reset ball position
                self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.ball_radius - 1], dtype=np.float64)
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = np.array([math.cos(angle) * 6.0, math.sin(angle) * 6.0], dtype=np.float64)
            else:
                self.game_over = True

        # 4. Update Particles
        self._update_particles()

        # 5. Check Termination
        self.steps += 1
        terminated = self._check_termination()
        if not self.game_over and terminated:
            self.game_over = True
            if not self.blocks: # Win condition
                reward += 100
                self.score += 1000 # Win bonus

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_block_collision(self, ball_rect, block_rect):
        # Determine collision side to correctly reverse velocity
        overlap = ball_rect.clip(block_rect)
        if overlap.width < overlap.height:
            # Horizontal collision
            self.ball_vel[0] *= -1
        else:
            # Vertical collision
            self.ball_vel[1] *= -1

    def _reset_combo(self):
        self.combo = 0
        self.last_hit_was_block = False

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({
                "pos": list(pos), "vel": vel, "radius": radius,
                "color": color, "lifetime": lifetime
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["radius"] *= 0.95
            if p["lifetime"] <= 0 or p["radius"] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks

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
        # Walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, 10, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - 10, 0, 10, self.HEIGHT))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            # Add a subtle inner highlight for depth
            highlight_color = tuple(min(255, c + 30) for c in block["color"])
            pygame.draw.rect(self.screen, highlight_color, block["rect"].inflate(-6, -6), 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=4)

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            if radius > 0:
                # Create a temporary surface for alpha blending
                alpha = max(0, min(255, int(255 * (p["lifetime"] / 15))))
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p["color"], alpha), (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius))


        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 20, 20))

        # Combo
        if self.combo > 1:
            combo_text = self.font_main.render(f"COMBO x{self.combo}", True, self.COLOR_BALL)
            text_rect = combo_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 50))
            self.screen.blit(combo_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "combo": self.combo,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # To play the game manually, un-comment the line below
    # os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # The display for human play is not part of the environment
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Breakout")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()
    pygame.quit()