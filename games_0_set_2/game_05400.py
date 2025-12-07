# Generated: 2025-08-28T04:52:38.181558
# Source Brief: brief_05400.md
# Brief Index: 5400

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade block breaker. Bounce the ball to destroy all the blocks before you run out of lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time

        # Colors (Neon/Retro Theme)
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_PADDLE_GLOW = (0, 200, 255)
        self.COLOR_BALL = (255, 0, 128)
        self.COLOR_BALL_GLOW = (255, 100, 200)
        self.BLOCK_COLORS = [
            (0, 255, 255), (0, 255, 128), (255, 255, 0), (255, 128, 0)
        ]
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER = (255, 0, 0)

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12

        # Ball settings
        self.BALL_RADIUS = 8
        self.BALL_SPEED_INITIAL = 7

        # Block settings
        self.NUM_BLOCK_COLS = 15
        self.NUM_BLOCK_ROWS = 5
        self.BLOCK_WIDTH = self.WIDTH // self.NUM_BLOCK_COLS
        self.BLOCK_HEIGHT = 18
        self.BLOCK_SPACING = 2
        self.BLOCK_START_Y = 50

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_held = False
        self.blocks = []
        self.particles = []
        self.last_hit_was_block = False

        # self.reset() is called by the wrapper/test, no need to call it here.
        # self.validate_implementation() is for debugging, not needed in final version.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.balls_left = 3
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT * 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.ball_held = True
        self._reset_ball()

        self.blocks = self._create_blocks()
        self.particles = []
        self.last_hit_was_block = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.steps += 1
        reward = -0.01  # Small penalty for each step

        if not self.game_over:
            # Unpack action
            movement = action[0]
            space_pressed = action[1] == 1

            # --- Handle Input ---
            if movement == 3:  # Left
                self.paddle.x -= self.PADDLE_SPEED
            elif movement == 4:  # Right
                self.paddle.x += self.PADDLE_SPEED
            
            # Clamp paddle to screen
            self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

            # Launch ball
            if self.ball_held and space_pressed:
                self.ball_held = False
                # Sound: Ball Launch
                angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL

            # --- Update Game Logic ---
            if self.ball_held:
                self._reset_ball()
            else:
                step_reward = self._update_ball()
                reward += step_reward
            
            self._update_particles()
            
            # --- Calculate Reward and Termination ---
            # Reward is handled inside _update_ball on collisions
            # Here we just check for termination conditions
            terminated = self.game_over
            if len(self.blocks) == 0:
                self.win = True
                terminated = True
                reward += 100 # Win bonus
            
            truncated = self.steps >= self.MAX_STEPS

            # If terminated due to win/loss, set game_over flag
            if terminated and not self.game_over:
                self.game_over = True
        else: # If game is already over
            terminated = True
            truncated = False
            reward = 0
            self._update_particles() # Keep particles animating

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_ball(self):
        reward = 0
        # Move ball
        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball_pos.x))
            self._create_particles(self.ball_pos, self.COLOR_BALL, 5)
            # Sound: Wall Bounce
        
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(self.BALL_RADIUS, self.ball_pos.y)
            self._create_particles(self.ball_pos, self.COLOR_BALL, 5)
            # Sound: Wall Bounce

        # Bottom wall (lose ball)
        if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
            self.balls_left -= 1
            self.ball_held = True
            self._reset_ball()
            if self.balls_left <= 0:
                self.game_over = True
                reward -= 100 # Large negative reward for losing
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle):
            # Sound: Paddle Hit
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
            
            # Change horizontal velocity based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * 2
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.BALL_SPEED_INITIAL
            
            self._create_particles(self.ball_pos, self.COLOR_PADDLE_GLOW, 15)
            self.last_hit_was_block = False

        # Block collisions
        hit_a_block = False
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                hit_a_block = True
                # Sound: Block Break
                self._handle_block_collision(ball_rect, block)
                self.blocks.remove(block)
                self.score += 1
                self._create_particles(pygame.Vector2(block["rect"].center), block["color"], 20)
                break # Only handle one block collision per frame
        
        if hit_a_block:
            if not self.last_hit_was_block:
                # This is the first block hit in a combo
                reward += 1.1 # +1 for block break, +0.1 for hit
            else:
                # This is a combo hit
                reward += 1.1 # Same reward for now, could be increased
        self.last_hit_was_block = hit_a_block

        # Anti-softlock: if ball gets stuck horizontally
        if abs(self.ball_vel.y) < 0.2:
            self.ball_vel.y = 0.5 * (-1 if self.ball_vel.y < 0 else 1)

        return reward

    def _handle_block_collision(self, ball_rect, block):
        block_rect = block["rect"]
        
        # Calculate overlap
        dx = (ball_rect.centerx - block_rect.centerx) / block_rect.width
        dy = (ball_rect.centery - block_rect.centery) / block_rect.height
        
        # Reverse velocity based on the side with less penetration
        if abs(dx) > abs(dy):
            self.ball_vel.x *= -1
            # Nudge ball out of collision
            self.ball_pos.x += self.ball_vel.x
        else:
            self.ball_vel.y *= -1
            # Nudge ball out of collision
            self.ball_pos.y += self.ball_vel.y

    def _create_blocks(self):
        blocks = []
        total_block_width = self.NUM_BLOCK_COLS * self.BLOCK_WIDTH
        start_x = (self.WIDTH - total_block_width) / 2
        for i in range(self.NUM_BLOCK_ROWS):
            for j in range(self.NUM_BLOCK_COLS):
                x = start_x + j * self.BLOCK_WIDTH + self.BLOCK_SPACING / 2
                y = self.BLOCK_START_Y + i * self.BLOCK_HEIGHT + self.BLOCK_SPACING / 2
                rect = pygame.Rect(
                    x, y, 
                    self.BLOCK_WIDTH - self.BLOCK_SPACING, 
                    self.BLOCK_HEIGHT - self.BLOCK_SPACING
                )
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                blocks.append({"rect": rect, "color": color})
        # Randomly remove some blocks to ensure 75
        num_to_remove = (self.NUM_BLOCK_COLS * self.NUM_BLOCK_ROWS) - 75
        for _ in range(num_to_remove):
            if blocks:
                blocks.pop(self.np_random.integers(0, len(blocks)))
        return blocks

    def _reset_ball(self):
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            radius = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append([pygame.Vector2(pos), vel, radius, color, lifespan])
    
    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[1]  # pos += vel
            p[4] -= 1     # lifespan -= 1
            if p[4] <= 0:
                self.particles.remove(p)

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
        # Draw background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], width=2, border_radius=3)

        # Draw paddle with glow
        glow_surf = self.paddle.copy()
        glow_surf.inflate_ip(10, 10)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_surf, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

        # Draw particles
        for p in self.particles:
            pos, vel, radius, color, lifespan = p
            alpha = max(0, min(255, int(255 * (lifespan / 20))))
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), (*color, alpha))
            except TypeError: # Sometimes color might not have alpha
                pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), color)


        # Draw ball with glow
        if self.balls_left > 0 or self.win:
            for i in range(4, 0, -1):
                alpha = 80 - i * 20
                pygame.gfxdraw.filled_circle(
                    self.screen, int(self.ball_pos.x), int(self.ball_pos.y),
                    self.BALL_RADIUS + i * 2, (*self.COLOR_BALL_GLOW, alpha)
                )
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Draw score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw remaining balls
        for i in range(self.balls_left):
            x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            y = 10 + self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS // 2, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS // 2, self.COLOR_BALL)
        
        # Draw game over/win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.BLOCK_COLORS[0]
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    # Re-enable display for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)

    env.close()