import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all the blocks to win, but lose the ball 3 times and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 3.0
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_GRID = (35, 35, 60)
        self.COLOR_PADDLE = (200, 255, 255)
        self.COLOR_PADDLE_OUTLINE = (100, 200, 200)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 200, 0)
        self.COLOR_UI = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 80, 120), (255, 180, 50), (80, 220, 120),
            (80, 150, 255), (200, 100, 255)
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # The environment must be runnable headless
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_left = None
        self.blocks_destroyed_count = None
        self.current_ball_speed = None
        
        # self.reset() is called here to ensure all state variables are initialized
        # before any other method is called.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.blocks_destroyed_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2, 
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH, self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._generate_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Time penalty

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Update game logic based on action
        self._handle_input(movement, space_held)
        
        # Update game state (ball, collisions)
        event_reward = self._update_game_state()
        reward += event_reward

        self._update_particles()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            if len(self.blocks) == 0:
                reward += 100 # Win bonus
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = max(0, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH))

        # Ball Launch
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # sfx: launch_ball
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = [math.cos(angle) * self.current_ball_speed, 
                             math.sin(angle) * self.current_ball_speed]

    def _update_game_state(self):
        reward = 0
        if not self.ball_launched:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            return reward

        # Risky play reward
        if self.ball_vel[1] > 0 and (self.paddle.left < 10 or self.paddle.right > self.WIDTH - 10):
            reward += 0.1

        # Move ball
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sfx: wall_bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # sfx: wall_bounce

        # Bottom wall (lose ball)
        if self.ball.top >= self.HEIGHT:
            self.balls_left -= 1
            reward -= 10
            # sfx: lose_life
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return reward

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Influence horizontal direction based on impact point
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = offset * self.current_ball_speed
            
            # Normalize to maintain constant speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed

        # Block collisions
        hit_block_idx = self.ball.collidelist(self.blocks)
        if hit_block_idx != -1:
            block = self.blocks.pop(hit_block_idx)
            block_color = self.BLOCK_COLORS[hit_block_idx % len(self.BLOCK_COLORS)]
            self._create_particles(block.center, block_color)
            # sfx: block_break
            
            reward += 1
            self.score += 10
            self.blocks_destroyed_count += 1

            # Increase speed every 20 blocks
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 20 == 0:
                self.current_ball_speed += 0.5
            
            # Determine bounce direction
            prev_ball_center = (self.ball.centerx - self.ball_vel[0], self.ball.centery - self.ball_vel[1])
            
            # Simple collision resolution: check which side was closer
            # to the block's center before the collision
            dx = abs(prev_ball_center[0] - block.centerx)
            dy = abs(prev_ball_center[1] - block.centery)
            
            if dx / block.width > dy / block.height:
                self.ball_vel[0] *= -1
            else:
                self.ball_vel[1] *= -1
        
        return reward

    def _check_termination(self):
        return self.balls_left <= 0 or len(self.blocks) == 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for i, block in enumerate(self.blocks):
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            pygame.draw.rect(self.screen, color, block)
            pygame.draw.rect(self.screen, self.COLOR_BG, block, 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, 2, border_radius=3)

        # Ball
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_BALL_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (int(self.ball.centerx - glow_radius), int(self.ball.centery - glow_radius)))
        
        # Ball itself (anti-aliased)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            size = max(0, int(p['size'] * (p['life'] / p['max_life'])))
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:05d}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 5))

        # Balls left
        for i in range(self.balls_left - 1):
             pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 20 - i * 25, 25, 8, self.COLOR_PADDLE)
             pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 20 - i * 25, 25, 8, self.COLOR_PADDLE_OUTLINE)

        # Blocks left
        blocks_left_text = self.font_small.render(f"Blocks: {len(self.blocks)}", True, self.COLOR_UI)
        text_rect = blocks_left_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 15))
        self.screen.blit(blocks_left_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = len(self.blocks) == 0
            end_text_str = "YOU WIN!" if win_condition else "GAME OVER"
            end_text_color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            end_text = self.font_large.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _reset_ball(self):
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]
        self.ball_launched = False

    def _generate_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        num_cols = 10
        num_rows = 10
        total_block_width = num_cols * (block_width + 4) - 4
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 50

        for row in range(num_rows):
            for col in range(num_cols):
                block_x = start_x + col * (block_width + 4)
                block_y = start_y + row * (block_height + 4)
                self.blocks.append(pygame.Rect(block_x, block_y, block_width, block_height))

    def _create_particles(self, position, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(2, 5),
                'color': color
            })
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # To run with a display window, comment out the next line
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # To display the game in a window, we need to create a new screen
    # because the environment's screen is a dummy surface.
    if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
        pygame.display.set_caption("Block Breaker")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    else:
        screen = None # No display
    
    obs, info = env.reset()
    done = False
    
    action = env.action_space.sample()
    action[0] = 0 # No movement initially
    action[1] = 0 # Not pressing space initially
    
    running = True
    while running:
        if screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            
            # Reset action
            action = [0, 0, 0]

            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1

            if keys[pygame.K_r]: # Press R to reset
                 obs, info = env.reset()
                 done = False
                 continue
            # --- End Human Controls ---
        else: # In headless mode, run for a fixed number of steps
            if env.steps > 2000:
                running = False
            action = env.action_space.sample()


        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            obs, info = env.reset()
            done = False


        if screen:
            # Display the observation
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Control the frame rate
        env.clock.tick(60)

    env.close()