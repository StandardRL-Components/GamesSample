
# Generated: 2025-08-28T05:02:38.815604
# Source Brief: brief_05448.md
# Brief Index: 5448

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
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
        "A fast-paced, top-down block breaker. Clear all blocks across 3 stages to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 5.0
        self.MAX_BALL_DX_FACTOR = 1.5

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 50, 50), (50, 255, 50), (50, 150, 255), 
            (255, 120, 50), (200, 50, 255), (50, 255, 200)
        ]

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
        self.font_main = pygame.font.SysFont('monospace', 20, bold=True)
        self.font_info = pygame.font.SysFont('monospace', 16)
        
        # Define block layouts for 3 stages
        self._define_block_layouts()
        
        # Initialize state variables
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_speed = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.balls_left = None
        self.stage = None
        self.consecutive_wall_hits = None
        
        self.reset()

        self.validate_implementation()
    
    def _define_block_layouts(self):
        self.block_layouts = []
        # Stage 1: Simple rectangle
        layout1 = []
        for r in range(5):
            for c in range(10):
                layout1.append((c, r))
        self.block_layouts.append(layout1)

        # Stage 2: Pyramid
        layout2 = []
        for r in range(6):
            for c in range(r, 10 - r):
                if c < 10:
                    layout2.append((c, r))
        self.block_layouts.append(layout2)

        # Stage 3: Hollow center
        layout3 = []
        for r in range(6):
            for c in range(10):
                if not (2 <= r <= 3 and 3 <= c <= 6):
                    layout3.append((c, r))
        self.block_layouts.append(layout3)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.stage = 0 
        self.particles = []
        
        self._setup_stage(1)
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self, stage_num):
        self.stage = stage_num
        self.ball_launched = False
        self.consecutive_wall_hits = 0
        
        # Set ball speed based on stage
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * 0.5
        
        # Reset paddle
        paddle_x = (self.WIDTH - self.PADDLE_WIDTH) / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 10
        self.paddle = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Reset ball attached to paddle
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]

        # Create blocks for the stage
        self.blocks = []
        layout = self.block_layouts[self.stage - 1]
        block_w, block_h = 60, 20
        start_x, start_y = 20, 50
        for c, r in layout:
            color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
            block = pygame.Rect(start_x + c * (block_w + 4), start_y + r * (block_h + 4), block_w, block_h)
            self.blocks.append({'rect': block, 'color': color})
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # Handle player input
        self._handle_input(movement, space_held)

        # Update game logic
        reward += self._update_game_state()
        
        terminated = self._check_termination()

        if terminated and not self.game_over: # Max steps reached
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle movement
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # sfx: ball_launch
            launch_angle = (self.np_random.random() * 0.6 - 0.3) * math.pi # -54 to +54 degrees
            self.ball_vel = [math.sin(launch_angle) * self.ball_speed, -math.cos(launch_angle) * self.ball_speed]

    def _update_game_state(self):
        reward = 0

        if not self.ball_launched:
            # Ball follows paddle
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
        else:
            reward -= 0.001 # Small penalty for time passing

            # Move ball
            self.ball.x += self.ball_vel[0]
            self.ball.y += self.ball_vel[1]

            # --- Collision Detection ---
            # Walls
            if self.ball.left < 0 or self.ball.right > self.WIDTH:
                self.ball_vel[0] *= -1
                self.ball.x = max(0, min(self.WIDTH - self.ball.width, self.ball.x))
                self.consecutive_wall_hits += 1
                # sfx: wall_bounce
            if self.ball.top < 0:
                self.ball_vel[1] *= -1
                self.ball.y = max(0, self.ball.y)
                self.consecutive_wall_hits += 1
                # sfx: wall_bounce

            # Paddle
            if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
                self.ball.bottom = self.paddle.top
                self.ball_vel[1] *= -1
                # sfx: paddle_bounce
                
                # Influence ball direction based on hit location
                offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
                self.ball_vel[0] = offset * self.ball_speed * self.MAX_BALL_DX_FACTOR
                # Renormalize to maintain speed
                current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
                if current_speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                    self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed
                
                self.consecutive_wall_hits = 0

            # Blocks
            hit_block = None
            for block_data in self.blocks:
                if self.ball.colliderect(block_data['rect']):
                    hit_block = block_data
                    break
            
            if hit_block:
                # sfx: block_break
                self.blocks.remove(hit_block)
                self.score += 10
                reward += 1.0

                # Create particles
                self._create_particles(hit_block['rect'].center, hit_block['color'])
                
                # Bounce logic
                self.ball_vel[1] *= -1
                self.consecutive_wall_hits = 0

            # Ball lost
            if self.ball.top > self.HEIGHT:
                # sfx: ball_lost
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._setup_stage(self.stage) # Resets ball on paddle
                else:
                    self.game_over = True
                    reward -= 100.0
            
            # Anti-softlock mechanism
            if self.consecutive_wall_hits > 50:
                self.ball.centerx = self.paddle.centerx
                self.ball.bottom = self.paddle.top - 20
                self.ball_vel = [self.np_random.uniform(-0.5, 0.5) * self.ball_speed, -0.8 * self.ball_speed]
                self.consecutive_wall_hits = 0

        # Update particles
        self._update_particles()
        
        # Check for stage clear
        if self.ball_launched and not self.blocks and not self.game_over:
            # sfx: stage_clear
            self.score += 100
            reward += 5.0
            if self.stage < 3:
                self._setup_stage(self.stage + 1)
            else: # Game won
                # sfx: game_win
                self.game_over = True
                reward += 100.0

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            p_vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
            p_lifetime = self.np_random.integers(15, 30)
            p_size = self.np_random.integers(2, 5)
            self.particles.append({'pos': list(pos), 'vel': p_vel, 'lifetime': p_lifetime, 'color': color, 'size': p_size})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 30.0))))
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            s.fill((*p['color'], alpha))
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw blocks
        for block_data in self.blocks:
            pygame.draw.rect(self.screen, block_data['color'], block_data['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block_data['color']), block_data['rect'], 1)

        # Draw paddle with glow
        glow_rect = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        pos = (int(self.ball.centerx), int(self.ball.centery))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL, 60))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
    
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        ball_text = self.font_main.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(ball_text, (self.WIDTH - 150, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 70 + i * 20, 20, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 70 + i * 20, 20, 6, self.COLOR_BALL)

        # Stage
        stage_text = self.font_main.render(f"STAGE {self.stage}", True, self.COLOR_TEXT)
        text_rect = stage_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(stage_text, text_rect)

        # Launch prompt
        if not self.ball_launched and not self.game_over:
            launch_text = self.font_info.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            text_rect = launch_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 50))
            self.screen.blit(launch_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "stage": self.stage,
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

# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run the game with manual controls ---
    # This part requires a display.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Map keyboard keys to actions
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation from the environment to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

    finally:
        env.close()
        print(f"Game Over. Final Score: {env.score}")