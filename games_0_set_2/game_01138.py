
# Generated: 2025-08-27T16:09:40.841933
# Source Brief: brief_01138.md
# Brief Index: 1138

        
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
        "A fast-paced, top-down block breaker where fast plays are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 8
    BALL_SPEED_INITIAL = 5.0
    MAX_STEPS = 2000
    INITIAL_LIVES = 3

    # --- Colors ---
    COLOR_BG = (10, 10, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_PADDLE = (220, 220, 220)
    COLOR_BALL = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    BLOCK_COLORS = [
        (255, 50, 50), (255, 150, 50), (255, 255, 50), 
        (50, 255, 50), (50, 150, 255), (150, 50, 255)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # State variables are initialized in reset()
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._setup_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _setup_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        num_cols = 10
        num_rows = 5
        gap = 6
        total_block_width = num_cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) // 2
        start_y = 50

        for i in range(num_rows):
            for j in range(num_cols):
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                rect = pygame.Rect(
                    start_x + j * (block_width + gap),
                    start_y + i * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({'rect': rect, 'color': color})

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1

        reward = -0.01  # Small penalty for each step to encourage speed

        # --- Handle Input ---
        self._handle_input(movement, space_pressed)
        
        # --- Update Game State ---
        if not self.ball_on_paddle:
            reward += self._update_ball_and_handle_collisions()

        self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if len(self.blocks) == 0:
            reward += 100  # Win bonus
            self.game_over = True
            terminated = True
        elif self.lives <= 0:
            reward -= 100  # Loss penalty
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.clamp_ip(self.screen.get_rect())

        # Ball Launch
        if self.ball_on_paddle and space_pressed:
            self.ball_on_paddle = False
            # sfx: launch_ball
            launch_angle = (self.np_random.random() - 0.5) * (math.pi / 4)
            self.ball_vel = [
                self.BALL_SPEED_INITIAL * math.sin(launch_angle),
                -self.BALL_SPEED_INITIAL * math.cos(launch_angle)
            ]
        
        if self.ball_on_paddle:
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]

    def _update_ball_and_handle_collisions(self):
        # --- Update Ball Position ---
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )
        
        step_reward = 0

        # --- Wall Collisions ---
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce
        
        # --- Life Loss ---
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            step_reward -= 10
            # sfx: lose_life
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
            return step_reward

        # --- Paddle Collision ---
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            step_reward += 0.1
            # sfx: paddle_hit
            
            offset = ball_rect.centerx - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            
            # Max bounce angle is ~60 degrees (math.pi / 3)
            bounce_angle = normalized_offset * (math.pi / 3)
            
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            self.ball_vel[0] = speed * math.sin(bounce_angle)
            self.ball_vel[1] = -speed * math.cos(bounce_angle)
            
            # Ensure ball is above paddle to prevent getting stuck
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # --- Block Collisions ---
        collided_block_index = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if collided_block_index != -1:
            block_data = self.blocks[collided_block_index]
            block_rect = block_data['rect']
            
            # sfx: block_break
            self._spawn_particles(block_rect.center, block_data['color'])
            
            # Determine collision side to correctly flip velocity
            prev_ball_rect = pygame.Rect(
                (self.ball_pos[0] - self.ball_vel[0]) - self.BALL_RADIUS,
                (self.ball_pos[1] - self.ball_vel[1]) - self.BALL_RADIUS,
                self.BALL_RADIUS * 2, self.BALL_RADIUS * 2
            )
            
            if prev_ball_rect.right <= block_rect.left or prev_ball_rect.left >= block_rect.right:
                 self.ball_vel[0] *= -1
            else:
                 self.ball_vel[1] *= -1
            
            self.blocks.pop(collided_block_index)
            self.score += 1
            step_reward += 1

        return step_reward

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 3
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = 20 + self.np_random.integers(0, 11)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render all game elements ---
        self._render_game()
        
        # --- Render UI overlay ---
        self._render_ui()
        
        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block_data in self.blocks:
            rect = block_data['rect']
            color = block_data['color']
            darker_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, darker_color, rect, width=2, border_radius=3)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 12)))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, p['life'] // 5)
            # Create a temporary surface for alpha blending
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color_with_alpha, (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'][0]-size), int(p['pos'][1]-size)))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, (150, 150, 150), self.paddle, width=2, border_radius=3)

        # Ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 160, 10))
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 80 + i * 25, 19, 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 80 + i * 25, 19, 8, self.COLOR_BALL)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if len(self.blocks) == 0:
                end_text = self.font_game_over.render("YOU WIN!", True, (100, 255, 100))
            else:
                end_text = self.font_game_over.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("      BLOCK BREAKER")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), so we need to transpose the obs from (h, w, c)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling (for closing the window) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Limit to 30 FPS for smooth human play

    print("="*30)
    print(f"Final Score: {info['score']}")
    print(f"Steps: {info['steps']}")
    print("="*30)

    env.close()