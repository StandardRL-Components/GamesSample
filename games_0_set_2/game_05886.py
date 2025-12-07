
# Generated: 2025-08-28T06:23:37.954089
# Source Brief: brief_05886.md
# Brief Index: 5886

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated block-breaking game where risk-taking is rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 3.0

    # --- Colors ---
    COLOR_BG_TOP = (15, 20, 35)
    COLOR_BG_BOTTOM = (35, 40, 60)
    COLOR_PADDLE = (200, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_BORDER = (80, 90, 120)

    BLOCK_COLORS = {
        1: (50, 205, 50),   # Green
        3: (65, 105, 225),  # Blue
        5: (220, 20, 60),   # Red
        7: (255, 215, 0),   # Yellow
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = None
        self.particles = None
        self.score = None
        self.balls_left = None
        self.steps = None
        self.game_over = None
        self.ball_speed = None
        self.blocks_destroyed_count = None
        self.np_random = None

        self.reset()

        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.ball_launched = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.blocks_destroyed_count = 0

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

        self.blocks = self._generate_blocks()
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        
        # --- Handle Input ---
        old_paddle_center_x = self.paddle.centerx
        
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Reward for paddle movement
        if self.ball_launched and self.ball_vel[1] > 0: # Ball moving towards paddle
            projected_impact_x = self._project_ball_impact_x()
            old_dist = abs(projected_impact_x - old_paddle_center_x)
            new_dist = abs(projected_impact_x - self.paddle.centerx)
            if new_dist > old_dist:
                reward -= 0.01 # Penalize moving away from projected impact

        # Launch ball
        if not self.ball_launched and space_held:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 0.7, -math.pi * 0.3) # Upwards angle
            self.ball_vel = [self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)]
            # SFX: Ball Launch

        # --- Update Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if len(self.blocks) == 0:
                reward += 50 # Win bonus
            elif self.steps >= self.MAX_STEPS:
                pass # No bonus/penalty for timeout
            # Ball loss penalty is handled in _handle_collisions

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _project_ball_impact_x(self):
        # Simplified projection assuming no side wall bounces
        if self.ball_vel[1] <= 0:
            return self.ball_pos[0]
        time_to_impact = (self.paddle.top - self.ball_pos[1]) / self.ball_vel[1]
        return self.ball_pos[0] + self.ball_vel[0] * time_to_impact

    def _update_ball(self):
        if not self.ball_launched:
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        else:
            self.ball_pos[0] += self.ball_vel[0]
            self.ball_pos[1] += self.ball_vel[1]

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # SFX: Wall Bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # SFX: Wall Bounce

        # Bottom wall (lose ball)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.balls_left -= 1
            self.ball_launched = False
            reward -= 20
            # SFX: Lose Ball
            if self.balls_left > 0:
                self._create_particles(self.paddle.centerx, self.paddle.centery, self.COLOR_PADDLE, 20)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            self.ball_vel[1] *= -1
            
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.ball_speed * offset * 1.5
            
            # Normalize and rescale to maintain consistent speed
            current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed
            # SFX: Paddle Hit

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                reward += 0.1 # Base hit reward
                reward += block['points']
                self.score += block['points']
                self.blocks.remove(block)
                self._create_particles(block['rect'].centerx, block['rect'].centery, block['color'], 30)
                # SFX: Block Break

                # Reverse velocity
                # A simple check to see if the collision was more horizontal or vertical
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                # Difficulty scaling
                self.blocks_destroyed_count += 1
                if self.blocks_destroyed_count % 5 == 0:
                    self.ball_speed = min(self.INITIAL_BALL_SPEED * 2.5, self.ball_speed + 0.2)

                break # Only handle one block collision per frame
        
        return reward

    def _check_termination(self):
        return self.balls_left <= 0 or len(self.blocks) == 0 or self.steps >= self.MAX_STEPS

    def _generate_blocks(self):
        blocks = []
        rows = self.np_random.integers(4, 7)
        cols = self.np_random.integers(8, 12)
        block_width = self.SCREEN_WIDTH / cols
        block_height = 20
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() < 0.85: # 85% chance of a block spawning
                    points = self.np_random.choice(list(self.BLOCK_COLORS.keys()))
                    color = self.BLOCK_COLORS[points]
                    
                    rect = pygame.Rect(
                        c * block_width,
                        r * block_height + 50,
                        block_width - 1,
                        block_height - 1
                    )
                    blocks.append({'rect': rect, 'points': points, 'color': color})
        return blocks

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'lifespan': lifespan, 'color': color})
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # --- Render Background ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30))))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            size = max(1, int(p['lifespan'] / 6))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BORDER, block['rect'], 1)

        # Render paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, self.paddle, 1, border_radius=3)
        
        # Render ball
        pos_x, pos_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        balls_text = self.font_small.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.balls_left - (0 if self.ball_launched else 1)):
            pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH - 80 + i * 20, 18, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH - 80 + i * 20, 18, 6, self.COLOR_BALL)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "LEVEL CLEAR" if len(self.blocks) == 0 else "GAME OVER"
            text_surface = self.font_large.render(win_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    env.validate_implementation()
    
    # To visualize, you would need to remove the dummy videodriver and use a different main loop
    # For example:
    # del os.environ["SDL_VIDEODRIVER"]
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # pygame.display.set_caption("Block Breaker")
    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()
    # while not done:
    #     action = env.action_space.sample() # Replace with actual controls
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #
    #     # Convert obs back to a Pygame surface to display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #     clock.tick(30)
    # env.close()